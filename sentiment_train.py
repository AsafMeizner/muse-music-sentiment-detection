import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

# =============================================================================
# 0) SET UP RESULTS DIRECTORY
# =============================================================================
results_dir = "./setiment-results"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# 1) SET PATHS AND PARAMETERS
# =============================================================================
BASE_PATH = "./deam-dataset"  # adjust if needed

# Corrected dynamic annotation paths:
DYN_AROUSAL_PATH = os.path.join(
    BASE_PATH, "DEAM_Annotations", "annotations", "annotations averaged per song",
    "dynamic (per second annotations)", "arousal.csv"
)
DYN_VALENCE_PATH = os.path.join(
    BASE_PATH, "DEAM_Annotations", "annotations", "annotations averaged per song",
    "dynamic (per second annotations)", "valence.csv"
)
AUDIO_FOLDER = os.path.join(BASE_PATH, "DEAM_audio", "MEMD_audio")

SAMPLE_RATE = 22050
# Use 5-second segments
SEGMENT_LENGTH = 5.0
BIN_SIZE = 5.0  # seconds per bin

# =============================================================================
# 2) UTILITY FUNCTIONS FOR FEATURE EXTRACTION
# =============================================================================
def extract_features_from_audio(y, sr):
    """
    Extract a 193-dimensional feature vector from audio segment y.
    Features:
      - 40 MFCCs (mean over time)
      - 12 Chroma STFT (mean)
      - 128 Mel-spectrogram (in dB, mean)
      - 7 Spectral Contrast (mean)
      - 6 Tonnetz (mean from harmonic component)
    Returns: numpy array of shape (193,)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    features = np.concatenate((mfcc_mean, chroma_mean, mel_mean, spec_contrast_mean, tonnetz_mean))
    return features

def extract_segment_features(file_path, start_sec, duration=SEGMENT_LENGTH, sr=SAMPLE_RATE):
    """
    Load a segment from file_path starting at start_sec for duration seconds,
    and extract a 193-dim feature vector.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True, offset=start_sec, duration=duration)
    except Exception as e:
        print("Error loading segment from", file_path, e)
        return None
    if len(y) < int(sr * duration):
        return None
    return extract_features_from_audio(y, sr)

# =============================================================================
# 3) BUILD DYNAMIC DATASET FROM ANNOTATIONS
# =============================================================================
# Load dynamic annotation CSV files
df_dyn_arousal = pd.read_csv(DYN_AROUSAL_PATH)
df_dyn_valence = pd.read_csv(DYN_VALENCE_PATH)
df_dyn_arousal.columns = df_dyn_arousal.columns.str.strip()
df_dyn_valence.columns = df_dyn_valence.columns.str.strip()

# Merge based on song_id; suffix _val for valence and _aro for arousal.
df_dyn = pd.merge(df_dyn_valence, df_dyn_arousal, on="song_id", suffixes=("_val", "_aro"))

# Get time points from columns (assume columns are like "sample_15000ms_val")
time_cols = [col for col in df_dyn.columns if col.startswith("sample_") and col.endswith("ms_val")]
# Convert column names to seconds: "sample_15000ms_val" -> 15.0
time_points = sorted([int(col.split('_')[1].replace('ms',''))/1000.0 for col in time_cols])

# For each song, bin annotations into 5-second intervals.
X_dyn = []
Y_dyn = []

for idx, row in tqdm(df_dyn.iterrows(), total=len(df_dyn), desc="Building dynamic dataset"):
    song_id = int(row["song_id"])
    file_path = os.path.join(AUDIO_FOLDER, f"{song_id}.mp3")
    if not os.path.exists(file_path):
        continue
    t_max = max(time_points)
    # Create bins from 0 to t_max with BIN_SIZE interval
    bins = np.arange(0, t_max+BIN_SIZE, BIN_SIZE)
    for b in range(len(bins)-1):
        bin_start = bins[b]
        bin_end = bins[b+1]
        val_values = []
        aro_values = []
        for t in time_points:
            if bin_start <= t < bin_end:
                col_val = f"sample_{int(t*1000)}ms_val"
                col_aro = f"sample_{int(t*1000)}ms_aro"
                if col_val in df_dyn.columns and col_aro in df_dyn.columns:
                    val_values.append(row[col_val])
                    aro_values.append(row[col_aro])
        if len(val_values)==0 or len(aro_values)==0:
            continue
        avg_val = np.mean(val_values)
        avg_aro = np.mean(aro_values)
        # Scale labels: (x - 5)*0.05
        avg_val_scaled = (avg_val - 5.0)*0.05
        avg_aro_scaled = (avg_aro - 5.0)*0.05
        seg_features = extract_segment_features(file_path, start_sec=bin_start, duration=SEGMENT_LENGTH, sr=SAMPLE_RATE)
        if seg_features is None:
            continue
        X_dyn.append(seg_features)
        Y_dyn.append([avg_val_scaled, avg_aro_scaled])

X_dyn = np.array(X_dyn, dtype=np.float32)
Y_dyn = np.array(Y_dyn, dtype=np.float32)
print(f"\nDynamic dataset: {X_dyn.shape[0]} segments extracted.")

# =============================================================================
# 3b) Plot distribution of dynamic labels (in original [1,9] scale)
# =============================================================================
def invert_scaling(scaled_val):
    return (scaled_val / 0.05) + 5.0

Y_dyn_orig = invert_scaling(Y_dyn)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(Y_dyn_orig[:,0], bins=30, color='skyblue', edgecolor='k')
plt.title("Dynamic Valence Distribution (Original Scale)")
plt.xlabel("Valence")
plt.ylabel("Frequency")
plt.subplot(1,2,2)
plt.hist(Y_dyn_orig[:,1], bins=30, color='salmon', edgecolor='k')
plt.title("Dynamic Arousal Distribution (Original Scale)")
plt.xlabel("Arousal")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "dynamic_label_distribution.png"))
plt.show()

# =============================================================================
# 4) SCALE FEATURES & SPLIT DYNAMIC DATASET
# =============================================================================
scaler_dyn = StandardScaler()
X_dyn_scaled = scaler_dyn.fit_transform(X_dyn)

X_train, X_test, y_train, y_test = train_test_split(X_dyn_scaled, Y_dyn, test_size=0.2, random_state=42)
print("Dynamic Train set:", X_train.shape, y_train.shape)
print("Dynamic Test set:", X_test.shape, y_test.shape)

# =============================================================================
# 5) BUILD A DEEP FEED-FORWARD NETWORK FOR DYNAMIC PREDICTION
# =============================================================================
input_dim_dyn = X_train.shape[1]  # should be 193
dyn_model = models.Sequential()
dyn_model.add(layers.Input(shape=(input_dim_dyn,)))
for i in range(6):
    dyn_model.add(layers.Dense(500, activation='relu'))
    dyn_model.add(layers.Dropout(0.5))
dyn_model.add(layers.Dense(2, activation='linear'))
dyn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse',
                  metrics=['mae'])
dyn_model.summary()

# =============================================================================
# 6) SET UP CALLBACKS: EARLY STOPPING & CHECKPOINTS (DYNAMIC MODEL)
# =============================================================================
checkpoint_path_dyn = os.path.join(results_dir, "best_dyn_model.h5")
checkpoint_cb_dyn = callbacks.ModelCheckpoint(filepath=checkpoint_path_dyn,
                                              monitor='val_loss',
                                              save_best_only=True,
                                              verbose=1)
early_stop_cb_dyn = callbacks.EarlyStopping(monitor='val_loss',
                                            patience=15,
                                            restore_best_weights=True,
                                            verbose=1)

# =============================================================================
# 7) TRAIN THE DYNAMIC MODEL
# =============================================================================
history_dyn = dyn_model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=100,
                            batch_size=32,
                            callbacks=[checkpoint_cb_dyn, early_stop_cb_dyn],
                            verbose=1)

# =============================================================================
# 8) SAVE THE FINAL DYNAMIC MODEL
# =============================================================================
final_dyn_model_path = os.path.join(results_dir, "final_dyn_model.h5")
dyn_model.save(final_dyn_model_path)
print("Final dynamic model saved to:", final_dyn_model_path)

# =============================================================================
# 9) PLOT TRAINING CURVES FOR DYNAMIC MODEL
# =============================================================================
plt.figure(figsize=(8,6))
plt.plot(history_dyn.history['loss'], label='Train Loss')
plt.plot(history_dyn.history['val_loss'], label='Val Loss')
plt.title("Dynamic Model: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (scaled)")
plt.legend()
plt.savefig(os.path.join(results_dir, "dyn_loss_curve.png"))
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history_dyn.history['mae'], label='Train MAE')
plt.plot(history_dyn.history['val_mae'], label='Val MAE')
plt.title("Dynamic Model: Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (scaled)")
plt.legend()
plt.savefig(os.path.join(results_dir, "dyn_mae_curve.png"))
plt.show()

# =============================================================================
# 10) EVALUATE THE DYNAMIC MODEL
# =============================================================================
test_loss_dyn, test_mae_dyn = dyn_model.evaluate(X_test, y_test, verbose=0)
print("Dynamic Model Test MSE (loss):", test_loss_dyn)
print("Dynamic Model Test MAE:", test_mae_dyn)
preds_dyn = dyn_model.predict(X_test)
val_corr_dyn, _ = pearsonr(y_test[:, 0], preds_dyn[:, 0])
aro_corr_dyn, _ = pearsonr(y_test[:, 1], preds_dyn[:, 1])
print("Dynamic Model Valence Pearson Corr:", val_corr_dyn)
print("Dynamic Model Arousal Pearson Corr:", aro_corr_dyn)

# =============================================================================
# 11) PRINT SOME EXAMPLES: ACTUAL VS PREDICTED FOR DYNAMIC SEGMENTS
# =============================================================================
num_examples = 10
print("\n--- Examples from Dynamic Test Set ---")
for i in range(num_examples):
    true_scaled = y_test[i]
    pred_scaled = preds_dyn[i]
    true_orig = invert_scaling(true_scaled)
    pred_orig = invert_scaling(pred_scaled)
    print(f"Example {i+1}:")
    print(f"  Scaled  -> True: {true_scaled}, Predicted: {pred_scaled}")
    print(f"  Original-> True: {true_orig}, Predicted: {pred_orig}")

# =============================================================================
# 12) FUNCTION TO PREDICT DYNAMIC EMOTION THROUGH A SONG
# =============================================================================
def predict_dynamic_emotion(file_path, model, scaler_X, seg_length=SEGMENT_LENGTH, sr=SAMPLE_RATE):
    """
    Given an audio file, split it into non-overlapping seg_length segments,
    extract the 193-dim feature vector for each segment, scale it, and predict
    dynamic [valence, arousal] for each segment.
    
    Returns:
       times: array of segment start times (in seconds)
       predictions: array of shape (num_segments, 2) with scaled predictions.
    """
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return None, None
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    num_segs = int(duration // seg_length)
    times = []
    preds = []
    for i in range(num_segs):
        start = i * seg_length
        feat = extract_segment_features(file_path, start_sec=start, duration=seg_length, sr=sr)
        if feat is None:
            continue
        feat_scaled = scaler_X.transform(feat.reshape(1, -1))
        pred = model.predict(feat_scaled)[0]
        preds.append(pred)
        times.append(start)
    if len(preds) == 0:
        return None, None
    return np.array(times), np.array(preds)

# =============================================================================
# 13) PLOT DYNAMIC PREDICTIONS FOR AN EXAMPLE SONG
# =============================================================================
example_song = os.path.join(AUDIO_FOLDER, "25.mp3")  # Replace with a valid file ID
times, dyn_preds = predict_dynamic_emotion(example_song, dyn_model, scaler_dyn, seg_length=SEGMENT_LENGTH)
if times is not None:
    # Invert scaling to original [1,9]
    dyn_preds_orig = invert_scaling(dyn_preds)
    mean_pred = np.mean(dyn_preds_orig, axis=0)
    
    plt.figure(figsize=(10,5))
    plt.plot(times, dyn_preds_orig[:,0], label='Valence', marker='o')
    plt.plot(times, dyn_preds_orig[:,1], label='Arousal', marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Predicted Value [1,9]")
    plt.title("Dynamic Emotion Prediction Over Song")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "dynamic_time_series.png"))
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(dyn_preds_orig[:,0], bins=20, color='skyblue', edgecolor='k')
    plt.title("Valence Distribution (Song)")
    plt.xlabel("Valence")
    plt.subplot(1,2,2)
    plt.hist(dyn_preds_orig[:,1], bins=20, color='salmon', edgecolor='k')
    plt.title("Arousal Distribution (Song)")
    plt.xlabel("Arousal")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dynamic_prediction_distribution.png"))
    plt.show()
    
    print("For example song, mean predicted (Valence, Arousal):", mean_pred)
else:
    print("No dynamic predictions could be made for the example song.")
