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

# =============================================================================
# 0) SET UP RESULTS DIRECTORY
# =============================================================================
results_dir = "./setiment-results"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# 1) SET PATHS AND PARAMETERS
# =============================================================================
# Adjust BASE_PATH to point to your DEAM dataset folder.
BASE_PATH = "./deam-dataset"
STATIC_ANNOTATIONS_PATH = os.path.join(
    BASE_PATH, "DEAM_Annotations", "annotations", "annotations averaged per song", "song_level", 
    "static_annotations_averaged_songs_1_2000.csv"
)
AUDIO_FOLDER = os.path.join(BASE_PATH, "DEAM_audio", "MEMD_audio")

# Audio processing parameters
SAMPLE_RATE = 22050  # Standard sampling rate for music
DURATION = None      # We load the full track (features are averaged)

# =============================================================================
# 2) FEATURE EXTRACTION FUNCTIONS
# =============================================================================
def extract_features(file_path, sr=SAMPLE_RATE):
    """
    Extract a 193-dimensional feature vector from an audio file.
    Features:
      - 40 MFCCs (mean over time)
      - 12 Chroma STFT values (mean)
      - 128 Mel-spectrogram (in dB, mean)
      - 7 Spectral Contrast values (mean)
      - 6 Tonnetz values (mean, from harmonic component)
    Returns a numpy array of shape (193,).
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print("Error loading", file_path, e)
        return None
    if y.size == 0:
        return None
    # 1. MFCC: 40 coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    # 2. Chromagram: 12 dimensions
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # 3. Mel-spectrogram: 128 mel bands
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    # 4. Spectral Contrast: 7 dimensions (default)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    # 5. Tonnetz: 6 dimensions (from harmonic component)
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    # Concatenate all features: 40 + 12 + 128 + 7 + 6 = 193
    features = np.concatenate((mfcc_mean, chroma_mean, mel_mean, spec_contrast_mean, tonnetz_mean))
    return features

# =============================================================================
# 3) LOAD STATIC ANNOTATIONS & EXTRACT FEATURES
# =============================================================================
df_static = pd.read_csv(STATIC_ANNOTATIONS_PATH)
df_static.columns = df_static.columns.str.strip()  # Remove extra spaces
# Use only song_id, valence_mean, and arousal_mean
df_static = df_static[["song_id", "valence_mean", "arousal_mean"]]
print("Total songs in static annotations:", len(df_static))

X_list = []
Y_list = []
skipped = 0

for idx, row in df_static.iterrows():
    song_id = int(row["song_id"])
    file_path = os.path.join(AUDIO_FOLDER, f"{song_id}.mp3")
    features = extract_features(file_path, sr=SAMPLE_RATE)
    if features is None:
        skipped += 1
        continue
    X_list.append(features)
    # Scale labels from [1, 9] to [-0.2, 0.2] using: scaled = (x - 5)*0.05
    val_scaled = (row["valence_mean"] - 5.0) * 0.05
    aro_scaled = (row["arousal_mean"] - 5.0) * 0.05
    Y_list.append([val_scaled, aro_scaled])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

print(f"Extracted features for {len(X)} tracks. Skipped {skipped} tracks.")
print("X shape:", X.shape)  # (num_tracks, 193)
print("Y shape:", Y.shape)  # (num_tracks, 2)

# =============================================================================
# 4) DATA SCALING & TRAIN/TEST SPLIT
# =============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# =============================================================================
# 5) BUILD A DEEP FEED-FORWARD NEURAL NETWORK
#    (Using 6 hidden layers with 500 nodes each, ReLU activation, and Dropout)
# =============================================================================
input_dim = X_train.shape[1]  # should be 193

model = models.Sequential()
model.add(layers.Input(shape=(input_dim,)))
for i in range(6):
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='linear'))  # Predict [valence, arousal]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# =============================================================================
# 6) SET UP CALLBACKS: EARLY STOPPING & MODEL CHECKPOINTS
# =============================================================================
checkpoint_path = os.path.join(results_dir, "best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stop_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# =============================================================================
# 7) TRAIN THE MODEL
# =============================================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint_cb, early_stop_cb],
    verbose=1
)

# =============================================================================
# 8) SAVE THE FINAL MODEL
# =============================================================================
final_model_path = os.path.join(results_dir, "final_model.h5")
model.save(final_model_path)
print("Final model saved to:", final_model_path)

# =============================================================================
# 9) PLOT TRAINING CURVES & SAVE GRAPHS
# =============================================================================
# Plot Loss Curves
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
loss_plot_path = os.path.join(results_dir, "loss_curve.png")
plt.savefig(loss_plot_path)
plt.show()

# Plot MAE Curves
plt.figure(figsize=(8, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
mae_plot_path = os.path.join(results_dir, "mae_curve.png")
plt.savefig(mae_plot_path)
plt.show()

# =============================================================================
# 10) EVALUATE THE MODEL
# =============================================================================
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE (loss):", test_loss)
print("Test MAE:", test_mae)

preds = model.predict(X_test)
val_corr, _ = pearsonr(y_test[:, 0], preds[:, 0])
aro_corr, _ = pearsonr(y_test[:, 1], preds[:, 1])
print("Valence Pearson Corr:", val_corr)
print("Arousal Pearson Corr:", aro_corr)

# =============================================================================
# 11) PREDICTION ON A NEW AUDIO FILE
#    (Inverts the scaling to return predictions on the original [1,9] scale)
# =============================================================================
def predict_emotion(file_path, scaler_X, model):
    """
    Given an audio file, extract the 193-dim feature vector, scale it,
    predict the scaled [valence, arousal], then invert scaling to return
    predictions on the original scale [1,9].
    """
    features = extract_features(file_path, sr=SAMPLE_RATE)
    if features is None:
        return None
    features_scaled = scaler_X.transform(features.reshape(1, -1))
    pred_scaled = model.predict(features_scaled)[0]
    # Invert scaling: (pred_scaled / 0.05) + 5
    val_pred = pred_scaled[0] / 0.05 + 5.0
    aro_pred = pred_scaled[1] / 0.05 + 5.0
    return np.array([val_pred, aro_pred])

# Example usage:
example_file = os.path.join(AUDIO_FOLDER, "25.mp3")  # replace with a valid file id
prediction = predict_emotion(example_file, scaler, model)
if prediction is not None:
    print("\nPrediction for track (Valence, Arousal) on original scale [1,9]:", prediction)
