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
import warnings

# Suppress warnings from librosa
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# ----------------------------
# Global Parameters and Paths
# ----------------------------
BASE_PATH = "./deam-dataset"  # Adjust to your DEAM dataset location
STATIC_CSV = os.path.join(BASE_PATH, "DEAM_Annotations", "annotations", "annotations averaged per song", "song_level", "static_annotations_averaged_songs_1_2000.csv")
AUDIO_FOLDER = os.path.join(BASE_PATH, "DEAM_audio", "MEMD_audio")
RESULTS_DIR = "./setiment-results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_RATE = 22050
CHUNK_DURATION = 30.0  # seconds to extract from each song (from the middle)

# Mel-spectrogram parameters
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
TARGET_FRAMES = 256  # Fix the time dimension (approx 30s -> 256 frames)

# ----------------------------
# Helper Functions
# ----------------------------
def load_fixed_chunk(file_path, sr=SAMPLE_RATE, duration=CHUNK_DURATION):
    """Load a fixed chunk (duration seconds) from the middle of the audio."""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print("Error loading", file_path, e)
        return None
    total_duration = librosa.get_duration(y=y, sr=sr)
    if total_duration < duration:
        return None
    start_sec = max(0, (total_duration / 2.0) - (duration / 2.0))
    end_sec = start_sec + duration
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return y[start_sample:end_sample], sr

def compute_mel_spectrogram(y, sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX):
    """Compute log-mel spectrogram (in dB) from audio y."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def fix_spectrogram_length(S, target_frames=TARGET_FRAMES):
    """Pad or truncate spectrogram S (shape: [n_mels, T]) to exactly target_frames along time axis."""
    n_mels, T = S.shape
    if T < target_frames:
        pad_width = target_frames - T
        S_fixed = np.pad(S, ((0,0), (0, pad_width)), mode='constant')
    else:
        S_fixed = S[:, :target_frames]
    return S_fixed

def process_audio_file(file_path, sr=SAMPLE_RATE, duration=CHUNK_DURATION, target_frames=TARGET_FRAMES):
    """
    Load a fixed chunk from the middle of the audio file,
    compute its log-mel spectrogram, fix the time axis to target_frames,
    transpose (to shape [T, n_mels]) and add a channel dimension.
    Returns an array of shape (target_frames, n_mels, 1).
    """
    chunk_data = load_fixed_chunk(file_path, sr=sr, duration=duration)
    if chunk_data is None:
        return None
    y_chunk, sr = chunk_data
    S_db = compute_mel_spectrogram(y_chunk, sr)
    S_fixed = fix_spectrogram_length(S_db, target_frames=target_frames)
    S_fixed = S_fixed.T  # shape becomes (T, n_mels)
    return np.expand_dims(S_fixed, -1)  # shape: (T, n_mels, 1)

def scale_label(label):
    """Scale label from [1,9] to approximately [-0.2, 0.2]."""
    return (label - 5.0) * 0.05

def invert_label(scaled):
    """Invert scaling: original = (scaled / 0.05) + 5."""
    return (scaled / 0.05) + 5.0

def build_dataset(static_csv, audio_folder, sr=SAMPLE_RATE, duration=CHUNK_DURATION, target_frames=TARGET_FRAMES, max_songs=None):
    """
    Build dataset from static annotations.
    For each song, load a fixed chunk and compute its mel-spectrogram.
    Returns X (shape: [num_songs, target_frames, n_mels, 1]) and Y (scaled labels).
    """
    df = pd.read_csv(static_csv)
    df.columns = df.columns.str.strip()
    df = df[["song_id", "valence_mean", "arousal_mean"]]
    X_list = []
    Y_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing static songs"):
        if max_songs is not None and idx >= max_songs:
            break
        song_id = int(row["song_id"])
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")
        if not os.path.exists(file_path):
            continue
        processed = process_audio_file(file_path, sr=sr, duration=duration, target_frames=target_frames)
        if processed is None:
            continue
        X_list.append(processed)
        val_scaled = scale_label(row["valence_mean"])
        aro_scaled = scale_label(row["arousal_mean"])
        Y_list.append([val_scaled, aro_scaled])
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y

def build_improved_model(input_shape):
    """
    Build an improved CNN model with a VGG-style architecture.
    Input shape: (TARGET_FRAMES, N_MELS, 1)
    The model applies several convolutional blocks, then global average pooling,
    followed by dense layers to predict valence and arousal.
    """
    inputs = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Block 2
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Block 3
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse',
                  metrics=['mae'])
    return model

def plot_training_curves(history, results_dir):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "improved_loss_curve.png"))
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title("Training and Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (scaled)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "improved_mae_curve.png"))
    plt.show()

def plot_evaluation_scatter(y_true, y_pred, results_dir):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(y_true[:,0], y_pred[:,0], alpha=0.5, color='blue')
    plt.xlabel("True Valence (scaled)")
    plt.ylabel("Predicted Valence (scaled)")
    plt.title("Predicted vs True Valence")
    plt.subplot(1,2,2)
    plt.scatter(y_true[:,1], y_pred[:,1], alpha=0.5, color='red')
    plt.xlabel("True Arousal (scaled)")
    plt.ylabel("Predicted Arousal (scaled)")
    plt.title("Predicted vs True Arousal")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved_predicted_vs_true_scatter.png"))
    plt.show()

def predict_song_emotion(audio_path, model, sr=SAMPLE_RATE, duration=CHUNK_DURATION, target_frames=TARGET_FRAMES):
    """
    Given a new audio file, load a fixed chunk (from the middle), compute its mel-spectrogram,
    and predict song-level valence and arousal.
    Returns the prediction (scaled).
    """
    processed = process_audio_file(audio_path, sr=sr, duration=duration, target_frames=target_frames)
    if processed is None:
        print("Could not process audio:", audio_path)
        return None
    processed = np.expand_dims(processed, 0)
    pred_scaled = model.predict(processed)[0]
    return pred_scaled

# ----------------------------
# Main Function
# ----------------------------
def main():
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Build dataset from static annotations
    print("Building dataset from static annotations...")
    X, Y = build_dataset(STATIC_CSV, AUDIO_FOLDER, sr=SAMPLE_RATE, duration=CHUNK_DURATION, target_frames=TARGET_FRAMES, max_songs=300)
    if X is None or len(X) == 0:
        print("No data loaded. Check paths.")
        return
    print("Dataset shapes:", X.shape, Y.shape)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Build improved model
    input_shape = (TARGET_FRAMES, N_MELS, 1)
    model = build_improved_model(input_shape)
    model.summary()
    
    # Set up callbacks
    checkpoint_path = os.path.join(RESULTS_DIR, "best_improved_model.h5")
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=16, callbacks=[checkpoint_cb, early_stop_cb], verbose=1)
    
    # Save final model
    final_model_path = os.path.join(RESULTS_DIR, "final_improved_model.h5")
    model.save(final_model_path)
    print("Final model saved to:", final_model_path)
    
    # Plot training curves and scatter plots
    plot_training_curves(history, RESULTS_DIR)
    preds = model.predict(X_test)
    plot_evaluation_scatter(y_test, preds, RESULTS_DIR)
    
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", test_loss, "Test MAE:", test_mae)
    val_corr, _ = pearsonr(y_test[:,0], preds[:,0])
    aro_corr, _ = pearsonr(y_test[:,1], preds[:,1])
    print("Valence Pearson Corr (scaled):", val_corr)
    print("Arousal Pearson Corr (scaled):", aro_corr)
    
    # Inference on an example song
    example_song = os.path.join(AUDIO_FOLDER, "25.mp3")  # Replace with a valid file ID
    pred_song = predict_song_emotion(example_song, model, sr=SAMPLE_RATE, duration=CHUNK_DURATION, target_frames=TARGET_FRAMES)
    if pred_song is not None:
        print("Predicted (scaled) for example song:", pred_song)
        print("Predicted (original scale) for example song:", invert_label(pred_song))
    else:
        print("Failed to predict for example song.")

if __name__ == '__main__':
    main()
