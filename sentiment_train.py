import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
import multiprocessing

# Suppress warnings from librosa
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# ----------------------------
# GLOBAL PARAMETERS AND PATHS
# ----------------------------
BASE_PATH = "./deam-dataset"  # Adjust to your dataset location
STATIC_CSV = os.path.join(
    BASE_PATH, 
    "DEAM_Annotations", 
    "annotations", 
    "annotations averaged per song", 
    "song_level", 
    "static_annotations_averaged_songs_1_2000.csv"
)
AUDIO_FOLDER = os.path.join(BASE_PATH, "DEAM_audio", "MEMD_audio")
RESULTS_DIR = "./setiment-results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Audio settings
SAMPLE_RATE = 22050
SEGMENT_DURATION = 10.0  # each segment is 10 seconds

# Mel-spectrogram parameters
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
TARGET_FRAMES = 200  # fixed time dimension

# Data augmentation parameters (pitch shift only)
AUGMENT_PROB = 0.5
PITCH_SHIFT_RANGE = 1.0  # ±1 semitone

# ----------------------------
# LABEL SCALING UTILS
# ----------------------------
def scale_label(label):
    return (label - 5.0) * 0.05

def invert_label(scaled):
    return (scaled / 0.05) + 5.0

# ----------------------------
# AUDIO FEATURE EXTRACTION FUNCTIONS
# ----------------------------
def compute_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                       hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
    return librosa.power_to_db(S, ref=np.max)

def fix_spectrogram_length(S_db, target_frames=TARGET_FRAMES):
    n_mels, T = S_db.shape
    if T >= target_frames:
        return S_db[:, :target_frames]
    pad_width = target_frames - T
    return np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')

def load_entire_song(file_path, sr=SAMPLE_RATE):
    try:
        y, sr_ = librosa.load(file_path, sr=sr, mono=True)
        return y, sr_
    except Exception as e:
        print("Error loading:", file_path, e)
        return None

def process_audio_file(file_path, sr=SAMPLE_RATE, duration=SEGMENT_DURATION, target_frames=TARGET_FRAMES):
    data = load_entire_song(file_path, sr=sr)
    if data is None:
        return None
    y, sr_ = data
    total_dur = librosa.get_duration(y=y, sr=sr_)
    if total_dur < duration:
        return None
    start_sec = max(0, (total_dur / 2.0) - (duration / 2.0))
    end_sec = start_sec + duration
    start_sample = int(start_sec * sr_)
    end_sample = int(end_sec * sr_)
    y_chunk = y[start_sample:end_sample]
    S_db = compute_mel_spectrogram(y_chunk, sr_)
    S_fixed = fix_spectrogram_length(S_db, target_frames=target_frames)
    S_fixed = S_fixed.T  # shape becomes (target_frames, n_mels)
    return np.expand_dims(S_fixed, -1)  # shape: (target_frames, n_mels, 1)

# ----------------------------
# MULTI-SEGMENT DATASET BUILDING
# ----------------------------
def build_multiseg_dataset(static_csv, audio_folder, seg_duration=SEGMENT_DURATION, max_songs=None):
    df = pd.read_csv(static_csv)
    df.columns = df.columns.str.strip()
    df = df[["song_id", "valence_mean", "arousal_mean"]]
    
    X_list = []
    Y_list = []
    song_segment_map = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building multi-seg dataset"):
        if max_songs is not None and idx >= max_songs:
            break
        sid = int(row["song_id"])
        label_scaled = np.array([scale_label(row["valence_mean"]), scale_label(row["arousal_mean"])], dtype=np.float32)
        mp3_path = os.path.join(audio_folder, f"{sid}.mp3")
        if not os.path.exists(mp3_path):
            continue
        data = load_entire_song(mp3_path, sr=SAMPLE_RATE)
        if data is None:
            continue
        y_song, sr_ = data
        total_dur = librosa.get_duration(y=y_song, sr=sr_)
        if total_dur < seg_duration:
            continue
        num_segs = int(total_dur // seg_duration)
        for seg_idx in range(num_segs):
            start_sec = seg_idx * seg_duration
            end_sec = start_sec + seg_duration
            start_sample = int(start_sec * sr_)
            end_sample = int(end_sec * sr_)
            y_segment = y_song[start_sample:end_sample]
            X_list.append((y_segment, sr_))
            Y_list.append(label_scaled)
            song_segment_map.append((idx, seg_idx))
    
    return X_list, np.array(Y_list, dtype=np.float32), song_segment_map

# ----------------------------
# DATA AUGMENTATION (PITCH SHIFT)
# ----------------------------
def augment_segment(y_segment, sr_):
    if np.random.rand() < AUGMENT_PROB:
        semitones = np.random.uniform(-PITCH_SHIFT_RANGE, PITCH_SHIFT_RANGE)
        # Pass n_steps as keyword argument
        y_segment = librosa.effects.pitch_shift(y_segment, sr=sr_, n_steps=semitones)
    return y_segment

# ----------------------------
# DATA GENERATORS
# ----------------------------
def data_generator(X_list, Y, batch_size=8, train_mode=True):
    num_samples = len(X_list)
    idxs = np.arange(num_samples)
    while True:
        np.random.shuffle(idxs)
        for i in range(0, num_samples, batch_size):
            batch_idx = idxs[i:i+batch_size]
            batch_x = []
            batch_y = []
            for j in batch_idx:
                y_segment, sr_ = X_list[j]
                label = Y[j]
                if train_mode:
                    y_segment = augment_segment(y_segment, sr_)
                S_db = compute_mel_spectrogram(y_segment, sr_)
                S_fixed = fix_spectrogram_length(S_db, TARGET_FRAMES)
                S_fixed = S_fixed.T  # (TARGET_FRAMES, n_mels)
                S_fixed = np.expand_dims(S_fixed, -1)
                batch_x.append(S_fixed)
                batch_y.append(label)
            yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))

def data_generator_eval(X_list, Y, batch_size=8):
    num_samples = len(X_list)
    idxs = np.arange(num_samples)
    while True:
        for i in range(0, num_samples, batch_size):
            batch_idx = idxs[i:i+batch_size]
            batch_x = []
            batch_y = []
            for j in batch_idx:
                y_segment, sr_ = X_list[j]
                label = Y[j]
                S_db = compute_mel_spectrogram(y_segment, sr_)
                S_fixed = fix_spectrogram_length(S_db, TARGET_FRAMES)
                S_fixed = S_fixed.T
                S_fixed = np.expand_dims(S_fixed, -1)
                batch_x.append(S_fixed)
                batch_y.append(label)
            yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))

# ----------------------------
# CUSTOM SCALING LAYER WITH GET_CONFIG()
# ----------------------------
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value=1.0, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(1,), 
                                     initializer=tf.keras.initializers.Constant(self.initial_value),
                                     trainable=True)
        super(ScalingLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.alpha

    def get_config(self):
        config = super(ScalingLayer, self).get_config()
        config.update({"initial_value": self.initial_value})
        return config

# ----------------------------
# MODEL: VGG-like CNN with Learnable Scaling
# ----------------------------
def build_vgg_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation='linear')(x)
    outputs = ScalingLayer()(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    return model

# ----------------------------
# SONG-LEVEL EVALUATION
# ----------------------------
def evaluate_song_level(X_list, Y, song_segment_map, model, batch_size=8):
    from collections import defaultdict
    num_samples = len(X_list)
    steps = int(np.ceil(num_samples / batch_size))
    eval_gen = data_generator_eval(X_list, Y, batch_size=batch_size)
    seg_preds = model.predict(eval_gen, steps=steps, verbose=1)
    
    song_preds = defaultdict(list)
    song_trues = {}
    for i, (song_idx, seg_idx) in enumerate(song_segment_map):
        song_preds[song_idx].append(seg_preds[i])
        if song_idx not in song_trues:
            song_trues[song_idx] = Y[i]
    
    song_ids_sorted = sorted(song_preds.keys())
    y_true_song = []
    y_pred_song = []
    for s_idx in song_ids_sorted:
        seg_preds_song = np.array(song_preds[s_idx])
        avg_pred = np.mean(seg_preds_song, axis=0)
        y_true_song.append(song_trues[s_idx])
        y_pred_song.append(avg_pred)
    return np.array(y_true_song, dtype=np.float32), np.array(y_pred_song, dtype=np.float32)

def plot_scatter_song_level(y_true_song, y_pred_song, results_dir):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(y_true_song[:,0], y_pred_song[:,0], alpha=0.5, color='blue')
    plt.xlabel("True Valence (scaled)")
    plt.ylabel("Predicted Valence (scaled)")
    plt.title("Song-Level Predicted vs True Valence")
    plt.subplot(1,2,2)
    plt.scatter(y_true_song[:,1], y_pred_song[:,1], alpha=0.5, color='red')
    plt.xlabel("True Arousal (scaled)")
    plt.ylabel("Predicted Arousal (scaled)")
    plt.title("Song-Level Predicted vs True Arousal")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "song_level_scatter.png"))
    plt.show()

# ----------------------------
# INFERENCE FUNCTION FOR NEW SONGS
# ----------------------------
def predict_song_emotion(audio_path, model, sr=SAMPLE_RATE, seg_duration=SEGMENT_DURATION, target_frames=TARGET_FRAMES):
    data = load_entire_song(audio_path, sr=sr)
    if data is None:
        print("Error loading audio:", audio_path)
        return None
    y_song, sr_ = data
    total_dur = librosa.get_duration(y=y_song, sr=sr_)
    num_segs = int(total_dur // seg_duration)
    seg_preds = []
    for i in range(num_segs):
        start_sec = i * seg_duration
        end_sec = start_sec + seg_duration
        start_sample = int(start_sec * sr_)
        end_sample = int(end_sec * sr_)
        y_seg = y_song[start_sample:end_sample]
        S_db = compute_mel_spectrogram(y_seg, sr_)
        S_fixed = fix_spectrogram_length(S_db, target_frames=target_frames)
        S_fixed = S_fixed.T
        S_fixed = np.expand_dims(S_fixed, -1)
        batch_x = np.expand_dims(S_fixed, 0)
        pred = model.predict(batch_x)[0]
        seg_preds.append(pred)
    return np.mean(np.array(seg_preds), axis=0) if seg_preds else None

# ----------------------------
# PLOTTING TRAINING CURVES
# ----------------------------
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

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    multiprocessing.freeze_support()

    print("Building multi-seg dataset from static CSV ...")
    X_list, Y, song_segment_map = build_multiseg_dataset(STATIC_CSV, AUDIO_FOLDER, seg_duration=SEGMENT_DURATION, max_songs=200)
    print("Number of segments:", len(X_list), "Label shape:", Y.shape)
    if len(X_list) == 0:
        print("No data loaded. Check your paths or dataset.")
        return

    # Song-level split based on song index
    song_idxs = np.array([s_idx for (s_idx, seg_idx) in song_segment_map])
    unique_songs = np.unique(song_idxs)
    np.random.shuffle(unique_songs)
    cutoff = int(0.8 * len(unique_songs))
    train_songs = unique_songs[:cutoff]
    test_songs = unique_songs[cutoff:]
    train_mask = np.isin(song_idxs, train_songs)
    test_mask = np.isin(song_idxs, test_songs)

    X_list_train = [X_list[i] for i in range(len(X_list)) if train_mask[i]]
    Y_train = Y[train_mask]
    map_train = [song_segment_map[i] for i in range(len(X_list)) if train_mask[i]]

    X_list_test = [X_list[i] for i in range(len(X_list)) if test_mask[i]]
    Y_test = Y[test_mask]
    map_test = [song_segment_map[i] for i in range(len(X_list)) if test_mask[i]]

    print("Train segments:", len(X_list_train), "Test segments:", len(X_list_test))

    # Build model
    input_shape = (TARGET_FRAMES, N_MELS, 1)
    model = build_vgg_cnn(input_shape)
    model.summary()

    # Create data generators
    batch_size = 8
    steps_train = int(np.ceil(len(X_list_train) / batch_size))
    steps_val = int(np.ceil(len(X_list_test) / batch_size))
    train_gen = data_generator(X_list_train, Y_train, batch_size=batch_size, train_mode=True)
    val_gen = data_generator_eval(X_list_test, Y_test, batch_size=batch_size)

    # Callbacks
    checkpoint_path = os.path.join(RESULTS_DIR, "best_multiseg_vgg.h5")
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(RESULTS_DIR, "final_multiseg_vgg.h5")
    model.save(final_model_path)
    print("Final model saved to:", final_model_path)

    # Plot training curves
    plot_training_curves(history, RESULTS_DIR)

    # Evaluate segment-level
    steps_eval = int(np.ceil(len(X_list_test) / batch_size))
    seg_gen = data_generator_eval(X_list_test, Y_test, batch_size=batch_size)
    seg_loss, seg_mae = model.evaluate(seg_gen, steps=steps_eval, verbose=0)
    seg_preds = model.predict(seg_gen, steps=steps_eval, verbose=0)
    print("Segment-Level Test MSE:", seg_loss, "MAE:", seg_mae)
    _extracted_from_main_72(
        Y_test,
        seg_preds,
        "Segment-Level Valence Corr (scaled):",
        "Segment-Level Arousal Corr (scaled):",
    )
    # Evaluate song-level
    y_true_song, y_pred_song = evaluate_song_level(X_list_test, Y_test, map_test, model, batch_size=batch_size)
    _extracted_from_main_72(
        y_true_song,
        y_pred_song,
        "Song-Level Valence Corr (scaled):",
        "Song-Level Arousal Corr (scaled):",
    )
    plot_scatter_song_level(y_true_song, y_pred_song, RESULTS_DIR)

    # Inference on an example song
    example_sid = 25
    example_path = os.path.join(AUDIO_FOLDER, f"{example_sid}.mp3")
    loaded = load_entire_song(example_path, sr=SAMPLE_RATE)
    if loaded is None:
        print("Could not load example song.")
        return
    y_song, sr_ = loaded
    total_dur = librosa.get_duration(y=y_song, sr=sr_)
    num_segs_ex = int(total_dur // SEGMENT_DURATION)
    seg_preds_list = []
    for i in range(num_segs_ex):
        start_sec = i * SEGMENT_DURATION
        end_sec = start_sec + SEGMENT_DURATION
        start_sample = int(start_sec * sr_)
        end_sample = int(end_sec * sr_)
        y_seg = y_song[start_sample:end_sample]
        S_db = compute_mel_spectrogram(y_seg, sr_)
        S_fixed = fix_spectrogram_length(S_db, TARGET_FRAMES)
        S_fixed = S_fixed.T
        S_fixed = np.expand_dims(S_fixed, -1)
        batch_x = np.expand_dims(S_fixed, 0)
        seg_pred_scaled = model.predict(batch_x)[0]
        seg_preds_list.append(seg_pred_scaled)

    if seg_preds_list:
        seg_preds_arr = np.array(seg_preds_list)
        avg_pred_scaled = np.mean(seg_preds_arr, axis=0)
        print(f"Example Song ID: {example_sid}")
        print("Average Predicted (scaled):", avg_pred_scaled)
        print("Average Predicted (original scale):", invert_label(avg_pred_scaled))
    else:
        print("No segments for example song.")


# TODO Rename this here and in `main`
def _extracted_from_main_72(arg0, arg1, arg2, arg3):
    val_corr_seg, _ = pearsonr(arg0[:,0], arg1[:,0])
    aro_corr_seg, _ = pearsonr(arg0[:,1], arg1[:,1])
    print(arg2, val_corr_seg)
    print(arg3, aro_corr_seg)

if __name__ == '__main__':
    main()
