import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# =============================================================================
# 1) SETUP & PARAMETERS
# =============================================================================
# Paths (adjust BASE_PATH to your system)
BASE_PATH = "./deam-dataset"  
STATIC_ANNOTATIONS_PATH = os.path.join(
    BASE_PATH,
    "DEAM_Annotations", "annotations", "annotations averaged per song", "song_level", 
    "static_annotations_averaged_songs_1_2000.csv"
)
AUDIO_FOLDER = os.path.join(BASE_PATH, "DEAM_audio", "MEMD_audio")

# Audio processing parameters
SAMPLE_RATE = 22050        # Standard sampling rate
SEGMENT_DURATION = 3.0       # each segment is 3 seconds long
NUM_SEGMENTS = 10            # number of segments to extract per track

# Mel-spectrogram parameters for a 3-second segment
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
# Calculate number of frames per 3-second segment
# Approx frames = SEGMENT_DURATION * (SAMPLE_RATE / HOP_LENGTH)
SEG_FRAMES = int(np.ceil(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH))
# For example, 3 * 22050 / 512 ~ 129 frames
print("SEG_FRAMES:", SEG_FRAMES)

# =============================================================================
# 2) LOAD STATIC (SONG-LEVEL) ANNOTATIONS
# =============================================================================
df_static = pd.read_csv(STATIC_ANNOTATIONS_PATH)
df_static.columns = df_static.columns.str.strip()  # remove extra spaces
df_static = df_static[["song_id", "valence_mean", "arousal_mean"]]
print("Total songs in static annotations:", len(df_static))

# =============================================================================
# 3) UTILITY FUNCTIONS FOR AUDIO PROCESSING
# =============================================================================
def load_audio_segments(mp3_path, sr=SAMPLE_RATE, seg_duration=SEGMENT_DURATION, num_segments=NUM_SEGMENTS):
    """
    Load an audio file and extract num_segments segments of seg_duration seconds,
    evenly spaced through the track.
    Returns a list of audio segments (np.array) or None if the track is too short.
    """
    if not os.path.exists(mp3_path):
        return None
    y, _ = librosa.load(mp3_path, sr=sr, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    required_duration = seg_duration * num_segments
    if total_duration < required_duration:
        return None  # skip tracks that are too short
    # Calculate start times for segments (evenly spaced)
    start_times = np.linspace(0, total_duration - seg_duration, num_segments)
    segments = []
    for start in start_times:
        start_sample = int(start * sr)
        end_sample = start_sample + int(seg_duration * sr)
        segment = y[start_sample:end_sample]
        segments.append(segment)
    return segments

def extract_mel_spectrogram(y, sr, n_mels=N_MELS):
    """
    Compute log-mel-spectrogram from audio segment.
    Returns array of shape [n_mels, time_frames].
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def pad_or_truncate(S_db, target_frames=SEG_FRAMES):
    """
    Pad with zeros or truncate the mel-spectrogram S_db (shape [n_mels, t])
    so that the time dimension is target_frames.
    """
    n_mels, t = S_db.shape
    if t == target_frames:
        return S_db
    elif t < target_frames:
        pad_width = target_frames - t
        pad_array = np.zeros((n_mels, pad_width), dtype=S_db.dtype)
        return np.concatenate((S_db, pad_array), axis=1)
    else:
        return S_db[:, :target_frames]

# =============================================================================
# 4) BUILD THE DATASET
#    For each song, extract NUM_SEGMENTS segments; each segment is converted
#    to a mel-spectrogram of shape [N_MELS, SEG_FRAMES, 1]. The track-level input
#    is then an array of shape [NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1].
#    The target is the song-level [valence_mean, arousal_mean].
# =============================================================================
X_list = []
Y_list = []
skipped = 0

for idx, row in df_static.iterrows():
    sid = row["song_id"]
    val = row["valence_mean"]
    aro = row["arousal_mean"]
    mp3_path = os.path.join(AUDIO_FOLDER, f"{int(sid)}.mp3")
    
    segments = load_audio_segments(mp3_path, sr=SAMPLE_RATE, seg_duration=SEGMENT_DURATION, num_segments=NUM_SEGMENTS)
    if segments is None:
        skipped += 1
        continue
    
    seg_spectrograms = []
    for seg in segments:
        S_db = extract_mel_spectrogram(seg, SAMPLE_RATE, n_mels=N_MELS)
        S_db = pad_or_truncate(S_db, target_frames=SEG_FRAMES)
        # Expand dims to get a channel dimension: [N_MELS, SEG_FRAMES, 1]
        S_db = np.expand_dims(S_db, axis=-1)
        seg_spectrograms.append(S_db)
    # Stack segments: shape becomes [NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1]
    X_song = np.stack(seg_spectrograms, axis=0)
    X_list.append(X_song)
    Y_list.append([val, aro])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

print(f"Built dataset with {len(X)} tracks. Skipped {skipped} tracks.")
print("X shape:", X.shape)  # (num_tracks, NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1)
print("Y shape:", Y.shape)  # (num_tracks, 2)

# =============================================================================
# 5) TRAIN/TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# =============================================================================
# 6) BUILD A HYBRID TIME-DISTRIBUTED CNN + LSTM MODEL
# =============================================================================
# The model processes each segment via a CNN (shared across segments)
# and then uses an LSTM to aggregate the segment-level features into a track-level prediction.
num_segments = NUM_SEGMENTS  # e.g., 10
input_shape = (N_MELS, SEG_FRAMES, 1)  # shape for one segment

# Define a CNN feature extractor for one segment
def create_cnn_extractor(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    # Output dimension can be tuned; here we use 256 units
    model.add(layers.Dense(256, activation='relu'))
    return model

cnn_extractor = create_cnn_extractor(input_shape)
cnn_extractor.summary()

# Now build the full model that accepts a sequence of segments
# Input shape: (NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1)
segment_input = layers.Input(shape=(num_segments, N_MELS, SEG_FRAMES, 1))
# Apply TimeDistributed to use the same CNN for each segment
td = layers.TimeDistributed(cnn_extractor)(segment_input)
# td now has shape (batch_size, NUM_SEGMENTS, feature_dim) where feature_dim=256
# Use an LSTM to aggregate across segments
lstm_out = layers.LSTM(128, return_sequences=False)(td)
# Optionally add dropout and dense layers
dense1 = layers.Dense(64, activation='relu')(lstm_out)
drop = layers.Dropout(0.3)(dense1)
output = layers.Dense(2, activation='linear')(drop)  # Predict [valence, arousal]

model = models.Model(inputs=segment_input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='mse',
              metrics=['mae'])
model.summary()

# =============================================================================
# 7) TRAIN THE MODEL
# =============================================================================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=4,  # you might need to adjust batch size depending on memory
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 8) EVALUATE THE MODEL
# =============================================================================
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE (loss):", test_loss)
print("Test MAE:", test_mae)

# Additionally compute Pearson correlation for each target dimension
preds = model.predict(X_test)
from scipy.stats import pearsonr
val_corr, _ = pearsonr(y_test[:, 0], preds[:, 0])
aro_corr, _ = pearsonr(y_test[:, 1], preds[:, 1])
print("Valence Pearson Corr:", val_corr)
print("Arousal Pearson Corr:", aro_corr)

# =============================================================================
# 9) PREDICTION ON A NEW TRACK
#    This function extracts NUM_SEGMENTS segments from a new track,
#    processes them through the model, and outputs a single prediction.
# =============================================================================
def predict_track_valence_arousal(mp3_path, num_segments=NUM_SEGMENTS, seg_duration=SEGMENT_DURATION):
    segments = load_audio_segments(mp3_path, sr=SAMPLE_RATE, seg_duration=seg_duration, num_segments=num_segments)
    if segments is None:
        print("Track too short or file not found.")
        return None
    seg_specs = []
    for seg in segments:
        S_db = extract_mel_spectrogram(seg, SAMPLE_RATE, n_mels=N_MELS)
        S_db = pad_or_truncate(S_db, target_frames=SEG_FRAMES)
        S_db = np.expand_dims(S_db, axis=-1)
        seg_specs.append(S_db)
    X_track = np.stack(seg_specs, axis=0)  # shape: (NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1)
    X_track = np.expand_dims(X_track, axis=0)  # add batch dimension: (1, NUM_SEGMENTS, N_MELS, SEG_FRAMES, 1)
    prediction = model.predict(X_track)[0]
    return prediction

# Example usage:
example_mp3 = os.path.join(AUDIO_FOLDER, "25.mp3")  # change to a valid file ID
predicted_values = predict_track_valence_arousal(example_mp3)
if predicted_values is not None:
    print("\nExample Track Prediction (Valence, Arousal):", predicted_values)
