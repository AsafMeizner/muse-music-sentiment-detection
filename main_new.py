"""
Multi‑Task Music Analysis Pipeline with On‑the‑Fly Segmentation, Multithreading, Input Normalization,
and Limited Label Scope (Top Genres/Seeds)
==============================================================================================
This script:
  • Loads your CSV (with columns such as lastfm_url, track, artist, seeds, 
    valence_tags, arousal_tags, dominance_tags, mbid, genre, audio_previews)
  • Preprocesses labels by filtering to only the most frequent genres and seed tags.
     - Removes the genre "indie"
     - Keeps only the top 30 genres and top 30 seed tags.
  • Splits the dataset by song (using mbid) into train/validation/test sets.
  • Uses a Python generator wrapped by tf.data.Dataset to stream segment–level features
    extracted on the fly. Each song is segmented into 30‑second non–overlapping pieces 
    (plus a final segment if ≥20 seconds, padded to 30 seconds) using Librosa.
  • Uses multithreading (via ThreadPoolExecutor) to process files concurrently.
  • Adapts a normalization layer on a small sample of training features, showing progress.
  • Builds and trains a small multi–task dense model.
  
Adjust parameters such as SEGMENT_DURATION, MIN_SEGMENT_DURATION, max_workers, and the top–N cutoff as needed.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import ast
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers # type: ignore
from tqdm import tqdm  # for progress bar
import warnings

# ---------------------------
# Suppress Specific Librosa Warning
# ---------------------------
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set")

# ---------------------------
# Global Parameters
# ---------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Audio extraction and segmentation parameters
SR = 16000                    # Sampling rate
SEGMENT_DURATION = 30         # Target segment length in seconds
MIN_SEGMENT_DURATION = 20     # Minimum duration (in seconds) for a leftover segment to be used
N_MFCC = 20                   # Number of MFCC coefficients
N_MELS = 128                  # Number of Mel bands

# ---------------------------
# Feature Extraction Functions
# ---------------------------
def pad_audio(y, target_length):
    """Pad audio vector y with zeros up to target_length samples."""
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    return y

def extract_mfcc_from_audio(y, sr, n_mfcc=N_MFCC):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def extract_melspec_from_audio(y, sr, n_mels=N_MELS):
    return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

def extract_chroma_from_audio(y, sr):
    return librosa.feature.chroma_stft(y=y, sr=sr)

def extract_tonnetz_from_audio(y, sr):
    y_harm = librosa.effects.harmonic(y)
    return librosa.feature.tonnetz(y=y_harm, sr=sr)

def aggregate_feature(feature):
    """
    Compute mean, min, and max along the time axis for a 2D feature array.
    For an array of shape (feature_dim, time), the output vector will have length feature_dim * 3.
    """
    mean = np.mean(feature, axis=1)
    min_val = np.min(feature, axis=1)
    max_val = np.max(feature, axis=1)
    return np.concatenate([mean, min_val, max_val])

def extract_features_from_audio_segment(y, sr):
    """
    From an audio segment y, extract and aggregate the following features:
      - Chroma: 12 bins  -> 12*3 = 36 values
      - Mel Spectrogram: 128 bands  -> 128*3 = 384 values
      - MFCC: 20 coefficients  -> 20*3 = 60 values
      - Tonnetz: 6 features  -> 6*3 = 18 values
    Total dimension = 36 + 384 + 60 + 18 = 498.
    """
    chroma = extract_chroma_from_audio(y, sr)
    mel_spec = extract_melspec_from_audio(y, sr)
    mfcc = extract_mfcc_from_audio(y, sr)
    tonnetz = extract_tonnetz_from_audio(y, sr)
    
    chroma_feat = aggregate_feature(chroma)
    mel_feat = aggregate_feature(mel_spec)
    mfcc_feat = aggregate_feature(mfcc)
    tonnetz_feat = aggregate_feature(tonnetz)
    
    return np.concatenate([chroma_feat, mel_feat, mfcc_feat, tonnetz_feat])

def extract_segment_features(file_path, offset, duration=SEGMENT_DURATION, sr=SR):
    """
    Load a segment from file_path starting at offset (in seconds) for a given duration.
    If the segment is too short, pad with zeros.
    Returns a 498-dimensional feature vector.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration)
    except Exception as e:
        print(f"Error loading {file_path} at offset {offset}: {e}")
        return None
    target_length = int(duration * sr)
    y = pad_audio(y, target_length)
    return extract_features_from_audio_segment(y, sr)

def extract_all_segments_from_file(file_path, segment_duration=SEGMENT_DURATION,
                                   min_segment_duration=MIN_SEGMENT_DURATION, sr=SR):
    """
    For the given file, extract non–overlapping segments.
      - For each full segment (segment_duration seconds), extract features.
      - If the remaining audio is at least min_segment_duration seconds, extract it (after padding).
    Returns a tuple: (list_of_feature_vectors, list_of_offsets_in_seconds)
    """
    segments = []
    offsets = []
    try:
        total_duration = librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Could not get duration for {file_path}: {e}")
        return segments, offsets

    offset = 0.0
    while offset + segment_duration <= total_duration:
        feat = extract_segment_features(file_path, offset, duration=segment_duration, sr=sr)
        if feat is not None:
            segments.append(feat)
            offsets.append(offset)
        offset += segment_duration

    # Check for a remaining segment
    remaining = total_duration - offset
    if remaining >= min_segment_duration:
        feat = extract_segment_features(file_path, offset, duration=remaining, sr=sr)
        if feat is not None:
            segments.append(feat)
            offsets.append(offset)
    return segments, offsets

# ---------------------------
# Generator Function (with Multithreading)
# ---------------------------
def segment_generator(df_subset, max_workers=8):
    """
    A generator that yields one sample at a time.
    For each row in the dataframe subset, process the file in a separate thread and yield:
      (features, sentiment, genre, seeds)
    where:
      - features: np.array of shape (498,)
      - sentiment: np.array of shape (3,) [valence, arousal, dominance] as floats
      - genre: int (already integer–encoded)
      - seeds: np.array (binary vector) as float32
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_row(row):
        file_path = os.path.normpath(row['audio_previews'])
        segments, offsets = extract_all_segments_from_file(file_path)
        samples = []
        if segments:
            sentiment = np.array([float(row['valence_tags']),
                                  float(row['arousal_tags']),
                                  float(row['dominance_tags'])])
            genre = int(row['genre_encoded'])
            seeds = np.array(row['seeds_encoded'], dtype=np.float32)
            for feat in segments:
                samples.append((feat, sentiment, genre, seeds))
        return samples

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_subset.iterrows()]
        for future in as_completed(futures):
            samples = future.result()
            for sample in samples:
                yield sample

# ---------------------------
# TF.Data Pipeline Wrapper
# ---------------------------
def create_dataset_from_df(df_subset, max_workers, num_seeds):
    """
    Wrap the segment_generator in a tf.data.Dataset.
    Each yielded element is a tuple:
      (features, (sentiment, genre, seeds))
    with shapes: (498,), (3,), (), (num_seeds,)
    """
    def gen():
        for sample in segment_generator(df_subset, max_workers=max_workers):
            feat, sentiment, genre, seeds = sample
            yield feat.astype(np.float32), (
                sentiment.astype(np.float32),
                np.int32(genre),
                seeds.astype(np.float32)
            )

    output_types = (tf.float32, (tf.float32, tf.int32, tf.float32))
    output_shapes = (tf.TensorShape([498]), 
                     (tf.TensorShape([3]), tf.TensorShape([]), tf.TensorShape([num_seeds])))
    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------
# Model Definition with Input Normalization
# ---------------------------
def build_model(input_dim, num_genres, num_seeds, normalizer):
    """
    Build a small dense network with:
      - A dedicated normalization layer (pre-adapted)
      - A clipping layer to reduce the impact of outliers
      - Three output heads:
          * Sentiment regression: using a sigmoid activation scaled to [0,10]
          * Genre classification: softmax activation
          * Seeds multi-label classification: sigmoid activation
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Normalize inputs
    x = normalizer(inputs)
    # Clip values to a fixed range to mitigate extreme outliers.
    x = layers.Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0))(x)
    
    # Shared dense layers
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Sentiment head: use a sigmoid then multiply by 10 to constrain output to [0,10]
    sentiment_branch = layers.Dense(128, activation='relu')(x)
    sentiment_pred = layers.Dense(3, activation='sigmoid')(sentiment_branch)
    sentiment_output = layers.Lambda(lambda x: x * 10, name='sentiment')(sentiment_pred)

    # Genre head
    genre_branch = layers.Dense(128, activation='relu')(x)
    genre_output = layers.Dense(num_genres, activation='softmax', name='genre')(genre_branch)

    # Seeds head
    seeds_branch = layers.Dense(128, activation='relu')(x)
    seeds_output = layers.Dense(num_seeds, activation='sigmoid', name='seeds')(seeds_branch)

    model = models.Model(inputs=inputs, outputs=[sentiment_output, genre_output, seeds_output])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),  # Consider lowering further if needed
        loss={
            'sentiment': 'mse',
            'genre': 'sparse_categorical_crossentropy',
            'seeds': 'binary_crossentropy'
        },
        loss_weights={
            'sentiment': 1.0,
            'genre': 1.2,
            'seeds': 2.0
        },
        metrics={
            'sentiment': ['mae'],
            'genre': ['sparse_categorical_accuracy'],
            'seeds': ['binary_accuracy']
        }
    )
    return model

# ---------------------------
# Main Execution: Pipeline, Normalization Adaptation, and Training
# ---------------------------
def main():
    csv_path = 'filtered_dataset.csv'
    print("Loading CSV and pre-processing labels...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Process seeds: convert string representation to list if necessary
    df['seeds'] = df['seeds'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # --- Filter Genres ---
    # Count genres and remove "indie", then keep only the top 30 genres.
    genre_counts = df['genre'].value_counts()
    if 'indie' in genre_counts.index:
        genre_counts = genre_counts.drop('indie')
    top_genres = genre_counts.nlargest(30).index
    df = df[df['genre'].isin(top_genres)]
    print(f"Filtered genres: {len(df['genre'].unique())} retained.")

    # --- Filter Seeds ---
    # Flatten all seeds and keep only the top 30.
    all_seeds = [seed for seeds_list in df['seeds'] for seed in seeds_list]
    seed_counts = pd.Series(all_seeds).value_counts()
    top_seeds = seed_counts.nlargest(30).index.tolist()
    df['seeds'] = df['seeds'].apply(lambda seeds_list: [s for s in seeds_list if s in top_seeds])
    print(f"Filtered seed tags: {len(top_seeds)} top seeds retained.")

    # Encode genre (sparse integer labels)
    genre_encoder = LabelEncoder()
    df['genre_encoded'] = genre_encoder.fit_transform(df['genre'])

    # Encode seeds (multi-label binary)
    seed_encoder = MultiLabelBinarizer()
    seeds_encoded = seed_encoder.fit_transform(df['seeds'])
    df['seeds_encoded'] = list(seeds_encoded)  # store as list in each row

    # Ensure sentiment columns are float
    df['valence_tags'] = df['valence_tags'].astype(float)
    df['arousal_tags'] = df['arousal_tags'].astype(float)
    df['dominance_tags'] = df['dominance_tags'].astype(float)

    # ---------------------------
    # Split by Song (mbid)
    # ---------------------------
    unique_mbids = df['mbid'].unique()
    train_mbids, temp_mbids = train_test_split(unique_mbids, test_size=0.3, random_state=SEED)
    val_mbids, test_mbids = train_test_split(temp_mbids, test_size=0.5, random_state=SEED)
    df_train = df[df['mbid'].isin(train_mbids)]
    df_val   = df[df['mbid'].isin(val_mbids)]
    df_test  = df[df['mbid'].isin(test_mbids)]
    print(f"Number of songs (mbid): Train: {len(train_mbids)}, Val: {len(val_mbids)}, Test: {len(test_mbids)}")

    # Determine number of seeds (for output shape)
    num_seeds = seeds_encoded.shape[1]

    # ---------------------------
    # Create tf.data Datasets (Streaming Pipeline)
    # ---------------------------
    print("Creating dataset pipelines...")
    max_workers = 8  # Adjust based on your CPU/RAM
    train_ds = create_dataset_from_df(df_train, max_workers=max_workers, num_seeds=num_seeds)
    val_ds   = create_dataset_from_df(df_val, max_workers=max_workers, num_seeds=num_seeds)
    test_ds  = create_dataset_from_df(df_test, max_workers=max_workers, num_seeds=num_seeds)

        # ---------------------------
    # Adapt a Normalization Layer on a Subset of Training Features
    # ---------------------------
    sample_size = 50  # Reduced sample size for faster adaptation; adjust if needed.
    print(f"Collecting {sample_size} samples for normalization adaptation (this may take a few minutes)...")
    sample_list = []
    # Cache the unbatched samples to speed up repeated access.
    sample_ds = train_ds.map(lambda x, y: x).unbatch().take(sample_size).cache()
    
    # Use as_numpy_iterator() and iterate over a fixed number of samples.
    iterator = sample_ds.as_numpy_iterator()
    for i in range(sample_size):
        try:
            sample = next(iterator)
            sample_list.append(sample)
            if (i + 1) % 10 == 0:
                print(f"Collected {i + 1} samples...")
        except StopIteration:
            print("No more samples available.")
            break

    print(f"Collected {len(sample_list)} samples. Adapting normalization layer now...")
    sample_array = np.array(sample_list)
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(sample_array)
    print("Normalization adaptation complete.")

    # ---------------------------
    # Build and Train the Model
    # ---------------------------
    input_dim = 498  # Fixed feature dimension from our extraction
    num_genres = len(genre_encoder.classes_)
    model = build_model(input_dim, num_genres, num_seeds, normalizer)
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    print("Starting training...")
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        callbacks=[es, rlrop])

    results = model.evaluate(test_ds)
    print("Test results:", results)

    # ---------------------------
    # (Optional) Inference on a New Song
    # ---------------------------
    # To see how emotion changes over time, you can process a new song:
    # new_song_path = 'path/to/new_song.mp3'
    # segments, offsets = extract_all_segments_from_file(new_song_path)
    # if segments:
    #     segments = np.array(segments, dtype=np.float32)
    #     preds = model.predict(segments)
    #     # preds[0] are sentiment predictions; you can plot these versus offsets.
    # else:
    #     print("No segments extracted from the new song.")

if __name__ == '__main__':
    main()