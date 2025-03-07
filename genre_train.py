import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import numpy as np
import pandas as pd
import librosa
from matplotlib import pyplot as plt
from pydub import AudioSegment
from tensorflow import keras
import concurrent.futures
from tqdm import tqdm

# -------------------------------
# Audio Loading and Feature Extraction (in memory)
# -------------------------------
def load_audio_from_mp3(mp3_path, offset=0, duration=30):
    """
    Load an MP3 file, convert it to a numpy array (in memory), and return (audio, sample_rate).
    If loading fails, returns (None, None).
    """
    try:
        audio = AudioSegment.from_file(mp3_path, format="mp3")
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return None, None

    start_ms = offset * 1000
    end_ms = start_ms + duration * 1000
    audio = audio[start_ms:end_ms]
    
    # Convert to numpy array (if stereo, average channels)
    y = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        y = y.reshape((-1, audio.channels)).mean(axis=1)
    
    # Normalize to [-1, 1]
    max_val = float(2 ** (8 * audio.sample_width - 1))
    y = y / max_val
    sr = audio.frame_rate
    return y, sr

def get_mfcc(mp3_path):
    y, sr = load_audio_from_mp3(mp3_path, offset=0, duration=30)
    return None if y is None else librosa.feature.mfcc(y=y, sr=sr)

def get_melspectrogram(mp3_path):
    y, sr = load_audio_from_mp3(mp3_path, offset=0, duration=30)
    return None if y is None else librosa.feature.melspectrogram(y=y, sr=sr)

def get_chroma_vector(mp3_path):
    y, sr = load_audio_from_mp3(mp3_path, offset=0, duration=30)
    return None if y is None else librosa.feature.chroma_stft(y=y, sr=sr)

def get_tonnetz(mp3_path):
    y, sr = load_audio_from_mp3(mp3_path, offset=0, duration=30)
    return None if y is None else librosa.feature.tonnetz(y=y, sr=sr)

def get_feature(mp3_path):
    """
    Extracts MFCC, Mel spectrogram, Chroma, and Tonnetz features from an MP3 file.
    For each feature, the mean, min, and max (across time) are computed and concatenated.
    Returns a single fixed-length feature vector or None if any feature extraction fails.
    """
    mfcc = get_mfcc(mp3_path)
    melspectrogram = get_melspectrogram(mp3_path)
    chroma = get_chroma_vector(mp3_path)
    tonnetz = get_tonnetz(mp3_path)

    if mfcc is None or melspectrogram is None or chroma is None or tonnetz is None:
        return None

    # Compute statistics for each feature matrix
    mfcc_feature = np.concatenate((mfcc.mean(axis=1), mfcc.min(axis=1), mfcc.max(axis=1)))
    melspectrogram_feature = np.concatenate((melspectrogram.mean(axis=1),
                                             melspectrogram.min(axis=1),
                                             melspectrogram.max(axis=1)))
    chroma_feature = np.concatenate((chroma.mean(axis=1),
                                     chroma.min(axis=1),
                                     chroma.max(axis=1)))
    tonnetz_feature = np.concatenate((tonnetz.mean(axis=1),
                                      tonnetz.min(axis=1),
                                      tonnetz.max(axis=1)))

    return np.concatenate(
        (chroma_feature, melspectrogram_feature, mfcc_feature, tonnetz_feature)
    )

# -------------------------------
# Parallel Processing Function
# -------------------------------
def process_audio_file(args):
    """
    Processes a single file: extracts features from the given mp3 file.
    Returns a tuple (feature, genre) if successful, otherwise None.
    """
    mp3_path, genre = args
    if not os.path.exists(mp3_path):
        print(f"MP3 file not found: {mp3_path}")
        return None
    feat = get_feature(mp3_path)
    if feat is None:
        print(f"Feature extraction failed for {mp3_path}")
        return None
    return feat, genre

# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    # Load dataset CSV (adjust the path if needed)
    data = pd.read_csv("muse_dataset.csv")
    
    # Determine the top 15 genres and remove "indie" if present.
    genre_counts = data['genre'].value_counts()
    top_genres = genre_counts.head(15).index.tolist()
    if "indie" in top_genres:
        top_genres.remove("indie")
    data = data[data['genre'].isin(top_genres)].reset_index(drop=True)
    print("Selected top genres:", top_genres)
    
    # Create a list of tuples: (mp3 file path, genre)
    records = list(zip(data["audio_previews"].tolist(), data["genre"].tolist()))
    
    # Use parallel processing to extract features.
    features = []
    labels = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_audio_file, records),
                           total=len(records),
                           desc="Processing audio files"):
            if result is not None:
                feat, genre = result
                features.append(feat)
                labels.append(genre)
    
    features = np.array(features)
    labels = np.array(labels)
    print("Extracted features shape:", features.shape)
    
    # Encode genres as integers.
    unique_genres = np.unique(labels)
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    idx_to_genre = {idx: genre for genre, idx in genre_to_idx.items()}
    labels_numeric = np.array([genre_to_idx[label] for label in labels])
    print("Genre mapping:", genre_to_idx)
    
    # Shuffle and split the dataset (60% training, 20% validation, 20% testing)
    num_samples = features.shape[0]
    indices = np.random.permutation(num_samples)
    features = features[indices]
    labels_numeric = labels_numeric[indices]
    
    train_split = int(0.6 * num_samples)
    val_split = int(0.8 * num_samples)
    features_train = features[:train_split]
    labels_train = labels_numeric[:train_split]
    features_val = features[train_split:val_split]
    labels_val = labels_numeric[train_split:val_split]
    features_test = features[val_split:]
    labels_test = labels_numeric[val_split:]
    
    print(f"Training samples: {features_train.shape[0]}")
    print(f"Validation samples: {features_val.shape[0]}")
    print(f"Testing samples: {features_test.shape[0]}")
    
    # Build a simple neural network model.
    input_shape = features_train.shape[1]
    num_classes = len(unique_genres)
    
    inputs = keras.Input(shape=(input_shape,), name="feature")
    x = keras.layers.Dense(300, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(200, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    model.summary()
    
    # Train the model.
    model.fit(
        x=features_train,
        y=labels_train,
        verbose=1,
        validation_data=(features_val, labels_val),
        epochs=64
    )
    
    # Evaluate on the test set.
    score = model.evaluate(x=features_test, y=labels_test, verbose=0)
    print('Test Accuracy : {:.2f}%'.format(score[1] * 100))
    
if __name__ == "__main__":
    main()
