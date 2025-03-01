import os
import ast
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# ============================
# Configuration and Constants
# ============================
SR = 22050             # Sampling rate
DURATION = 10          # Duration (seconds) to use from each audio file
HOP_LENGTH = 1024      # Hop length for spectrogram computation
N_MFCC = 40            # Number of MFCCs to compute
TARGET_FRAMES = 215    # Fixed number of time frames (approx. for 10 sec with hop_length=1024)
BATCH_SIZE = 16        # Adjust based on your available RAM/compute
NUM_EPOCHS = 50

# ============================
# 1. Load and Process the CSV
# ============================
df = pd.read_csv("filtered_dataset.csv")

# Process the genre column (categorical classification)
genre_encoder = LabelEncoder()
df['genre_encoded'] = genre_encoder.fit_transform(df['genre'])
num_genres = len(genre_encoder.classes_)

# Process the seeds column (a string representation of a list)
df['seeds_list'] = df['seeds'].apply(lambda x: ast.literal_eval(x))
seed_mlb = MultiLabelBinarizer()
seeds_encoded = seed_mlb.fit_transform(df['seeds_list'])
num_seed_labels = seeds_encoded.shape[1]
df['seeds_encoded'] = list(seeds_encoded)

# Process continuous labels (valence, arousal, dominance)
df['valence'] = df['valence_tags'].astype(float)
df['arousal'] = df['arousal_tags'].astype(float)
df['dominance'] = df['dominance_tags'].astype(float)

# Split into training and validation sets (80/20 split)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create numpy arrays for each field (for tf.data)
train_audio_paths = train_df['audio_previews'].values
train_genres = train_df['genre_encoded'].values
train_seeds = np.stack(train_df['seeds_encoded'].values)
train_valence = train_df['valence'].values
train_arousal = train_df['arousal'].values
train_dominance = train_df['dominance'].values

val_audio_paths = val_df['audio_previews'].values
val_genres = val_df['genre_encoded'].values
val_seeds = np.stack(val_df['seeds_encoded'].values)
val_valence = val_df['valence'].values
val_arousal = val_df['arousal'].values
val_dominance = val_df['dominance'].values

# ======================================
# 2. Audio Feature Extraction Functions
# ======================================
def extract_features(audio_path, duration=DURATION, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
    """
    Loads an audio file and extracts a set of features:
      - MFCC (n_mfcc coefficients)
      - Chroma (12 bins)
      - Tonnetz (6 dimensions from the harmonic component)
    The features are vertically stacked to form a (40+12+6=58) x time feature matrix.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    except Exception as e:
        print("Error loading {}: {}".format(audio_path, e))
        return None
    # Pad with zeros if audio is shorter than the target duration
    if len(y) < duration * sr:
        pad_length = duration * sr - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')
    # Compute MFCCs using keyword arguments to avoid positional argument error
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # Compute Chroma from the short-time Fourier transform
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=hop_length)
    # Compute Tonnetz features (requires harmonic component)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr, hop_length=hop_length)
    # Ensure all feature matrices have the same time dimension
    min_frames = min(mfcc.shape[1], chroma.shape[1], tonnetz.shape[1])
    mfcc = mfcc[:, :min_frames]
    chroma = chroma[:, :min_frames]
    tonnetz = tonnetz[:, :min_frames]
    # Stack features vertically: resulting shape = (40+12+6, min_frames)
    features = np.vstack([mfcc, chroma, tonnetz])
    # Normalize features (global normalization)
    features = (features - np.mean(features)) / np.std(features)
    # Expand dims to add a channel dimension -> shape: (58, time, 1)
    features = np.expand_dims(features, axis=-1)
    return features

def pad_or_crop(features, target_frames=TARGET_FRAMES):
    """
    Pads (if too short) or crops (if too long) the time dimension of features to target_frames.
    Assumes features shape is (freq_bins, time, 1).
    """
    shape = tf.shape(features)
    time_dim = shape[1]
    def crop():
        return features[:, :target_frames, :]
    def pad():
        pad_amount = target_frames - time_dim
        return tf.pad(features, [[0, 0], [0, pad_amount], [0, 0]])
    features = tf.cond(time_dim > target_frames, crop, pad)
    features.set_shape([features.shape[0], target_frames, 1])
    return features

# ======================================
# 3. Build the tf.data Pipeline
# ======================================
def load_audio_and_features(audio_path, genre, seeds, valence, arousal, dominance):
    """
    Given the file path and labels, load the audio, extract features, pad/crop to a fixed size,
    and return the feature tensor along with a dictionary of labels.
    """
    def _extract(path):
        # Convert EagerTensor to numpy
        path = path.numpy()
        # If it's bytes, decode it; otherwise, assume it's already a string
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        feat = extract_features(path)
        return feat.astype(np.float32)
    
    features = tf.py_function(func=_extract, inp=[audio_path], Tout=tf.float32)
    # Our feature extractor returns an array of shape (58, variable_time, 1)
    features.set_shape([58, None, 1])
    features = pad_or_crop(features, TARGET_FRAMES)
    return features, {'genre': genre, 
                      'seeds': seeds, 
                      'valence': valence, 
                      'arousal': arousal, 
                      'dominance': dominance}

# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_audio_paths, train_genres, train_seeds, train_valence, train_arousal, train_dominance)
)
train_dataset = train_dataset.map(load_audio_and_features, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_audio_paths, val_genres, val_seeds, val_valence, val_arousal, val_dominance)
)
val_dataset = val_dataset.map(load_audio_and_features, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ======================================
# 4. Define the Multi-Task Model in Keras
# ======================================
def build_model(input_shape=(58, TARGET_FRAMES, 1), num_genres=num_genres, num_seed_labels=num_seed_labels):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output branch for genre classification (categorical)
    genre_out = layers.Dense(num_genres, activation='softmax', name='genre')(x)
    # Output branch for seeds (multi-label, using sigmoid)
    seeds_out = layers.Dense(num_seed_labels, activation='sigmoid', name='seeds')(x)
    # Output branches for continuous predictions (regression)
    valence_out = layers.Dense(1, activation='linear', name='valence')(x)
    arousal_out = layers.Dense(1, activation='linear', name='arousal')(x)
    dominance_out = layers.Dense(1, activation='linear', name='dominance')(x)
    
    model = models.Model(inputs=inputs, outputs=[genre_out, seeds_out, valence_out, arousal_out, dominance_out])
    return model

model = build_model()
model.summary()

# ======================================
# 5. Compile and Train the Model
# ======================================
losses = {
    'genre': 'sparse_categorical_crossentropy',  # Genre label is an integer
    'seeds': 'binary_crossentropy',                # Multi-label classification loss
    'valence': 'mse',
    'arousal': 'mse',
    'dominance': 'mse'
}
metrics = {
    'genre': 'accuracy',
    'seeds': 'accuracy',
    'valence': 'mae',
    'arousal': 'mae',
    'dominance': 'mae'
}

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss=losses,
              metrics=metrics)

# Callbacks for early stopping and saving the best model
earlystop = callbacks.EarlyStopping(monitor='val_genre_accuracy',
                                    mode='max',
                                    patience=10,
                                    restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint("best_model.keras", monitor='val_genre_accuracy', save_best_only=True)

# Train the model
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=NUM_EPOCHS,
                    callbacks=[earlystop, checkpoint])

# ======================================
# 6. Inference: Predicting on a New Audio File
# ======================================
def predict_audio(audio_path):
    """
    Given an audio file path, extract features, pad/crop to fixed size,
    and predict the genre, seeds, and sentiment values.
    """
    feat = extract_features(audio_path)
    if feat is None:
        return None
    # Ensure the feature has the target number of time frames
    if feat.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - feat.shape[1]
        feat = np.pad(feat, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    else:
        feat = feat[:, :TARGET_FRAMES, :]
    feat = np.expand_dims(feat, axis=0)  # Add batch dimension
    predictions = model.predict(feat)
    
    # Decode genre prediction
    genre_pred = np.argmax(predictions[0], axis=1)[0]
    genre_label = genre_encoder.inverse_transform([genre_pred])[0]
    
    # Decode seeds (apply a threshold of 0.5)
    seeds_pred = (predictions[1][0] > 0.5).astype(int)
    seeds_labels = [seed for seed, flag in zip(seed_mlb.classes_, seeds_pred) if flag == 1]
    
    valence_pred = predictions[2][0][0]
    arousal_pred = predictions[3][0][0]
    dominance_pred = predictions[4][0][0]
    
    return {
        'genre': genre_label,
        'seeds': seeds_labels,
        'valence': valence_pred,
        'arousal': arousal_pred,
        'dominance': dominance_pred
    }

# Example usage:
# result = predict_audio("path/to/your/audio_file.mp3")
# print(result)
