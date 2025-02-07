import ast
import os

from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks # type: ignore

# ======================
# Configuration
# ======================
SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
TARGET_SR = 16000
DURATION = 30  # Seconds per track
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = (TARGET_SR * DURATION) // HOP_LENGTH

# ======================
# Audio Processing Pipeline
# ======================
def load_and_process_audio(file_path):
    # Read and decode audio using TensorFlow I/O
    audio = tf.io.read_file(file_path)
    audio = tfio.audio.decode_mp3(audio, rate=TARGET_SR)[:, 0]  # Convert to mono
    
    # Trim/pad to exact duration
    target_samples = TARGET_SR * DURATION
    audio = audio[:target_samples]
    if tf.size(audio) < target_samples:
        padding = target_samples - tf.size(audio)
        audio = tf.pad(audio, [[0, padding]])
    
    # Generate Mel spectrogram using TensorFlow ops
    stft = tf.signal.stft(audio, frame_length=2048, frame_step=HOP_LENGTH)
    spectrogram = tf.abs(stft)
    
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=TARGET_SR,
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0
    )
    mel_spectrogram = tf.tensordot(spectrogram**2, mel_filter, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # Normalize
    mean, var = tf.nn.moments(mel_spectrogram, axes=[0,1])
    mel_spectrogram = (mel_spectrogram - mean) / tf.sqrt(var + 1e-6)
    
    return mel_spectrogram[..., tf.newaxis]  # Add channel dimension

# ======================
# Data Augmentation
# ======================
def augment_spectrogram(mel_spec):
    # Time masking
    if tf.random.uniform(()) > 0.5:
        t_mask = tf.random.uniform((), 0, MAX_FRAMES//10, dtype=tf.int32)
        t_start = tf.random.uniform((), 0, MAX_FRAMES - t_mask, dtype=tf.int32)
        mask = tf.concat([
            tf.ones((N_MELS, t_start, 1)),
            tf.zeros((N_MELS, t_mask, 1)),
            tf.ones((N_MELS, MAX_FRAMES - t_start - t_mask, 1))
        ], axis=1)
        mel_spec *= mask
    
    # Frequency masking
    if tf.random.uniform(()) > 0.5:
        f_mask = tf.random.uniform((), 0, N_MELS//8, dtype=tf.int32)
        f_start = tf.random.uniform((), 0, N_MELS - f_mask, dtype=tf.int32)
        mel_spec = tf.concat([
            mel_spec[:f_start],
            tf.zeros((f_mask, MAX_FRAMES, 1)),
            mel_spec[f_start + f_mask:]
        ], axis=0)
    
    return mel_spec

# ======================
# Model Architecture
# ======================
def create_multi_task_model(input_shape, num_genres, num_seeds):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stem convolution
    x = layers.Conv2D(32, 3, padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        # First block (may need downsampling)
        residual = layers.Conv2D(filters, 1, strides=2 if filters != 64 else 1)(x)
        residual = layers.BatchNormalization()(residual)
        
        # Main path
        x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(2)(x) if filters != 256 else x
    
    # Attention pooling
    attention = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.multiply([x, attention])
    x = layers.GlobalAveragePooling2D()(x)
    
    # Shared dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Task-specific heads
    # 1. Sentiment regression (Valence, Arousal, Dominance)
    sentiment = layers.Dense(256, activation='relu')(x)
    sentiment = layers.Dense(3, name='sentiment')(sentiment)
    
    # 2. Genre classification
    genre = layers.Dense(256, activation='relu')(x)
    genre = layers.Dense(num_genres, activation='softmax', name='genre')(genre)
    
    # 3. Seed tags
    seeds = layers.Dense(512, activation='relu')(x)
    seeds = layers.Dense(num_seeds, activation='sigmoid', name='seeds')(seeds)
    
    model = models.Model(inputs, [sentiment, genre, seeds])
    
    # Custom loss weights
    losses = {
        'sentiment': losses.MeanSquaredError(),
        'genre': losses.SparseCategoricalCrossentropy(),
        'seeds': losses.BinaryCrossentropy()
    }
    loss_weights = {'sentiment': 1.0, 'genre': 1.2, 'seeds': 2.0}
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            'sentiment': metrics.MeanAbsoluteError(),
            'genre': metrics.SparseCategoricalAccuracy(),
            'seeds': metrics.BinaryAccuracy(threshold=0.5)
        }
    )
    return model

# ======================
# Data Pipeline
# ======================
def create_dataset(df, label_encoder, seed_encoder, augment=False):
    def process_row(row):
        # Process audio
        audio = load_and_process_audio(row['audio_previews'])
        
        # Apply augmentation
        if augment:
            audio = augment_spectrogram(audio)
        
        # Process labels
        sentiment = tf.convert_to_tensor([
            row['valence_tags'],
            row['arousal_tags'],
            row['dominance_tags']
        ], dtype=tf.float32)
        
        genre = label_encoder.transform([row['genre']])[0]
        seeds = seed_encoder.transform([ast.literal_eval(row['seeds'])]).flatten()
        
        return audio, (sentiment, genre, seeds)

    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    dataset = dataset.map(
        lambda x: process_row(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ======================
# Audio Metadata Fixer
# ======================
def clean_audio_files(directory):
    """Fix corrupted audio files using FFmpeg"""
    from subprocess import run
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            output_path = os.path.join(directory, f"cleaned_{filename}")
            run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', filepath, '-map_metadata', '-1',
                '-c:a', 'libmp3lame', '-q:a', '2', output_path
            ], check=True)
            os.replace(output_path, filepath)

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Step 1: Clean audio files metadata
    print("Cleaning audio metadata...")
    clean_audio_files("audio_previews")
    
    # Step 2: Load and prepare data
    df = pd.read_csv('filtered_dataset.csv')
    
    # Encode labels
    genre_encoder = LabelEncoder()
    df['genre'] = genre_encoder.fit_transform(df['genre'])
    
    seed_encoder = MultiLabelBinarizer()
    df['seeds'] = df['seeds'].apply(ast.literal_eval)
    seed_encoder.fit(df['seeds'])
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
    
    # Create datasets
    train_ds = create_dataset(train_df, genre_encoder, seed_encoder, augment=True)
    val_ds = create_dataset(val_df, genre_encoder, seed_encoder)
    test_ds = create_dataset(test_df, genre_encoder, seed_encoder)
    
    # Initialize model
    model = create_multi_task_model(
        (N_MELS, MAX_FRAMES, 1),
        len(genre_encoder.classes_),
        len(seed_encoder.classes_)
    )
    
    # Train with optimized callbacks
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            callbacks.ModelCheckpoint(
                'best_model.keras',
                save_best_only=True,
                save_weights_only=True
            )
        ]
    )
    
    # Evaluate
    model.load_weights('best_model.keras')
    results = model.evaluate(test_ds)
    print(f"\nFinal Test Performance:")
    print(f"Sentiment MAE: {results[1]:.3f}")
    print(f"Genre Accuracy: {results[3]:.3f}")
    print(f"Seed Accuracy: {results[5]:.3f}")