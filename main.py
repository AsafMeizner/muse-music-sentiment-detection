import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Sometimes helps with certain CPU issues

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress common librosa/mp3 warnings

import ast
import random
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from tensorflow.keras import backend as K  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Reshape, Permute, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -----------------------------------------------------------------------------
# Global Config & Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
MAX_PAD_LENGTH = 300      # number of frames in Mel spectrogram
TOP_GENRES = 30
TOP_SEEDS = 30
BATCH_SIZE = 32
EPOCHS = 60               # can increase if you have time
INITIAL_LR = 1e-4
L2_REG = 1e-4
CHUNK_DURATION = 15       # 15s chunks
MAX_AUDIO_DURATION = 1200 # 20 minutes max (to avoid extremely long files)
TARGET_SR = 22050         # Fixed sample rate for faster consistent processing

# -----------------------------------------------------------------------------
# Data Augmentation Helpers
# -----------------------------------------------------------------------------
def augment_audio_waveform(y, sr):
    """Random time-stretch, pitch shift, and additive noise."""
    # Comment out or reduce if you want to speed up training/preprocessing
    if np.random.rand() < 0.3:  # reduce chance from 0.5 to 0.3
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if np.random.rand() < 0.3:
        steps = np.random.randint(-2, 3)  # from -2 to 2 semitones
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    return y

def spec_augment(mel_spec_db, num_masks=1,
                 freq_masking_max_percentage=0.10,
                 time_masking_max_percentage=0.10):
    """
    SpecAugment: random frequency and time masking.
    Reduced the default num_masks and percentages to speed up.
    """
    augmented = mel_spec_db.copy()
    num_mels, max_frames = augmented.shape

    # Frequency masking
    for _ in range(num_masks):
        mask_size = int(freq_masking_max_percentage * num_mels)
        start = np.random.randint(0, num_mels - mask_size)
        augmented[start:start+mask_size, :] = 0

    # Time masking
    for _ in range(num_masks):
        mask_size = int(time_masking_max_percentage * max_frames)
        start = np.random.randint(0, max_frames - mask_size)
        augmented[:, start:start+mask_size] = 0

    return augmented

# -----------------------------------------------------------------------------
# Audio Splitting / Chunking
# -----------------------------------------------------------------------------
def split_audio_into_chunks(y, sr, chunk_duration=15):
    """
    Split the entire audio array `y` into consecutive fixed-size chunks 
    of length `chunk_duration` seconds. Discards leftover if it 
    doesn't fit exactly.
    """
    chunk_samples = int(chunk_duration * sr)
    num_chunks = len(y) // chunk_samples
    segments = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        segments.append(y[start:end])
    return segments

# -----------------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------------
def extract_features_from_waveform(
    y, sr, max_pad_length=300, augment=False
):
    """
    Compute Mel spectrogram => dB => normalize => optional spec augment.
    """
    try:
        if augment:
            y = augment_audio_waveform(y, sr)

        # Compute mel-spectrogram
        # For speed, you can lower n_mels or hop_length, etc.
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000
        )
        if mel_spec.size == 0:
            return None

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

        if augment:
            mel_spec_db = spec_augment(mel_spec_db)

        # Pad or truncate
        pad_width = max_pad_length - mel_spec_db.shape[1]
        if pad_width > 0:
            mel_spec_db = np.pad(
                mel_spec_db, ((0, 0), (0, pad_width)), mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_length]

        return mel_spec_db
    except Exception as e:
        # Optionally log the error
        return None

def process_audio_file_to_chunks(
    file_path,
    max_duration=1200,   # Up to 20 min
    chunk_duration=15,
    max_pad_length=300,
    augment=False,
    sr=22050
):
    """
    Load up to `max_duration` seconds of audio at `sr` sample rate,
    split into chunks, and extract Mel-spectrogram features.
    """
    try:
        # Force a fixed sample rate for consistency + speed
        y, sr = librosa.load(
            file_path, sr=sr, mono=True, duration=max_duration
        )
        if y is None or len(y) == 0:
            return []

        # Split into chunks
        segments = split_audio_into_chunks(
            y, sr, chunk_duration=chunk_duration
        )
        all_features = []
        for seg in segments:
            feat = extract_features_from_waveform(
                seg, sr, max_pad_length=max_pad_length, augment=augment
            )
            if feat is not None:
                all_features.append(feat)
        return all_features

    except Exception as e:
        # Possibly log or skip
        return []

def parallel_extract_features_in_chunks(
    audio_paths,
    max_duration=1200,
    chunk_duration=15,
    max_pad_length=300,
    augment=False,
    sr=22050,
    num_workers=2
):
    """
    Parallelize feature extraction across files.
    Using a small num_workers (like 2) if I/O is an issue.
    """
    process_func = partial(
        process_audio_file_to_chunks,
        max_duration=max_duration,
        chunk_duration=chunk_duration,
        max_pad_length=max_pad_length,
        augment=augment,
        sr=sr
    )

    all_features = []
    all_file_indices = []

    # Sometimes, concurrency might slow things if the dataset is huge
    # due to I/O overhead. Adjust as needed.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_func, audio_paths),
            total=len(audio_paths),
            desc="Extracting Features"
        ))

    for file_idx, feats_list in enumerate(results):
        for f in feats_list:
            all_features.append(f)
            all_file_indices.append(file_idx)

    return np.array(all_features), all_file_indices

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
def filter_top_genres(df, top_n=TOP_GENRES):
    """Keep only the top_n most frequent genres, remove 'indie' if present."""
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(top_n).index.tolist()
    if 'indie' in top_genres:
        top_genres.remove('indie')
    df_filtered = df[df['genre'].isin(top_genres)].copy()
    return df_filtered

def filter_top_seeds(df, top_n=TOP_SEEDS):
    """Limit seeds to top_n most frequent across dataset."""
    from collections import Counter
    seed_list_flat = []
    for seeds in df['seeds_parsed']:
        seed_list_flat.extend(seeds)
    seed_counts = Counter(seed_list_flat)
    top_seeds = [s for s, _ in seed_counts.most_common(top_n)]

    def keep_top(lst):
        return [x for x in lst if x in top_seeds]

    df['seeds_parsed'] = df['seeds_parsed'].apply(keep_top)
    return df

def prepare_data_with_chunks(
    df, 
    scaler, 
    genre_encoder, 
    seed_encoder,
    fit_scaler=False, 
    augment=False,
    max_duration=1200,
    chunk_duration=15,
    sr=22050,
    num_workers=2
):
    """
    Extract chunk-based features from audio files in df.
    Returns:
        X => shape [N_chunks, 128, max_pad_length]
        y_reg => shape [N_chunks, 3]
        y_genre => shape [N_chunks]
        y_seed => shape [N_chunks, #seeds]
    """
    audio_paths = df['audio_previews'].apply(os.path.abspath).to_list()
    # -- You can optionally do caching here if you'd like. --

    X_list, file_indices = parallel_extract_features_in_chunks(
        audio_paths=audio_paths,
        max_duration=max_duration,
        chunk_duration=chunk_duration,
        max_pad_length=MAX_PAD_LENGTH,
        augment=augment,
        sr=sr,
        num_workers=num_workers  # small concurrency
    )
    if len(X_list) == 0:
        raise ValueError("No valid audio features extracted.")

    # Prepare regression labels
    original_reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    chunk_reg = [original_reg_labels[idx] for idx in file_indices]
    chunk_reg = np.array(chunk_reg)

    if fit_scaler:
        scaler.fit(chunk_reg)
    scaled_reg = scaler.transform(chunk_reg)

    # Genre labels
    original_genre_labels = genre_encoder.transform(df['genre'].astype(str))
    chunk_genre = [original_genre_labels[idx] for idx in file_indices]
    chunk_genre = np.array(chunk_genre)

    # Seeds
    df_seeds_multi_hot = seed_encoder.transform(df['seeds_parsed'])
    chunk_seeds = [df_seeds_multi_hot[idx] for idx in file_indices]
    chunk_seeds = np.array(chunk_seeds)

    return X_list, scaled_reg, chunk_genre, chunk_seeds

# -----------------------------------------------------------------------------
# WeightedBinaryCrossentropy for Seeds
# -----------------------------------------------------------------------------
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """
    Weighted binary crossentropy to penalize false-negatives more strongly,
    so it doesn't learn to predict "all zeros" for seeds.
    """
    def __init__(self, pos_weight=5.0, from_logits=False, label_smoothing=0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            reduction=tf.keras.losses.Reduction.NONE,  # we'll reduce manually
            label_smoothing=label_smoothing
        )

    def call(self, y_true, y_pred):
        bce_unreduced = self.bce(y_true, y_pred)
        # Weighted by pos_weight for positive samples
        weights = 1.0 + (self.pos_weight - 1.0) * y_true
        return tf.reduce_mean(weights * bce_unreduced)

# -----------------------------------------------------------------------------
# Transformer-based Model
# -----------------------------------------------------------------------------
def build_transformer_model(
    input_shape,
    num_genres,
    num_seeds,
    l2_reg=1e-4,
    initial_lr=1e-4,
    genre_label_smoothing=0.1,
    seed_pos_weight=5.0
):
    """
    CNN front-end -> Transformer Encoder -> heads: regression, single-label, multi-label.
    """
    from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention

    class TransformerEncoder(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
            super().__init__()
            self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential([
                Dense(ff_dim, activation='relu'),
                Dense(embed_dim),
            ])
            self.layernorm1 = LayerNormalization()
            self.layernorm2 = LayerNormalization()
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)

        def call(self, inputs, training=False):
            attn_output = self.attn(inputs, inputs)  # self-attention
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    reg = l2(l2_reg)
    inputs = Input(shape=input_shape)

    # CNN front-end
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # shape = [batch, freq, time, channels]
    x = Permute((2, 1, 3))(x)
    time_steps = x.shape[1]
    features = x.shape[2] * x.shape[3]
    x = Reshape((time_steps, features))(x)

    # Transformer Encoders
    embed_dim = features
    ff_dim = 256
    num_heads = 4
    dropout_rate = 0.2
    transformer_layers = 2

    for _ in range(transformer_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Global average pooling across time
    x = GlobalAveragePooling1D()(x)

    # Dense layer
    x = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)

    # 1) Regression
    reg_output = Dense(3, activation='linear', name='reg_output')(x)

    # 2) Single-label classification (genre)
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)

    # 3) Multi-label classification (seeds)
    seed_output = Dense(num_seeds, activation='sigmoid', name='seed_output')(x)

    model = Model(inputs, [reg_output, class_output, seed_output])

    # Define losses
    w_bce = WeightedBinaryCrossentropy(pos_weight=seed_pos_weight)
    genre_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, label_smoothing=genre_label_smoothing
    )

    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss={
            'reg_output': 'mse',
            'class_output': genre_loss,
            'seed_output': w_bce
        },
        loss_weights={
            'reg_output': 1.0,
            'class_output': 1.0,
            'seed_output': 3.0
        },
        metrics={
            'reg_output': ['mae'],
            'class_output': ['accuracy'],
            'seed_output': [tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
        }
    )
    return model

# -----------------------------------------------------------------------------
# TF Dataset Creation
# -----------------------------------------------------------------------------
def create_tf_dataset(X, y_reg, y_class, y_seed, batch_size=32, shuffle=True):
    """Create a tf.data.Dataset that yields (X, {'reg_output':..., ...})."""
    ds = tf.data.Dataset.from_tensor_slices(
        (X, {
            'reg_output': y_reg,
            'class_output': y_class,
            'seed_output': y_seed
        })
    )
    if shuffle:
        ds = ds.shuffle(len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------------------------------------------------------
# Plot Functions
# -----------------------------------------------------------------------------
def plot_training_history(history, output_path='training_history.png'):
    hist = history.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1) Total loss
    axes[0].plot(hist['loss'], label='Train Loss')
    axes[0].plot(hist['val_loss'], label='Val Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()

    # 2) Regression MAE
    if 'reg_output_mae' in hist:
        axes[1].plot(hist['reg_output_mae'], label='Train MAE (Reg)')
        axes[1].plot(hist['val_reg_output_mae'], label='Val MAE (Reg)')
        axes[1].set_title('Regression MAE')
        axes[1].legend()

    # 3) Genre Accuracy
    if 'class_output_accuracy' in hist:
        axes[2].plot(hist['class_output_accuracy'], label='Train Acc (Genre)')
        axes[2].plot(hist['val_class_output_accuracy'], label='Val Acc (Genre)')
        axes[2].set_title('Genre Accuracy')
        axes[2].legend()

    # 4) Seeds Binary Accuracy
    if 'seed_output_binary_accuracy' in hist:
        axes[3].plot(hist['seed_output_binary_accuracy'], label='Train Acc (Seeds)')
        axes[3].plot(hist['val_seed_output_binary_accuracy'], label='Val Acc (Seeds)')
        axes[3].set_title('Seeds Binary Accuracy')
        axes[3].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Genre Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()

def plot_regression_results(y_true_inv, y_pred_inv, output_path_prefix='regression'):
    """
    Plot predicted vs. actual for the 3 regression targets in the *original domain*.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Valence', 'Arousal', 'Dominance']

    for i, ax in enumerate(axes):
        ax.scatter(y_true_inv[:, i], y_pred_inv[:, i], alpha=0.3)
        ax.set_xlabel(f"True {target_names[i]}")
        ax.set_ylabel(f"Pred {target_names[i]}")
        ax.set_title(f"{target_names[i]}: True vs. Pred")

        min_val = min(y_true_inv[:, i].min(), y_pred_inv[:, i].min())
        max_val = max(y_true_inv[:, i].max(), y_pred_inv[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png")
    plt.close()

def plot_seed_distribution(y_true_seed, y_pred_seed, seed_names, output_path='seed_distribution.png'):
    """Bar chart comparing the total number of times each seed is present in True vs. Pred."""
    true_counts = np.sum(y_true_seed, axis=0)
    pred_counts = np.sum(y_pred_seed, axis=0)

    x_indices = np.arange(len(seed_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - 0.2, true_counts, width=0.4, label='True')
    plt.bar(x_indices + 0.2, pred_counts, width=0.4, label='Predicted')
    plt.xticks(x_indices, seed_names, rotation=45, ha='right')
    plt.title('Seed Distribution (True vs. Predicted)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        # Required columns
        required_cols = [
            'audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags',
            'genre', 'seeds'
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        # Convert seeds from string -> list
        def parse_seed_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return []
            elif isinstance(x, list):
                return x
            else:
                return []
        df['seeds_parsed'] = df['seeds'].apply(parse_seed_list)

        # Filter top genres and seeds
        df = filter_top_genres(df, top_n=TOP_GENRES)
        df = filter_top_seeds(df, top_n=TOP_SEEDS)

        # Label Encoder for genres
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))
        unique_genres = genre_encoder.classes_
        num_genres = len(unique_genres)

        # MultiLabelBinarizer for seeds
        seed_encoder = MultiLabelBinarizer()
        seed_encoder.fit(df['seeds_parsed'])
        unique_seeds = seed_encoder.classes_
        num_seeds = len(unique_seeds)

        # Train/Val/Test Split
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED)
        train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True, random_state=SEED)

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print("Genres:", unique_genres)
        print("Seeds:", unique_seeds)

        # Prepare data
        scaler = MinMaxScaler()

        # Increase num_workers with caution if disk I/O is a bottleneck
        X_train, y_train_reg, y_train_genre, y_train_seed = prepare_data_with_chunks(
            train_df, scaler, genre_encoder, seed_encoder,
            fit_scaler=True, augment=True,
            max_duration=MAX_AUDIO_DURATION,
            chunk_duration=CHUNK_DURATION,
            sr=TARGET_SR,
            num_workers=2
        )

        X_val, y_val_reg, y_val_genre, y_val_seed = prepare_data_with_chunks(
            val_df, scaler, genre_encoder, seed_encoder,
            fit_scaler=False, augment=False,
            max_duration=MAX_AUDIO_DURATION,
            chunk_duration=CHUNK_DURATION,
            sr=TARGET_SR,
            num_workers=2
        )

        X_test, y_test_reg, y_test_genre, y_test_seed = prepare_data_with_chunks(
            test_df, scaler, genre_encoder, seed_encoder,
            fit_scaler=False, augment=False,
            max_duration=MAX_AUDIO_DURATION,
            chunk_duration=CHUNK_DURATION,
            sr=TARGET_SR,
            num_workers=2
        )

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Build TF datasets
        train_ds = create_tf_dataset(
            X_train, y_train_reg, y_train_genre, y_train_seed,
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_ds = create_tf_dataset(
            X_val, y_val_reg, y_val_genre, y_val_seed,
            batch_size=BATCH_SIZE, shuffle=False
        )
        test_ds = create_tf_dataset(
            X_test, y_test_reg, y_test_genre, y_test_seed,
            batch_size=BATCH_SIZE, shuffle=False
        )

        # Build/compile model
        model = build_transformer_model(
            input_shape=X_train.shape[1:],  # e.g. (128, 300, 1)
            num_genres=num_genres,
            num_seeds=num_seeds,
            l2_reg=L2_REG,
            initial_lr=INITIAL_LR,
            genre_label_smoothing=0.1,
            seed_pos_weight=5.0
        )
        model.summary()

        # Callbacks
        ckpt_dir = 'model_checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        print("Starting training...")
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks
        )

        # Plot training
        plot_training_history(history, 'training_curves.png')

        # Evaluate
        print("Evaluating on test set...")
        test_results = model.evaluate(test_ds, verbose=1)
        print("Test Results:", test_results)

        # Predictions
        y_pred_reg, y_pred_class, y_pred_seed = model.predict(test_ds, verbose=1)

        # Invert scale for regression
        y_test_reg_inv = scaler.inverse_transform(y_test_reg)
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)

        # Plot regression results
        plot_regression_results(y_test_reg_inv, y_pred_reg_inv, 'regression')

        # Genre classification
        y_pred_genre = np.argmax(y_pred_class, axis=1)
        print("\nGenre Classification Report:")
        print(classification_report(
            y_test_genre, y_pred_genre, target_names=unique_genres, zero_division=0
        ))
        plot_confusion_matrix(y_test_genre, y_pred_genre, unique_genres, 'confusion_matrix.png')

        # Seeds multi-label
        seed_threshold = 0.5
        y_pred_seed_bin = (y_pred_seed >= seed_threshold).astype(int)

        print("\nSeeds Multi-label Classification Report:")
        print(classification_report(
            y_test_seed, y_pred_seed_bin, target_names=unique_seeds, zero_division=0
        ))
        plot_seed_distribution(y_test_seed, y_pred_seed_bin, unique_seeds, 'seed_distribution.png')

        # Save
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Random sample predictions
        test_indices = np.arange(len(y_test_reg))
        np.random.shuffle(test_indices)
        for i in test_indices[:5]:
            true_reg = y_test_reg_inv[i]
            pred_reg = y_pred_reg_inv[i]
            true_genre = unique_genres[y_test_genre[i]]
            pred_genre_argmax = unique_genres[y_pred_genre[i]]

            true_seed_labels = [
                s for s, val in zip(unique_seeds, y_test_seed[i]) if val == 1
            ]
            pred_seed_labels = [
                s for s, val in zip(unique_seeds, y_pred_seed_bin[i]) if val == 1
            ]

            # Sort probabilities for top 3 genres
            pred_probs = y_pred_class[i]
            sorted_idx = np.argsort(pred_probs)[::-1]
            top_3_idx = sorted_idx[:3]

            print(f"\nSample index: {i}")
            print(f"  True V/A/D:  {true_reg}")
            print(f"  Pred V/A/D:  {pred_reg}")
            print(f"  True Genre:  {true_genre}")
            print(f"  Pred Genre:  {pred_genre_argmax}")
            print("  Top 3 predicted genres w/ confidence:")
            for rank, g_id in enumerate(top_3_idx, 1):
                print(f"    {rank}) {unique_genres[g_id]} (conf={pred_probs[g_id]:.4f})")
            print(f"  True Seeds:  {true_seed_labels}")
            print(f"  Pred Seeds:  {pred_seed_labels}")

    except Exception as e:
        print(f"Error during training: {e}")
