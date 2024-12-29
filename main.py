import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional; sometimes helps with certain CPU issues

import ast
import random
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

# Keras / sklearn / multiprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Bidirectional, LSTM, Permute, Reshape, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --------------------------------
# Global Config & Reproducibility
# --------------------------------
# (We remove random_state below for the train_test_split so the test set is truly random)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------
# Hyperparameters
# --------------------------------
MAX_PAD_LENGTH = 300
TOP_GENRES = 10
TOP_SEEDS = 15        # Keep top 15 seeds
BATCH_SIZE = 32
EPOCHS = 50           # You can try more epochs if you have time
INITIAL_LR = 1e-4
L2_REG = 1e-4

# Longer chunk duration:
CHUNK_DURATION = 10   # 10-second chunks instead of 5

# --------------------------------
# Data Augmentation Helpers
# --------------------------------
def augment_audio_waveform(y, sr):
    """
    Augment waveform with random time-stretch, pitch shift, and additive noise.
    """
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if np.random.rand() < 0.5:
        steps = np.random.randint(-2, 3)  # from -2 to 2 semitones
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    return y

def spec_augment(mel_spec_db, num_masks=2,
                 freq_masking_max_percentage=0.15,
                 time_masking_max_percentage=0.15):
    """
    SpecAugment: random frequency and time masking for the spectrogram.
    mel_spec_db shape: (n_mels, time_frames)
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

# --------------------------------
# Audio Splitting / Chunking
# --------------------------------
def split_audio_into_chunks(y, sr, total_duration=30, chunk_duration=10):
    """
    Splits the waveform `y` (of length ~ total_duration seconds) into
    consecutive chunks of `chunk_duration` seconds.
    """
    max_samples = int(total_duration * sr)
    y = y[:max_samples]

    chunk_samples = int(chunk_duration * sr)
    num_chunks = len(y) // chunk_samples  # drop remainder if not a perfect multiple

    segments = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        segments.append(y[start:end])
    return segments

# --------------------------------
# Feature Extraction
# --------------------------------
def extract_features_from_waveform(y, sr, max_pad_length=300, augment=False):
    """
    Given an in-memory waveform y (and sr), compute the Mel-spectrogram features.
    Returns a (128 x max_pad_length) numpy array or None if something fails.
    """
    try:
        if augment:
            y = augment_audio_waveform(y, sr)

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

        pad_width = max_pad_length - mel_spec_db.shape[1]
        if pad_width > 0:
            mel_spec_db = np.pad(
                mel_spec_db, ((0, 0), (0, pad_width)), mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_length]

        return mel_spec_db

    except Exception:
        return None

def process_audio_file_to_chunks(file_path,
                                 total_duration=30,
                                 chunk_duration=10,
                                 max_pad_length=300,
                                 augment=False):
    """
    Loads the audio preview (up to `total_duration` seconds).
    Splits into chunks of `chunk_duration` seconds.
    Extracts features for each chunk.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=total_duration)
        if y is None or len(y) == 0:
            return []

        segments = split_audio_into_chunks(
            y, sr, total_duration=total_duration, chunk_duration=chunk_duration
        )
        all_features = []
        for seg in segments:
            feat = extract_features_from_waveform(
                seg, sr, max_pad_length=max_pad_length, augment=augment
            )
            if feat is not None:
                all_features.append(feat)

        return all_features

    except Exception:
        return []

def parallel_extract_features_in_chunks(audio_paths,
                                        total_duration=30,
                                        chunk_duration=10,
                                        max_pad_length=300,
                                        augment=False,
                                        num_workers=None):
    """
    Extracts features in parallel for each file, where each file
    can produce multiple chunks.
    """
    process_func = partial(
        process_audio_file_to_chunks,
        total_duration=total_duration,
        chunk_duration=chunk_duration,
        max_pad_length=max_pad_length,
        augment=augment
    )

    all_features = []
    all_file_indices = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)))

    for file_idx, feats_list in enumerate(results):
        for f in feats_list:
            all_features.append(f)
            all_file_indices.append(file_idx)

    return np.array(all_features), all_file_indices

# --------------------------------
# Data Preparation
# --------------------------------
def filter_top_genres(df, top_n=TOP_GENRES):
    """
    Keep only the top_n most frequent genres, 
    then remove 'indie' (too similar to 'rock').
    """
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(top_n).index.tolist()

    # Remove 'indie' from top_genres if it exists
    if 'indie' in top_genres:
        top_genres.remove('indie')

    df_filtered = df[df['genre'].isin(top_genres)].copy()
    return df_filtered

def filter_top_seeds(df, top_n=TOP_SEEDS):
    """
    Limit the seeds to top_n most frequent seeds across the entire dataset.
    """
    # Flatten all seed lists
    seed_list_flat = []
    for s in df['seeds_parsed']:
        seed_list_flat.extend(s)
    # Count frequencies
    from collections import Counter
    seed_counts = Counter(seed_list_flat)
    top_seeds = [s for s, _ in seed_counts.most_common(top_n)]

    # Filter each track's seeds to only keep top seeds
    def keep_top_seeds(seed_list):
        return [s for s in seed_list if s in top_seeds]

    df['seeds_parsed'] = df['seeds_parsed'].apply(keep_top_seeds)
    return df

def prepare_data_with_chunks(df, 
                             scaler, 
                             genre_encoder, 
                             seed_encoder,
                             fit_scaler=False, 
                             augment=False,
                             total_duration=30,
                             chunk_duration=10):
    """
    Extract chunk-based features from the audio previews in df.
    """
    audio_paths = df['audio_previews'].apply(os.path.abspath).to_list()
    X_list, file_indices = parallel_extract_features_in_chunks(
        audio_paths=audio_paths,
        total_duration=total_duration,
        chunk_duration=chunk_duration,
        max_pad_length=MAX_PAD_LENGTH,
        augment=augment,
        num_workers=4
    )

    if len(X_list) == 0:
        raise ValueError("No valid audio features extracted.")

    # Regression labels (V/A/D)
    original_reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    chunk_reg_labels = [original_reg_labels[idx] for idx in file_indices]
    chunk_reg_labels = np.array(chunk_reg_labels)

    # Fit or just transform
    if fit_scaler:
        scaler.fit(chunk_reg_labels)
    scaled_reg_labels = scaler.transform(chunk_reg_labels)

    # Genre labels
    original_genre_labels = genre_encoder.transform(df['genre'].astype(str))
    chunk_genre_labels = [original_genre_labels[idx] for idx in file_indices]
    chunk_genre_labels = np.array(chunk_genre_labels)

    # Seeds (multi-label)
    df_seeds_multi_hot = seed_encoder.transform(df['seeds_parsed'])
    chunk_seed_labels = [df_seeds_multi_hot[idx] for idx in file_indices]
    chunk_seed_labels = np.array(chunk_seed_labels)

    return X_list, scaled_reg_labels, chunk_genre_labels, chunk_seed_labels

# --------------------------------
# Model Building
# --------------------------------
def build_model(input_shape, num_genres, num_seeds, l2_reg=L2_REG, initial_lr=INITIAL_LR):
    """
    Larger CNN + 2-layer BiLSTM + 2x MultiHeadAttention + bigger Dense layers.
    Three heads: regression (3-dim), single-label genre, multi-label seeds.
    """
    optimizer = Adam(learning_rate=initial_lr)
    reg = l2(l2_reg)

    inputs = Input(shape=input_shape)

    # --- CNN feature extractor ---
    # Conv block 1
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv block 2
    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv block 3
    x = Conv2D(256, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv block 4
    x = Conv2D(256, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # shape = [batch, freq, time, channels]
    x = Permute((2, 1, 3))(x)
    time_steps = x.shape[1]
    features = x.shape[2] * x.shape[3]
    x = Reshape((time_steps, features))(x)

    # --- 2-layer BiLSTM ---
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)

    # --- 2x Multi-Head Attention ---
    attn1 = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)  # first self-attention
    x = GlobalAveragePooling1D()(attn1)

    # Optionally, we could do a second MHA with skip-connection:
    # attn2 = MultiHeadAttention(num_heads=4, key_dim=64)(attn1, attn1)
    # x2 = GlobalAveragePooling1D()(attn2)
    # x = tf.keras.layers.Concatenate()([x1, x2])
    # But let's keep it simpler: one global pooling from the first attn1

    # --- Dense layers ---
    x = Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)

    # --- Heads ---
    reg_output = Dense(3, activation='linear', name='reg_output')(x)
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)
    seed_output = Dense(num_seeds, activation='sigmoid', name='seed_output')(x)

    model = Model(inputs, [reg_output, class_output, seed_output])
    model.compile(
        optimizer=optimizer,
        loss={
            'reg_output': 'mse',
            'class_output': 'sparse_categorical_crossentropy',
            'seed_output': 'binary_crossentropy'
        },
        metrics={
            'reg_output': ['mae'],
            'class_output': ['accuracy'],
            'seed_output': [tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
        }
    )
    return model

# --------------------------------
# TF Dataset Creation
# --------------------------------
def create_tf_dataset(X, y_reg, y_class, y_seed, batch_size=32, shuffle=True):
    """Create a tf.data.Dataset that yields
       (X_batch, {'reg_output': y_batch_reg, 'class_output': y_batch_class, 'seed_output': y_batch_seed})."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, {
            'reg_output': y_reg,
            'class_output': y_class,
            'seed_output': y_seed
        })
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --------------------------------
# Plot Functions
# --------------------------------
def plot_training_history(history, output_path='training_history.png'):
    hist = history.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1) Total loss
    axes[0].plot(hist['loss'], label='Train Loss')
    axes[0].plot(hist['val_loss'], label='Val Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 2) Regression MAE
    if 'reg_output_mae' in hist:
        axes[1].plot(hist['reg_output_mae'], label='Train MAE (Reg)')
        axes[1].plot(hist['val_reg_output_mae'], label='Val MAE (Reg)')
        axes[1].set_title('Regression MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()

    # 3) Genre Accuracy
    if 'class_output_accuracy' in hist:
        axes[2].plot(hist['class_output_accuracy'], label='Train Acc (Genre)')
        axes[2].plot(hist['val_class_output_accuracy'], label='Val Acc (Genre)')
        axes[2].set_title('Genre Classification Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()

    # 4) Seeds Binary Accuracy
    if 'seed_output_binary_accuracy' in hist:
        axes[3].plot(hist['seed_output_binary_accuracy'], label='Train Acc (Seeds)')
        axes[3].plot(hist['val_seed_output_binary_accuracy'], label='Val Acc (Seeds)')
        axes[3].set_title('Seeds Binary Accuracy')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Accuracy')
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
    Plots predicted vs. actual for the 3 regression targets in the *original domain*.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Valence', 'Arousal', 'Dominance']

    for i, ax in enumerate(axes):
        ax.scatter(y_true_inv[:, i], y_pred_inv[:, i], alpha=0.4)
        ax.set_xlabel(f"True {target_names[i]}")
        ax.set_ylabel(f"Pred {target_names[i]}")
        ax.set_title(f"{target_names[i]}: True vs. Pred")

        # Diagonal reference line
        min_val = min(y_true_inv[:, i].min(), y_pred_inv[:, i].min())
        max_val = max(y_true_inv[:, i].max(), y_pred_inv[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png")
    plt.close()

def plot_seed_distribution(y_true_seed, y_pred_seed, seed_names, output_path='seed_distribution.png'):
    """
    Plots a bar chart comparing the total number of times each seed
    is present in the ground truth vs. the predicted.
    """
    # Sum across all samples
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

# --------------------------------
# Main Execution
# --------------------------------
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'  # Adjust if needed

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        required_cols = [
            'audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags', 
            'genre', 'seeds'
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        # Convert 'seeds' from string -> list
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

        # Filter top genres and remove 'indie'
        df = filter_top_genres(df, top_n=TOP_GENRES)

        # Limit seeds to top 15
        df = filter_top_seeds(df, top_n=TOP_SEEDS)

        print("Genres after filtering:")
        print(df['genre'].value_counts())

        print("Example seeds after filtering:")
        print(df['seeds_parsed'].head())

        # Label Encoder for genres
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))
        unique_genres = genre_encoder.classes_
        num_genres = len(unique_genres)
        print(f"Number of genres: {num_genres}")
        print("Genres:", unique_genres)

        # Multi-label binarizer for seeds
        seed_encoder = MultiLabelBinarizer()
        seed_encoder.fit(df['seeds_parsed'])
        unique_seeds = seed_encoder.classes_
        num_seeds = len(unique_seeds)
        print(f"Number of unique seeds: {num_seeds}")
        print("Seeds:", unique_seeds)

        # Train/Val/Test Split (no random_state => truly random!)
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
        train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True)
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

        # MinMaxScaler for V/A/D
        scaler = MinMaxScaler()

        # Prepare Data
        X_train, y_train_reg, y_train_genre, y_train_seed = prepare_data_with_chunks(
            train_df, scaler, genre_encoder, seed_encoder, 
            fit_scaler=True, augment=True,
            total_duration=30, chunk_duration=CHUNK_DURATION
        )
        X_val, y_val_reg, y_val_genre, y_val_seed = prepare_data_with_chunks(
            val_df, scaler, genre_encoder, seed_encoder, 
            fit_scaler=False, augment=False,
            total_duration=30, chunk_duration=CHUNK_DURATION
        )
        X_test, y_test_reg, y_test_genre, y_test_seed = prepare_data_with_chunks(
            test_df, scaler, genre_encoder, seed_encoder, 
            fit_scaler=False, augment=False,
            total_duration=30, chunk_duration=CHUNK_DURATION
        )

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create TF Datasets
        train_dataset = create_tf_dataset(X_train, y_train_reg, y_train_genre, y_train_seed,
                                          batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val_reg, y_val_genre, y_val_seed,
                                        batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test_reg, y_test_genre, y_test_seed,
                                         batch_size=BATCH_SIZE, shuffle=False)

        # Build model
        model = build_model(
            input_shape=X_train.shape[1:], 
            num_genres=num_genres, 
            num_seeds=num_seeds,
            l2_reg=L2_REG,
            initial_lr=INITIAL_LR
        )
        model.summary()

        # Callbacks
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)
        ]

        print("Starting training...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        # Save training history
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # Plot training curves
        plot_training_history(history, output_path='training_curves.png')

        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        print("Test Results:", test_results)

        # Predict on test set
        y_pred_reg, y_pred_class, y_pred_seed = model.predict(test_dataset, verbose=1)

        # Invert scale for regression
        y_test_reg_inv = scaler.inverse_transform(y_test_reg)
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)

        # Argmax for genre
        y_pred_genre = np.argmax(y_pred_class, axis=1)

        # Make regression plots (original domain)
        plot_regression_results(y_test_reg_inv, y_pred_reg_inv, output_path_prefix='regression')

        # Genre classification report
        print("\nGenre Classification Report:")
        print(classification_report(
            y_test_genre, y_pred_genre,
            labels=range(num_genres),
            target_names=unique_genres,
            zero_division=0
        ))

        # Genre confusion matrix
        plot_confusion_matrix(y_test_genre, y_pred_genre, unique_genres, 'confusion_matrix.png')

        # Seeds classification: threshold
        seed_threshold = 0.5
        y_pred_seed_binary = (y_pred_seed >= seed_threshold).astype(np.int32)

        # Seeds multi-label classification report (by label)
        print("\nSeeds Multi-label Classification Report:")
        print(classification_report(
            y_test_seed,
            y_pred_seed_binary,
            target_names=unique_seeds,
            zero_division=0
        ))

        # Plot seed distribution
        plot_seed_distribution(y_test_seed, y_pred_seed_binary, unique_seeds, output_path='seed_distribution.png')

        # Save final model
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Sample predictions
        print("\nSample predictions on random test items:")
        test_indices = np.arange(len(y_test_reg))
        np.random.shuffle(test_indices)
        for i in test_indices[:5]:  # 5 random samples
            true_reg_inv = y_test_reg_inv[i]
            pred_reg_inv = y_pred_reg_inv[i]

            true_genre = unique_genres[y_test_genre[i]]
            pred_genre_argmax = unique_genres[y_pred_genre[i]]

            # Seeds
            true_seed_labels = [seed for (seed, val) in zip(unique_seeds, y_test_seed[i]) if val == 1]
            pred_seed_labels = [seed for (seed, val) in zip(unique_seeds, y_pred_seed_binary[i]) if val == 1]

            # Sort probabilities for top-k genres
            pred_probs = y_pred_class[i]
            sorted_idx = np.argsort(pred_probs)[::-1]
            top_3_idx = sorted_idx[:3]

            print(f"\n  Sample index: {i}")
            print(f"    True V/A/D (orig scale):  {true_reg_inv}")
            print(f"    Pred V/A/D (orig scale):  {pred_reg_inv}")
            print(f"    True Genre:               {true_genre}")
            print(f"    Pred Genre (argmax):      {pred_genre_argmax}")
            print(f"    Top predicted genres with confidence:")
            for rank, genre_id in enumerate(top_3_idx, start=1):
                confidence = pred_probs[genre_id]
                print(f"      {rank}) {unique_genres[genre_id]} (confidence={confidence:.4f})")
            print(f"    True Seeds:               {true_seed_labels}")
            print(f"    Pred Seeds:               {pred_seed_labels}")

    except Exception as e:
        print(f"Error during training: {e}")
