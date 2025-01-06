import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional CPU setting

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress librosa/mp3 warnings

import ast
import random
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import ( #type: ignore
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Reshape, Permute, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention #type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
MAX_PAD_LENGTH = 300       # spectrogram width
TOP_GENRES = 30
TOP_SEEDS = 30
BATCH_SIZE = 16  # Lower batch size if you have memory issues
EPOCHS = 5       # for demo; adjust as needed
INITIAL_LR = 1e-4
L2_REG = 1e-4
CHUNK_DURATION = 15        # seconds
MAX_AUDIO_DURATION = 1200  # load at most 1200s (~20min)
TARGET_SR = 22050

# ---------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------
def augment_audio_waveform(y, sr):
    """Random time-stretch, pitch-shift, additive noise on the waveform."""
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if np.random.rand() < 0.3:
        steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    return y

def spec_augment(mel_spec_db, num_masks=1, freq_masking_max_percentage=0.10, time_masking_max_percentage=0.10):
    """Mask out random frequency/time segments on the mel-spectrogram."""
    augmented = mel_spec_db.copy()
    num_mels, max_frames = augmented.shape

    for _ in range(num_masks):
        mask_size = int(freq_masking_max_percentage * num_mels)
        start = np.random.randint(0, num_mels - mask_size)
        augmented[start:start+mask_size, :] = 0

    for _ in range(num_masks):
        mask_size = int(time_masking_max_percentage * max_frames)
        start = np.random.randint(0, max_frames - mask_size)
        augmented[:, start:start+mask_size] = 0

    return augmented

# ---------------------------------------------------------------------
# Audio Utility
# ---------------------------------------------------------------------
def split_audio_into_chunks(y, sr, chunk_duration=15):
    """
    Split the waveform y into full chunks of `chunk_duration` seconds.
    Leftover (shorter than chunk_duration) is ignored by default.
    """
    chunk_samples = int(chunk_duration * sr)
    num_chunks = len(y) // chunk_samples
    segments = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        segments.append(y[start:end])
    return segments

def extract_features_from_waveform(y, sr, max_pad_length=300, augment=False):
    """
    Convert a waveform into a log-mel-spectrogram (128 x max_pad_length).
    - If augment=True, apply time-stretch, pitch-shift, noise, spec augment.
    - Return shape (128, max_pad_length) or None if there's an error.
    """
    if augment:
        y = augment_audio_waveform(y, sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    if mel_spec.size == 0:
        return None

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize
    mean = np.mean(mel_spec_db)
    std = np.std(mel_spec_db)
    mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

    if augment:
        mel_spec_db = spec_augment(mel_spec_db)

    # Pad or truncate horizontally
    pad_width = max_pad_length - mel_spec_db.shape[1]
    if pad_width > 0:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_length]

    return mel_spec_db.astype(np.float32)

# ---------------------------------------------------------------------
# Weighted BCE
# ---------------------------------------------------------------------
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """
    For multi-label seeds output. 
    A higher pos_weight will up-weight positive samples.
    """
    def __init__(self, pos_weight=5.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        # formula for weighting positives: w = 1 + (pos_weight - 1) * y_true
        weights = 1.0 + (self.pos_weight - 1.0) * y_true
        return tf.reduce_mean(weights * bce)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_transformer_model(
    input_shape,
    num_genres,
    num_seeds,
    l2_reg=1e-4,
    initial_lr=1e-4,
    seed_pos_weight=5.0
):
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

    # Now shape is (None, 32, 75, 128)
    # Move time to the first dimension for the Transformer
    x = Permute((2, 1, 3))(x)  # => (batch, time=75, freq=32, channels=128)
    time_steps = x.shape[1]
    features_per_timestep = x.shape[2] * x.shape[3]  # 32*128=4096
    x = Reshape((time_steps, features_per_timestep))(x)

    # 2 Transformer layers
    embed_dim = features_per_timestep  # 4096
    ff_dim = 256
    num_heads = 4
    dropout_rate = 0.2

    for _ in range(2):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)

    # Outputs
    reg_output = Dense(3, activation='linear', name='reg_output')(x)  # (Val/Aro/Dom)
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)
    seed_output = Dense(num_seeds, activation='sigmoid', name='seed_output')(x)

    # Losses
    w_bce = WeightedBinaryCrossentropy(pos_weight=seed_pos_weight)
    genre_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model = Model(inputs, [reg_output, class_output, seed_output])
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

# ---------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------
def plot_training_history(history, output_path='training_history.png'):
    hist = history.history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].plot(hist['loss'], label='Train Loss')
    axes[0].plot(hist['val_loss'], label='Val Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()

    if 'reg_output_mae' in hist:
        axes[1].plot(hist['reg_output_mae'], label='Train Reg MAE')
        axes[1].plot(hist['val_reg_output_mae'], label='Val Reg MAE')
        axes[1].set_title('Regression MAE')
        axes[1].legend()

    if 'class_output_accuracy' in hist:
        axes[2].plot(hist['class_output_accuracy'], label='Train Genre Acc')
        axes[2].plot(hist['val_class_output_accuracy'], label='Val Genre Acc')
        axes[2].set_title('Genre Accuracy')
        axes[2].legend()

    if 'seed_output_binary_accuracy' in hist:
        axes[3].plot(hist['seed_output_binary_accuracy'], label='Train Seeds Acc')
        axes[3].plot(hist['val_seed_output_binary_accuracy'], label='Val Seeds Acc')
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Valence', 'Arousal', 'Dominance']

    for i, ax in enumerate(axes):
        ax.scatter(y_true_inv[:, i], y_pred_inv[:, i], alpha=0.3)
        ax.set_xlabel(f"True {target_names[i]}")
        ax.set_ylabel(f"Pred {target_names[i]}")
        ax.set_title(f"{target_names[i]}: True vs. Pred")

        mn = float(min(y_true_inv[:, i].min(), y_pred_inv[:, i].min()))
        mx = float(max(y_true_inv[:, i].max(), y_pred_inv[:, i].max()))
        ax.plot([mn, mx], [mn, mx], 'r--')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png")
    plt.close()

def plot_seed_distribution(y_true_seed, y_pred_seed, seed_names, output_path='seed_distribution.png'):
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

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def keep_top(lst, top_seeds):
    if not isinstance(lst, list):
        return []
    return [x for x in lst if x in top_seeds]

# Encode seeds to a fixed-size multi-hot vector
def encode_seed_list(seed_list, seed_encoder):
    arr = seed_encoder.transform([seed_list])  # shape => (1, num_seeds)
    return arr[0].astype(np.float32)

# ---------------------------------------------------------------------
# Python Generator for Audio Chunks
# ---------------------------------------------------------------------
def audio_generator(
    df,
    genre_encoder,
    seed_encoder,
    top_seeds,
    max_pad_length=300,
    chunk_duration=15,
    sr=22050,
    max_duration=1200,
    augment=False
):
    """
    Pure Python generator that:
      1) Iterates rows in df
      2) Loads audio
      3) Splits into chunks
      4) Extracts log-mel features
      5) Yields (features, {'reg_output':..., 'class_output':..., 'seed_output':...})
    """
    for i, row in df.iterrows():
        file_path = row['audio_previews']
        valence = float(row['valence_tags'])
        arousal = float(row['arousal_tags'])
        dominance = float(row['dominance_tags'])
        reg_targets = np.array([valence, arousal, dominance], dtype=np.float32)

        genre_str = str(row['genre'])
        genre_idx = int(genre_encoder.transform([genre_str])[0])

        # multi-hot seed vector
        seed_vec = row['seed_vector']  # already multi-hot
        # or row['seeds_parsed'] => then encode with seed_encoder if you prefer

        try:
            y, _ = librosa.load(file_path, sr=sr, mono=True, duration=max_duration)
        except Exception as e:
            # skip if cannot load
            print(f"[WARN] Could not load audio {file_path}: {e}")
            continue

        if y is None or len(y) == 0:
            continue

        segments = split_audio_into_chunks(y, sr, chunk_duration=chunk_duration)
        for seg in segments:
            feat = extract_features_from_waveform(
                seg, sr=sr,
                max_pad_length=max_pad_length,
                augment=augment
            )
            if feat is None:
                continue
            # Expand dims => (128, max_pad_length, 1)
            feat = feat[..., np.newaxis]
            if feat.shape != (128, max_pad_length, 1):
                continue

            yield (
                feat,
                {
                    'reg_output': reg_targets,
                    'class_output': np.array(genre_idx, dtype=np.int32),
                    'seed_output': seed_vec,  # shape => (num_seeds,)
                }
            )

# ---------------------------------------------------------------------
# Build a tf.data.Dataset from the generator
# ---------------------------------------------------------------------
def build_dataset(
    df,
    genre_encoder,
    seed_encoder,
    top_seeds,
    batch_size=16,
    augment=False
):
    # We'll define the output signatures to match exactly the shapes & dtypes
    #   features: tf.float32, shape=(128, 300, 1)
    #   labels: {
    #       'reg_output': tf.float32, shape=(3,),
    #       'class_output': tf.int32, shape=(),
    #       'seed_output': tf.float32, shape=(num_seeds,)
    #   }
    num_seeds = len(top_seeds)
    output_types = (
        tf.float32,
        {
            'reg_output': tf.float32,
            'class_output': tf.int32,
            'seed_output': tf.float32
        }
    )
    output_shapes = (
        (128, MAX_PAD_LENGTH, 1),
        {
            'reg_output': (3,),
            'class_output': (),
            'seed_output': (num_seeds,)
        }
    )

    # Create the generator (no arguments for the generator function,
    # so we wrap it with a lambda)
    def gen():
        return audio_generator(
            df=df,
            genre_encoder=genre_encoder,
            seed_encoder=seed_encoder,
            top_seeds=top_seeds,
            max_pad_length=MAX_PAD_LENGTH,
            chunk_duration=CHUNK_DURATION,
            sr=TARGET_SR,
            max_duration=MAX_AUDIO_DURATION,
            augment=augment
        )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=output_types,
        output_shapes=output_shapes
    )
    ds = ds.shuffle(1000, reshuffle_each_iteration=True) if augment else ds
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'
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

        # parse seeds from string repr
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

        # Filter top genres
        genre_counts = df['genre'].value_counts()
        top_genres = genre_counts.head(TOP_GENRES).index.tolist()
        # example: remove 'indie' from top_genres if needed
        if 'indie' in top_genres:
            top_genres.remove('indie')
        df = df[df['genre'].isin(top_genres)]

        # Filter top seeds
        seed_list_flat = []
        for seeds in df['seeds_parsed']:
            seed_list_flat.extend(seeds)
        seed_counts = Counter(seed_list_flat)
        top_seeds = [s for s, _ in seed_counts.most_common(TOP_SEEDS)]
        df['seeds_parsed'] = df['seeds_parsed'].apply(lambda x: keep_top(x, top_seeds))

        # Build label encoders
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))
        unique_genres = genre_encoder.classes_
        num_genres = len(unique_genres)

        seed_encoder = MultiLabelBinarizer(classes=top_seeds)
        seed_encoder.fit(df['seeds_parsed'])
        unique_seeds = seed_encoder.classes_
        num_seeds = len(unique_seeds)

        # Encode seeds as fixed-size vectors
        df['seed_vector'] = df['seeds_parsed'].apply(lambda s: encode_seed_list(s, seed_encoder))

        # Train/Val/Test split
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED)
        train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True, random_state=SEED)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print("Genres:", unique_genres)
        print("Seeds:", unique_seeds)

        # Build tf.data pipelines
        train_dataset = build_dataset(train_df, genre_encoder, seed_encoder, unique_seeds, batch_size=BATCH_SIZE, augment=True)
        val_dataset   = build_dataset(val_df,   genre_encoder, seed_encoder, unique_seeds, batch_size=BATCH_SIZE, augment=False)
        test_dataset  = build_dataset(test_df,  genre_encoder, seed_encoder, unique_seeds, batch_size=BATCH_SIZE, augment=False)

        # Build model
        model = build_transformer_model(
            input_shape=(128, MAX_PAD_LENGTH, 1),
            num_genres=num_genres,
            num_seeds=num_seeds,
            l2_reg=L2_REG,
            initial_lr=INITIAL_LR,
            seed_pos_weight=5.0
        )
        model.summary()

        # Checkpoints
        ckpt_dir = 'model_checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]

        print("Starting training...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        plot_training_history(history, 'training_curves.png')

        print("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        print("Test Results:", test_results)

        # Collect predictions on the test set
        y_true_reg_all = []
        y_true_genre_all = []
        y_true_seed_all = []
        y_pred_reg_all = []
        y_pred_genre_all = []
        y_pred_seed_all = []

        for X_batch, y_batch in test_dataset:
            preds = model.predict_on_batch(X_batch)
            pred_reg, pred_genre, pred_seed = preds

            y_pred_reg_all.append(pred_reg)
            y_pred_genre_all.append(pred_genre)
            y_pred_seed_all.append(pred_seed)

            y_true_reg_all.append(y_batch['reg_output'].numpy())
            y_true_genre_all.append(y_batch['class_output'].numpy())
            y_true_seed_all.append(y_batch['seed_output'].numpy())

        # Convert lists to arrays
        y_true_reg_all = np.concatenate(y_true_reg_all, axis=0)
        y_true_genre_all = np.concatenate(y_true_genre_all, axis=0)
        y_true_seed_all = np.concatenate(y_true_seed_all, axis=0)
        y_pred_reg_all = np.concatenate(y_pred_reg_all, axis=0)
        y_pred_genre_all = np.concatenate(y_pred_genre_all, axis=0)
        y_pred_seed_all = np.concatenate(y_pred_seed_all, axis=0)

        # Plot regression
        plot_regression_results(y_true_reg_all, y_pred_reg_all, 'regression')

        # Genre classification
        y_pred_genre_argmax = np.argmax(y_pred_genre_all, axis=1)
        print("\nGenre Classification Report:")
        print(classification_report(
            y_true_genre_all, y_pred_genre_argmax,
            target_names=unique_genres, zero_division=0
        ))
        plot_confusion_matrix(y_true_genre_all, y_pred_genre_argmax, unique_genres)

        # Seeds multi-label
        seed_threshold = 0.5
        y_pred_seed_bin = (y_pred_seed_all >= seed_threshold).astype(int)
        print("\nSeeds Multi-label Classification Report:")
        print(classification_report(
            y_true_seed_all, y_pred_seed_bin,
            target_names=unique_seeds, zero_division=0
        ))
        plot_seed_distribution(y_true_seed_all, y_pred_seed_bin, unique_seeds, 'seed_distribution.png')

        # Save final model
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Show random predictions
        indices = np.arange(len(y_true_reg_all))
        np.random.shuffle(indices)
        for i in indices[:5]:
            true_reg = y_true_reg_all[i]
            pred_reg = y_pred_reg_all[i]
            true_genre_idx = int(y_true_genre_all[i])
            pred_genre_idx = int(y_pred_genre_argmax[i])
            true_genre = unique_genres[true_genre_idx]
            pred_genre_str = unique_genres[pred_genre_idx]

            true_seed_labels = [
                s for s, val in zip(unique_seeds, y_true_seed_all[i]) if val == 1
            ]
            pred_seed_labels = [
                s for s, val in zip(unique_seeds, y_pred_seed_bin[i]) if val == 1
            ]

            pred_probs = y_pred_genre_all[i]
            sorted_idx = np.argsort(pred_probs)[::-1]
            top_3_idx = sorted_idx[:3]

            print(f"\nSample index: {i}")
            print(f"  True V/A/D:  {true_reg}")
            print(f"  Pred V/A/D:  {pred_reg}")
            print(f"  True Genre:  {true_genre}")
            print(f"  Pred Genre:  {pred_genre_str}")
            print("  Top 3 predicted genres w/ confidence:")
            for rank, g_id in enumerate(top_3_idx, 1):
                print(f"    {rank}) {unique_genres[g_id]} (conf={pred_probs[g_id]:.4f})")
            print(f"  True Seeds:  {true_seed_labels}")
            print(f"  Pred Seeds:  {pred_seed_labels}")

    except Exception as e:
        print(f"Error during training: {e}")
