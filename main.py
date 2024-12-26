import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional; sometimes helps with certain CPU issues

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
    MaxPooling2D, Bidirectional, LSTM, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------
# Hyperparameters
# --------------------------------
MAX_PAD_LENGTH = 300     # number of frames in Mel spectrogram
TOP_GENRES = 10
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-4        # initial learning rate
L2_REG = 1e-4            # L2 regularization factor

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
def split_audio_into_chunks(y, sr, total_duration=30, chunk_duration=5):
    """
    Splits the waveform `y` (of length ~ total_duration seconds) into
    consecutive chunks of `chunk_duration` seconds.
    
    Returns a list of waveforms.
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
            # Waveform-level augmentation
            y = augment_audio_waveform(y, sr)

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000
        )
        if mel_spec.size == 0:
            return None

        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

        # Optional SpecAugment
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

    except Exception:
        return None

def process_audio_file_to_chunks(file_path,
                                 total_duration=30,
                                 chunk_duration=5,
                                 max_pad_length=300,
                                 augment=False):
    """
    Loads the audio preview (up to `total_duration` seconds).
    Splits into chunks of `chunk_duration` seconds.
    Extracts features for each chunk.
    
    Returns a list of feature arrays, one per chunk.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=total_duration)
        if y is None or len(y) == 0:
            return []

        # Split into segments
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
                                        chunk_duration=5,
                                        max_pad_length=300,
                                        augment=False,
                                        num_workers=None):
    """
    Extracts features in parallel for each file, where each file
    can produce multiple chunks.
    Returns:
      - A numpy array of shape [N_chunks, 128, max_pad_length]
      - A list of file indices mapping each chunk back to its file
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

    # Flatten the results
    for file_idx, feats_list in enumerate(results):
        # feats_list is a list of arrays for each chunk from file_idx
        for f in feats_list:
            all_features.append(f)
            all_file_indices.append(file_idx)

    return np.array(all_features), all_file_indices

# --------------------------------
# Data Preparation
# --------------------------------
def filter_top_genres(df, top_n=TOP_GENRES):
    """
    Keep only the top_n most frequent genres in the dataset.
    """
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(top_n).index
    df_filtered = df[df['genre'].isin(top_genres)].copy()
    return df_filtered

def prepare_data_with_chunks(df, 
                             scaler, 
                             genre_encoder, 
                             fit_scaler=False, 
                             augment=False,
                             total_duration=30,
                             chunk_duration=5):
    """
    Extract chunk-based features from the audio previews in df.
    Each 30s preview is split into multiple chunks (e.g., 5s each).
    Returns:
      - X_list:   np.array of shape [N_chunks, 128, max_pad_length]
      - y_reg:    np.array of shape [N_chunks, 3]  (scaled V/A/D)
      - y_genre:  np.array of shape [N_chunks]     (genre indices)
    """
    audio_paths = df['audio_previews'].apply(os.path.abspath).to_list()

    # 1) Extract features in parallel for all chunks
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

    # 2) Build the regression labels for each chunk (replicate parent's labels)
    original_reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    chunk_reg_labels = [original_reg_labels[idx] for idx in file_indices]
    chunk_reg_labels = np.array(chunk_reg_labels)

    # Fit and transform (or just transform) the regression labels
    if fit_scaler:
        scaler.fit(chunk_reg_labels)
    scaled_reg_labels = scaler.transform(chunk_reg_labels)

    # 3) Build the genre labels for each chunk
    original_genre_labels = genre_encoder.transform(df['genre'].astype(str))
    chunk_genre_labels = [original_genre_labels[idx] for idx in file_indices]
    chunk_genre_labels = np.array(chunk_genre_labels)

    return X_list, scaled_reg_labels, chunk_genre_labels

# --------------------------------
# Model Building
# --------------------------------
def build_model(input_shape, num_genres, l2_reg=L2_REG, initial_lr=INITIAL_LR):
    """
    CNN + BiLSTM + Two-task heads (regression + classification).
    Includes dropout, L2 regularization, and a learning rate schedule.
    """
    # Learning rate schedule: Exponential Decay
    decay_steps = 1000  # Adjust per dataset size
    decay_rate = 0.9
    staircase = True
    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    optimizer = Adam(learning_rate=lr_schedule)
    reg = l2(l2_reg)

    inputs = Input(shape=input_shape)

    # Convolutional block 1
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    # Convolutional block 2
    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    # Convolutional block 3
    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    # Reshape for LSTM
    x = Permute((2, 1, 3))(x)  # [batch, freq, time, channels] -> [batch, time, freq, channels]
    time_steps = x.shape[1]
    features = x.shape[2] * x.shape[3]
    x = Reshape((time_steps, features))(x)

    # BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)

    # Regression head
    reg_output = Dense(3, activation='linear', name='reg_output')(x)

    # Classification head
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)

    model = Model(inputs, [reg_output, class_output])
    model.compile(
        optimizer=optimizer,
        loss={
            'reg_output': 'mse',
            'class_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'reg_output': ['mae'],
            'class_output': ['accuracy']
        }
    )
    return model

# --------------------------------
# TF Dataset Creation
# --------------------------------
def create_tf_dataset(X, y_reg, y_class, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, {'reg_output': y_reg, 'class_output': y_class})
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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

    # 3) Classification Accuracy
    if 'class_output_accuracy' in hist:
        axes[2].plot(hist['class_output_accuracy'], label='Train Acc (Class)')
        axes[2].plot(hist['val_class_output_accuracy'], label='Val Acc (Class)')
        axes[2].set_title('Classification Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Genre Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()

def plot_regression_results(y_true, y_pred, output_path_prefix='regression'):
    """
    Plot predicted vs. actual for the 3 regression targets:
    valence, arousal, dominance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Valence', 'Arousal', 'Dominance']

    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        ax.set_xlabel(f"True {target_names[i]}")
        ax.set_ylabel(f"Predicted {target_names[i]}")
        ax.set_title(f"{target_names[i]}: Predicted vs. True")

        # Diagonal reference line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png")
    plt.close()

# --------------------------------
# Main Execution
# --------------------------------
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'  # Replace with your CSV path

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        required_cols = ['audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags', 'genre']
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        # Filter top genres
        df = filter_top_genres(df, top_n=TOP_GENRES)
        print(f"Filtered to top {TOP_GENRES} genres.")
        print("Genre counts after filtering:")
        print(df['genre'].value_counts())

        # Label Encoder for genres
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))
        unique_genres = genre_encoder.classes_
        num_genres = len(unique_genres)
        print(f"Number of genres: {num_genres}")
        print("Genres:", unique_genres)

        # Train/Val/Test Split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

        # Prepare Data (with chunking)
        scaler = MinMaxScaler()

        # Here we chunk each 30s preview into multiple 5s segments for training
        X_train, y_train_reg, y_train_genre = prepare_data_with_chunks(
            train_df, scaler, genre_encoder, fit_scaler=True, augment=True,
            total_duration=30, chunk_duration=5
        )
        X_val, y_val_reg, y_val_genre = prepare_data_with_chunks(
            val_df, scaler, genre_encoder, fit_scaler=False, augment=False,
            total_duration=30, chunk_duration=5
        )
        X_test, y_test_reg, y_test_genre = prepare_data_with_chunks(
            test_df, scaler, genre_encoder, fit_scaler=False, augment=False,
            total_duration=30, chunk_duration=5
        )

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create TF Datasets
        train_dataset = create_tf_dataset(X_train, y_train_reg, y_train_genre,
                                          batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val_reg, y_val_genre,
                                        batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test_reg, y_test_genre,
                                         batch_size=BATCH_SIZE, shuffle=False)

        # Build and compile the model
        model = build_model(input_shape=X_train.shape[1:],
                            num_genres=num_genres,
                            l2_reg=L2_REG,
                            initial_lr=INITIAL_LR)
        model.summary()

        # Callbacks
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
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
        print("Training history saved to training_history.pkl")

        # Plot training curves
        plot_training_history(history, output_path='training_curves.png')
        print("Training curves saved to training_curves.png")

        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        print("Test Results:", test_results)

        # Predictions on test set
        print("\nPredicting on test set...")
        y_pred_reg, y_pred_class = model.predict(test_dataset, verbose=1)
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)  # invert scaling
        y_pred_genre = np.argmax(y_pred_class, axis=1)

        # Plot regression results
        plot_regression_results(y_test_reg, y_pred_reg_inv, output_path_prefix='regression')
        print("Regression scatter plots saved (regression_scatter.png)")

        # Classification Report
        print("\nGenre Classification Report:")
        print(classification_report(
            y_test_genre, y_pred_genre, labels=range(num_genres), target_names=unique_genres
        ))

        # Confusion Matrix
        plot_confusion_matrix(y_test_genre, y_pred_genre, unique_genres, 'confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")

        # Save final model
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Sample predictions with top 3 genres
        print("\nSample predictions on test set (showing top 3 genres):")
        for i in range(min(5, len(y_test_reg))):
            true_reg = scaler.inverse_transform([y_test_reg[i]])[0]
            pred_reg = y_pred_reg_inv[i]

            true_genre = unique_genres[y_test_genre[i]]

            # Argmax-based predicted genre
            pred_genre_argmax = unique_genres[y_pred_genre[i]]

            # Sort probabilities for top-k
            pred_probs = y_pred_class[i]
            sorted_idx = np.argsort(pred_probs)[::-1]
            top_3_idx = sorted_idx[:3]

            print(f"  Sample {i}:")
            print(f"    True V/A/D:  {true_reg}")
            print(f"    Pred V/A/D:  {pred_reg}")
            print(f"    True Genre:  {true_genre}")
            print(f"    Top predicted genres:")
            for rank, genre_id in enumerate(top_3_idx, start=1):
                confidence = pred_probs[genre_id]
                print(f"      {rank}) {unique_genres[genre_id]} (confidence={confidence:.4f})")

    except Exception as e:
        print(f"Error during training: {e}")
