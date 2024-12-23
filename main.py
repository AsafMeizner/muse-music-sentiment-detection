import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional; sometimes helps with certain CPU issues

import random
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore 
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Bidirectional, LSTM, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay # type: ignore
from tensorflow.keras.callbacks import ( # type: ignore
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Hyperparameters
# ----------------------------
MAX_PAD_LENGTH = 300     # number of frames in mel spectrogram
TOP_GENRES = 10
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-4        # initial learning rate
L2_REG = 1e-4            # L2 regularization factor

# ----------------------------
# Data Augmentation
# ----------------------------
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

    # Frequency Masking
    for _ in range(num_masks):
        mask_size = int(freq_masking_max_percentage * num_mels)
        start = np.random.randint(0, num_mels - mask_size)
        augmented[start:start+mask_size, :] = 0

    # Time Masking
    for _ in range(num_masks):
        mask_size = int(time_masking_max_percentage * max_frames)
        start = np.random.randint(0, max_frames - mask_size)
        augmented[:, start:start+mask_size] = 0

    return augmented

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_audio_features(file_path, max_pad_length=300, augment=False):
    """
    Load audio, do optional waveform augmentation, compute Mel spectrogram (128xT),
    normalize, pad, optionally do SpecAugment.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)
        if y is None or len(y) == 0:
            return None
        
        if augment:
            # Waveform augment
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

def process_audio_file(file_path, max_pad_length=300, augment=False):
    return extract_audio_features(file_path, max_pad_length=max_pad_length, augment=augment)

def parallel_extract_features(audio_paths, max_pad_length=300, augment=False, num_workers=None):
    process_func = partial(process_audio_file, max_pad_length=max_pad_length, augment=augment)
    features_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for feat in tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)):
            features_list.append(feat)
    # Filter out None
    filtered_features = [f for f in features_list if f is not None]
    if len(filtered_features) < len(features_list):
        print(f"[Warning] Some files failed. Processed {len(filtered_features)} out of {len(features_list)}.")
    return np.array(filtered_features)

# ----------------------------
# Data Preparation
# ----------------------------
def filter_top_genres(df, top_n=TOP_GENRES):
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(top_n).index
    df_filtered = df[df['genre'].isin(top_genres)].copy()
    return df_filtered

def prepare_data(df, scaler, genre_encoder, fit_scaler=False, augment=False):
    audio_paths = df['audio_previews'].apply(os.path.abspath).to_list()
    X = parallel_extract_features(
        audio_paths, max_pad_length=MAX_PAD_LENGTH, augment=augment, num_workers=4
    )
    if X.size == 0:
        raise ValueError("No valid audio features extracted.")

    # Regression labels
    reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    if fit_scaler:
        scaler.fit(reg_labels)
    scaled_reg_labels = scaler.transform(reg_labels)

    # Genre labels
    genre_labels = genre_encoder.transform(df['genre'].astype(str))
    return X, scaled_reg_labels, genre_labels

# ----------------------------
# Model Building
# ----------------------------
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

    reg = l2(l2_reg)  # L2 regularizer

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

    # Convolutional block 3 (optional: you can remove to reduce complexity)
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

# ----------------------------
# TF Dataset Creation
# ----------------------------
def create_tf_dataset(X, y_reg, y_class, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, {'reg_output': y_reg, 'class_output': y_class})
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ----------------------------
# Plot Functions
# ----------------------------
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
        ax.set_title(f"{target_names[i]}: Predicted vs True")

        # If you want a diagonal reference line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png")
    plt.close()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'  # replace with your CSV path

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

        # Prepare Data
        scaler = MinMaxScaler()
        X_train, y_train_reg, y_train_genre = prepare_data(train_df, scaler, genre_encoder, fit_scaler=True, augment=True)
        X_val, y_val_reg, y_val_genre = prepare_data(val_df, scaler, genre_encoder, fit_scaler=False, augment=False)
        X_test, y_test_reg, y_test_genre = prepare_data(test_df, scaler, genre_encoder, fit_scaler=False, augment=False)

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create TF Datasets
        train_dataset = create_tf_dataset(X_train, y_train_reg, y_train_genre, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val_reg, y_val_genre, batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test_reg, y_test_genre, batch_size=BATCH_SIZE, shuffle=False)

        # Build and Compile Model
        model = build_model(input_shape=X_train.shape[1:], num_genres=num_genres, l2_reg=L2_REG, initial_lr=INITIAL_LR)
        model.summary()

        # Callbacks
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
            # You can choose between ReduceLROnPlateau or rely solely on ExponentialDecay
            # If you want both, keep in mind they can conflict. For demonstration, let's keep it:
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
        y_pred_reg, y_pred_class = model.predict(test_dataset, verbose=1)
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)  # invert scaling
        y_pred_genre = np.argmax(y_pred_class, axis=1)

        # Regression Scatter Plots
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

        # Show sample predictions
        print("\nSample predictions on test set:")
        for i in range(min(5, len(y_test_reg))):
            true_reg = scaler.inverse_transform([y_test_reg[i]])
            pred_reg = y_pred_reg_inv[i]
            true_genre = unique_genres[y_test_genre[i]]
            pred_genre = unique_genres[y_pred_genre[i]]
            print(f"  Sample {i}:")
            print(f"    True V/A/D: {true_reg[0]} | Predicted V/A/D: {pred_reg}")
            print(f"    True Genre: {true_genre} | Predicted Genre: {pred_genre}")

    except Exception as e:
        print(f"Error during training: {e}")
