import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optionally disable oneDNN optimizations

import random
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Bidirectional, LSTM, Permute, Reshape
    # Optionally add attention layer if desired
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#############################################
# Hyperparameters
#############################################
MAX_PAD_LENGTH = 300
TOP_GENRES = 10  # Adjust as needed for your dataset
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

#############################################
# Data Augmentation for Training
#############################################
def augment_audio(y, sr):
    # Simple augmentation: random time-stretch, pitch shift, and noise
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    if np.random.rand() > 0.5:
        steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    return y

#############################################
# Audio Feature Extraction
#############################################
def extract_audio_features_librosa(file_path, max_pad_length=300, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)
        if y is None or len(y) == 0:
            return None

        if augment:
            y = augment_audio(y, sr)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        if mel_spec is None or mel_spec.size == 0:
            return None
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

        pad_width = max(0, max_pad_length - mel_spec_db.shape[1])
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_length]

        return mel_spec_db
    except Exception:
        return None

def process_audio_file(file_path, max_pad_length=300, augment=False):
    return extract_audio_features_librosa(file_path, max_pad_length=max_pad_length, augment=augment)

def parallel_extract_audio_features(audio_paths, max_pad_length=300, augment=False, num_workers=None):
    process_func = partial(process_audio_file, max_pad_length=max_pad_length, augment=augment)
    features_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for feat in tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)):
            features_list.append(feat)

    filtered_features = [f for f in features_list if f is not None]
    if len(filtered_features) < len(features_list):
        print(f"[Warning] Some audio files could not be processed. "
              f"Processed {len(filtered_features)} out of {len(features_list)}.")
    return np.array(filtered_features)

#############################################
# Data Preparation
#############################################
def filter_top_genres(df, top_n=10):
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(top_n).index
    df_filtered = df[df['genre'].isin(top_genres)].copy()
    return df_filtered

def prepare_data(df, scaler, genre_encoder, fit_scaler=False, augment=False):
    audio_paths = df['audio_previews'].apply(os.path.abspath).to_list()
    audio_features = parallel_extract_audio_features(audio_paths, max_pad_length=MAX_PAD_LENGTH, augment=augment, num_workers=4)

    if audio_features.size == 0:
        raise ValueError("No valid audio features extracted. Check your audio files and paths.")

    reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    if fit_scaler:
        scaler.fit(reg_labels)
    scaled_reg_labels = scaler.transform(reg_labels)

    genre_labels = genre_encoder.transform(df['genre'].astype(str))

    return audio_features, scaled_reg_labels, genre_labels

#############################################
# Model Building
#############################################
def build_model(input_shape, num_genres):
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(64, kernel_size=(3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Reshape for LSTM
    x = Permute((2, 1, 3))(x)
    x_shape = x.shape
    time_steps = x_shape[1]
    features = x_shape[2] * x_shape[3]
    x = Reshape((time_steps, features))(x)

    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.3))(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Regression head
    reg_output = Dense(3, activation='linear', name='reg_output')(x)

    # Classification head
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)

    model = Model(inputs, [reg_output, class_output])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss={'reg_output':'mse', 'class_output':'sparse_categorical_crossentropy'},
                  metrics={'reg_output':['mae'], 'class_output':['accuracy']})
    return model

#############################################
# TF Dataset Creation
#############################################
def create_tf_dataset(features, reg_labels, class_labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, {'reg_output': reg_labels, 'class_output': class_labels}))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#############################################
# Plot Functions
#############################################
def plot_training_history(history, output_path='training_history.png'):
    history_dict = history.history
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    # Plot total loss
    axes[0].plot(history_dict['loss'], label='Train Loss')
    axes[0].plot(history_dict['val_loss'], label='Val Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot regression MAE
    if 'reg_output_mae' in history_dict:
        axes[1].plot(history_dict['reg_output_mae'], label='Train MAE (Reg)')
        axes[1].plot(history_dict['val_reg_output_mae'], label='Val MAE (Reg)')
        axes[1].set_title('Regression MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()

    # Plot classification accuracy
    if 'class_output_accuracy' in history_dict:
        axes[2].plot(history_dict['class_output_accuracy'], label='Train Acc (Class)')
        axes[2].plot(history_dict['val_class_output_accuracy'], label='Val Acc (Class)')
        axes[2].set_title('Classification Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Genre Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'  # Ensure this path is correct
    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        required_columns = ['audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags', 'genre']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataset must contain column: {col}")

        # Filter to top genres
        df = filter_top_genres(df, top_n=TOP_GENRES)
        print(f"Dataset filtered to top {TOP_GENRES} genres.")
        print("Genre counts:")
        print(df['genre'].value_counts())

        # Fit LabelEncoder on filtered dataset
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))
        unique_genres = genre_encoder.classes_
        num_genres = len(unique_genres)
        print(f"Number of classes after filtering: {num_genres}")
        print("Genres:", unique_genres)

        # Train/Val/Test split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Testing samples: {len(test_df)}")

        # Prepare data
        scaler = MinMaxScaler()
        X_train, y_train_reg, y_train_genre = prepare_data(train_df, scaler, genre_encoder, fit_scaler=True, augment=True)
        X_val, y_val_reg, y_val_genre = prepare_data(val_df, scaler, genre_encoder, fit_scaler=False, augment=False)
        X_test, y_test_reg, y_test_genre = prepare_data(test_df, scaler, genre_encoder, fit_scaler=False, augment=False)

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create TF datasets
        train_dataset = create_tf_dataset(X_train, y_train_reg, y_train_genre, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val_reg, y_val_genre, batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test_reg, y_test_genre, batch_size=BATCH_SIZE, shuffle=False)

        # Build model
        model = build_model(X_train.shape[1:], num_genres)

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
        history_path = 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"Training history saved to {history_path}")

        # Plot training curves
        plot_training_history(history, output_path='training_curves.png')
        print("Training curves saved to training_curves.png")

        # Evaluate on test set
        print("Evaluating the model on the test set...")
        results = model.evaluate(test_dataset, verbose=1)
        print("Test Results:", results)

        # Predictions on test set
        y_pred_reg, y_pred_class = model.predict(test_dataset, verbose=1)
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)
        y_pred_genre = np.argmax(y_pred_class, axis=1)

        # Classification report
        print("Genre Classification Report:")
        print(classification_report(y_test_genre, y_pred_genre, labels=range(num_genres), target_names=unique_genres))

        # Confusion matrix
        plot_confusion_matrix(y_test_genre, y_pred_genre, classes=unique_genres, output_path='confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")

        # Save final model
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Show sample predictions
        print("Sample predictions on test set:")
        for i in range(min(5, len(y_test_reg))):
            true_reg = scaler.inverse_transform([y_test_reg[i]])
            pred_reg = y_pred_reg_inv[i]
            true_genre = unique_genres[y_test_genre[i]]
            pred_gen = unique_genres[y_pred_genre[i]]
            print(f"Sample {i}:")
            print(f"  True Val/Aro/Dom: {true_reg[0]} | Pred Val/Aro/Dom: {pred_reg}")
            print(f"  True Genre: {true_genre} | Predicted Genre: {pred_gen}")

    except Exception as e:
        print(f"Error during training: {e}")