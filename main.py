import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optionally disable oneDNN optimizations if needed

import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Bidirectional, LSTM, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#############################################
# Audio Feature Extraction
#############################################
def extract_audio_features_librosa(file_path, max_pad_length=300):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)
        if y is None or len(y) == 0:
            print(f"[Warning] Loaded audio is empty for {file_path}")
            return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        if mel_spec is None or mel_spec.size == 0:
            print(f"[Warning] Mel-spectrogram is empty for {file_path}")
            return None
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize per sample
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

        # Pad/Trim
        pad_width = max(0, max_pad_length - mel_spec_db.shape[1])
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_length]

        return mel_spec_db
    except Exception as e:
        print(f"[Error] Processing {file_path}: {e}")
        return None

def process_audio_file(file_path, max_pad_length=300):
    return extract_audio_features_librosa(file_path, max_pad_length=max_pad_length)

def parallel_extract_audio_features(audio_paths, max_pad_length=300, num_workers=None):
    process_func = partial(process_audio_file, max_pad_length=max_pad_length)
    features_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for feat in tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)):
            features_list.append(feat)

    # Filter out None entries
    filtered_features = [f for f in features_list if f is not None]
    if len(filtered_features) < len(features_list):
        print(f"[Warning] Some audio files could not be processed. "
              f"Processed {len(filtered_features)} out of {len(features_list)}.")
    return np.array(filtered_features)

#############################################
# Data Preparation
#############################################
def prepare_data(df, max_pad_length=300, fit_scaler=False, scaler=None, genre_encoder=None):
    audio_paths = [os.path.abspath(preview) for preview in df['audio_previews']]
    audio_features = parallel_extract_audio_features(audio_paths, max_pad_length=max_pad_length, num_workers=4)

    if audio_features.size == 0:
        raise ValueError("No valid audio features extracted. Check your audio files and paths.")

    # Extract regression labels
    reg_labels = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    if fit_scaler:
        scaler = MinMaxScaler()
        scaler.fit(reg_labels)
    scaled_reg_labels = scaler.transform(reg_labels)

    # Encode genre labels with the pre-fitted encoder
    genre_labels = genre_encoder.transform(df['genre'].astype(str))

    return audio_features, scaled_reg_labels, genre_labels, scaler, genre_encoder

#############################################
# Model Building
#############################################
def build_model(input_shape, num_genres):
    inputs = Input(shape=input_shape)

    # Convolutional Layers
    x = Conv2D(32, kernel_size=(3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Reshape for LSTM
    x = Permute((2, 1, 3))(x)
    x_shape = x.shape
    time_steps = x_shape[1]
    features = x_shape[2] * x_shape[3]
    x = Reshape((time_steps, features))(x)

    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Regression head for valence, arousal, dominance
    reg_output = Dense(3, activation='linear', name='reg_output')(x)

    # Classification head for genre
    class_output = Dense(num_genres, activation='softmax', name='class_output')(x)

    model = Model(inputs, [reg_output, class_output])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss={'reg_output':'mse', 'class_output':'sparse_categorical_crossentropy'},
                  metrics={'reg_output':['mae'], 'class_output':['accuracy']})
    model.summary()
    return model

#############################################
# TensorFlow Dataset Creation
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
    if 'reg_output_mae' in history_dict and 'val_reg_output_mae' in history_dict:
        axes[1].plot(history_dict['reg_output_mae'], label='Train MAE (Reg)')
        axes[1].plot(history_dict['val_reg_output_mae'], label='Val MAE (Reg)')
        axes[1].set_title('Regression MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()

    # Plot classification accuracy
    if 'class_output_accuracy' in history_dict and 'val_class_output_accuracy' in history_dict:
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
    cm = confusion_matrix(y_true, y_pred)
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
    dataset_path = 'filtered_dataset.csv'
    max_pad_length = 300

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        required_columns = ['audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags', 'genre']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataset must contain column: {col}")

        # Print raw label stats for regression
        print("Raw label stats (Regression):")
        print(df[['valence_tags', 'arousal_tags', 'dominance_tags']].describe())

        # Fit LabelEncoder on full dataset genres to avoid unseen labels
        genre_encoder = LabelEncoder()
        genre_encoder.fit(df['genre'].astype(str))

        unique_genres = genre_encoder.classes_
        print(f"Number of unique genres: {len(unique_genres)}")
        print("Genres:", unique_genres)

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Testing samples: {len(test_df)}")

        # Prepare training data
        X_train, y_train_reg, y_train_genre, scaler, _ = prepare_data(train_df, fit_scaler=True, max_pad_length=max_pad_length, genre_encoder=genre_encoder)

        # Prepare validation data
        X_val, y_val_reg, y_val_genre, _, _ = prepare_data(val_df, fit_scaler=False, scaler=scaler, genre_encoder=genre_encoder, max_pad_length=max_pad_length)

        # Prepare test data
        X_test, y_test_reg, y_test_genre, _, _ = prepare_data(test_df, fit_scaler=False, scaler=scaler, genre_encoder=genre_encoder, max_pad_length=max_pad_length)

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create TensorFlow datasets
        print("Creating TensorFlow datasets...")
        train_dataset = create_tf_dataset(X_train, y_train_reg, y_train_genre, batch_size=32, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val_reg, y_val_genre, batch_size=32, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test_reg, y_test_genre, batch_size=32, shuffle=False)

        # Build the model
        print("Building the CRNN model...")
        num_genres = len(unique_genres)
        input_shape = X_train.shape[1:]
        model = build_model(input_shape, num_genres)

        # Callbacks
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        print("Starting training...")
        history = model.fit(
            train_dataset,
            epochs=50,
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

        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        results = model.evaluate(test_dataset)
        # results: [total_loss, reg_output_loss, class_output_loss, reg_output_mae, class_output_accuracy]
        print("Test Results:", results)

        # Predictions on test set
        y_pred_reg, y_pred_class = model.predict(test_dataset)
        # Inverse transform regression predictions
        y_pred_reg_inv = scaler.inverse_transform(y_pred_reg)

        # Get predicted genres
        y_pred_genre = np.argmax(y_pred_class, axis=1)

        # Classification report
        print("Genre Classification Report:")
        print(classification_report(y_test_genre, y_pred_genre, target_names=unique_genres))

        # Plot confusion matrix
        plot_confusion_matrix(y_test_genre, y_pred_genre, classes=unique_genres, output_path='confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")

        # Save the final model
        final_model_path = 'final_multitask_model.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

        # Show some sample predictions
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
