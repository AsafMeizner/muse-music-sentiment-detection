# train_model.py

import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Bidirectional, LSTM, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import pickle  # For saving the scaler

from tensorflow.keras.utils import register_keras_serializable # type: ignore
from tensorflow.keras.layers import Layer # type: ignore

#############################################
# Custom Attention Layer
#############################################
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.score_dense = Dense(self.units, activation='tanh')
        self.output_dense = Dense(1)

    def call(self, inputs):
        # inputs shape: (batch, time, features)
        scores = self.output_dense(self.score_dense(inputs))  # (batch, time, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)     # (batch, time, 1)
        weighted_sum = inputs * attention_weights
        context_vector = tf.reduce_sum(weighted_sum, axis=1)  # (batch, features)
        return context_vector

#############################################
# Audio Feature Extraction
#############################################
def extract_audio_features_librosa(file_path, max_pad_length=300, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)

        # Augmentation if training
        if augment:
            if np.random.rand() > 0.5:
                rate = np.random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=rate)
            if np.random.rand() > 0.5:
                steps = np.random.randint(-2, 3)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.005, y.shape)
                y = y + noise

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)

        # Pad/Trim
        pad_width = max(0, max_pad_length - mel_spec_db.shape[1])
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_length]

        return mel_spec_db

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((128, max_pad_length))

def process_audio_file(file_path, max_pad_length=300, augment=False):
    return extract_audio_features_librosa(file_path, max_pad_length=max_pad_length, augment=augment)

def parallel_extract_audio_features(audio_paths, max_pad_length=300, augment=False, num_workers=None):
    process_func = partial(process_audio_file, max_pad_length=max_pad_length, augment=augment)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)))
    return np.array(results)

#############################################
# Data Preparation
#############################################
def prepare_data(df, save_path, augment=False, max_pad_length=300):
    """
    Extracts audio features and scales labels. Saves the processed data and scaler.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        save_path (str): Path to save the processed data (.npz file).
        augment (bool): Whether to perform data augmentation.
        max_pad_length (int): Maximum length for padding/trimming audio features.

    Returns:
        Tuple of numpy arrays: (features, labels)
    """
    if os.path.exists(save_path):
        print(f"Loading prepared data from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        return data['features'], data['labels']

    print(f"Extracting audio features for {save_path}...")
    audio_paths = [os.path.abspath(preview) for preview in df['audio_previews']]
    audio_features = parallel_extract_audio_features(audio_paths, max_pad_length=max_pad_length, augment=augment)

    # Scale labels
    scaler = MinMaxScaler()
    emotion_labels = scaler.fit_transform(df[['valence_tags', 'arousal_tags', 'dominance_tags']])

    # Save the scaler for inverse transformation during testing
    scaler_path = os.path.splitext(save_path)[0] + '_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Save features and labels
    np.savez_compressed(save_path, features=audio_features, labels=emotion_labels)
    print(f"Prepared data saved to {save_path}")

    return audio_features, emotion_labels

#############################################
# Model Building
#############################################
def build_model(input_shape):
    """
    Builds and compiles the CRNN model with an attention layer.

    Parameters:
        input_shape (tuple): Shape of the input data (frequency, time, channels).

    Returns:
        Compiled Keras model.
    """
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

    x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Reshaping for LSTM
    x = Permute((2, 1, 3))(x)  # (batch, time, freq, channels)
    x_shape = x.shape
    time_steps = x_shape[1]
    features = x_shape[2] * x_shape[3]
    x = Reshape((time_steps, features))(x)  # (batch, time, features)

    # Recurrent Layer
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    # Attention Layer
    x = AttentionLayer()(x)  # (batch, features)

    # Fully Connected Layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(3, activation='linear')(x)

    # Model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    model.summary()
    return model

#############################################
# TensorFlow Dataset Creation
#############################################
def create_tf_dataset(features, labels, batch_size=32, shuffle=True):
    """
    Creates a TensorFlow dataset from features and labels.

    Parameters:
        features (np.ndarray): Audio features.
        labels (np.ndarray): Emotion labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Prepared dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'  # Path to your dataset CSV
    max_pad_length = 300  # Adjust based on your data

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        # Verify required columns
        required_columns = ['audio_previews', 'valence_tags', 'arousal_tags', 'dominance_tags']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")

        # Split dataset into training and testing
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

        # Prepare training data with augmentation
        train_data_path = 'train_data.npz'
        X_train, y_train = prepare_data(train_df, save_path=train_data_path, augment=True, max_pad_length=max_pad_length)

        # Prepare testing data without augmentation
        test_data_path = 'test_data.npz'
        X_test, y_test = prepare_data(test_df, save_path=test_data_path, augment=False, max_pad_length=max_pad_length)

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        print(f"Training samples after split: {len(X_train)}, Validation samples: {len(X_val)}")

        # Create TensorFlow datasets
        print("Creating TensorFlow datasets...")
        train_dataset = create_tf_dataset(X_train, y_train, batch_size=32, shuffle=True)
        val_dataset = create_tf_dataset(X_val, y_val, batch_size=32, shuffle=False)
        test_dataset = create_tf_dataset(X_test, y_test, batch_size=32, shuffle=False)

        # Build the model
        print("Building the CRNN model with Attention...")
        input_shape = X_train.shape[1:]  # (frequency, time, channels)
        model = build_model(input_shape)

        # Setup checkpoints and callbacks
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "crnn_attention_best.keras")

        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
            EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        ]

        # Train the model
        print("Starting training...")
        history = model.fit(
            train_dataset,
            epochs=15,  # Increase epochs as needed
            validation_data=val_dataset,
            callbacks=callbacks
        )

        # Save training history for visualization (optional)
        history_path = 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"Training history saved to {history_path}")

        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        loss, mae = model.evaluate(test_dataset)
        print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

        # Save the final model
        final_model_path = 'song_emotion_model_crnn_attention.keras'
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

    except Exception as e:
        print(f"Error during training: {e}")
