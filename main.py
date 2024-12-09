import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torchaudio
from functools import partial

# Toggle between CPU and GPU for feature extraction
USE_GPU = False
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for TensorFlow

# GPU-based audio feature extraction
def extract_audio_features_torch(file_path, max_pad_length=300, augment=False):
    try:
        waveform, sample_rate = torchaudio.load(file_path)

        # Data augmentation
        if augment:
            if np.random.rand() > 0.5:
                waveform = torchaudio.transforms.TimeStretch()(waveform)
            if np.random.rand() > 0.5:
                waveform = torchaudio.transforms.PitchShift(sample_rate, n_steps=np.random.randint(-2, 2))(waveform)

        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=128, f_max=8000
        )
        mel_spec = mel_spec_transform(waveform)
        mel_spec = mel_spec.squeeze(0).numpy()

        # Normalize and pad/truncate
        mean = np.mean(mel_spec)
        std = np.std(mel_spec)
        mel_spec = (mel_spec - mean) / (std + 1e-9)
        pad_width = max(0, max_pad_length - mel_spec.shape[1])
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_length]
        return mel_spec
    except Exception as e:
        print(f"Error processing {file_path} (GPU): {e}")
        return np.zeros((128, max_pad_length))

# CPU-based audio feature extraction
def extract_audio_features_librosa(file_path, max_pad_length=300, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)

        # Data augmentation
        if augment:
            if np.random.rand() > 0.5:
                y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
            if np.random.rand() > 0.5:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 2))
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.005, y.shape)
                y = y + noise

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize and pad/truncate
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-9)
        pad_width = max(0, max_pad_length - mel_spec_db.shape[1])
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_length]
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path} (CPU): {e}")
        return np.zeros((128, max_pad_length))

# Main feature extraction function
def extract_audio_features(file_path, max_pad_length=300, augment=False):
    if USE_GPU:
        return extract_audio_features_torch(file_path, max_pad_length, augment)
    else:
        return extract_audio_features_librosa(file_path, max_pad_length, augment)

# Moved to top-level to avoid pickling issues
def process_audio_file(file_path, max_pad_length=300, augment=False):
    return extract_audio_features(file_path, max_pad_length=max_pad_length, augment=augment)

# Parallel processing for feature extraction
def parallel_extract_audio_features(audio_paths, max_pad_length=300, augment=False):
    # Use functools.partial to fix arguments
    process_func = partial(process_audio_file, max_pad_length=max_pad_length, augment=augment)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_func, audio_paths), total=len(audio_paths)))
    return np.array(results)

# Prepare features and labels
def prepare_data(df, save_path='prepared_data_emotion.npz', augment=False):
    if os.path.exists(save_path):
        print("Loading prepared data...")
        data = np.load(save_path, allow_pickle=True)
        return data['features'], data['labels']

    print("Extracting audio features...")
    # Corrected the path joining
    audio_paths = [os.path.join(preview) for preview in df['audio_previews']]
    audio_features = parallel_extract_audio_features(audio_paths, augment=augment)

    # Scale valence, arousal, and dominance to [0, 1]
    scaler = MinMaxScaler()
    emotion_labels = scaler.fit_transform(df[['valence_tags', 'arousal_tags', 'dominance_tags']])

    np.savez_compressed(save_path, features=audio_features, labels=emotion_labels)
    return audio_features, emotion_labels

# Build an improved CNN model for multi-output regression
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN layers with batch normalization
    x = Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Multi-output regression
    outputs = Dense(3, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='huber', metrics=['mae'])

    model.summary()
    return model

# Create a TensorFlow Dataset
def create_tf_dataset(features, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Main script
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'
    audio_dir = 'audio_files'  # Provide the correct path to your audio previews directory
    save_path = 'prepared_data_emotion.npz'  # Path to save or load prepared data

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)
        print("Preparing data...")
        features, labels = prepare_data(df, save_path=save_path, augment=True)
        features = features[..., np.newaxis]  # Add channel dimension

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        print("Creating TensorFlow datasets...")
        train_dataset = create_tf_dataset(X_train, y_train)
        test_dataset = create_tf_dataset(X_test, y_test, shuffle=False)

        print("Building model...")
        model = build_model(X_train.shape[1:])

        # Ensure the directory for model checkpoints exists
        checkpoint_dir = 'model_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "emotion_epoch-{epoch:02d}-val_loss-{val_loss:.2f}.keras")

        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        print("Training model...")
        model.fit(
            train_dataset,
            epochs=50,
            validation_data=test_dataset,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )

        loss, mae = model.evaluate(test_dataset)
        print(f"Test Loss: {loss}, Test MAE: {mae}")

        # Save the trained model in .keras format
        model.save('song_emotion_model.keras')
        print("Model saved as song_emotion_model.keras")

        # Debugging predictions
        for batch_features, batch_labels in test_dataset.take(1):
            predictions = model.predict(batch_features)
            print("Predictions:", predictions[:5])  # Show first 5 predictions
            print("Actual Labels:", batch_labels.numpy()[:5])  # Show first 5 actual labels

    except Exception as e:
        print(f"Error in main execution: {e}")