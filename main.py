import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(file_path):
    """Extract audio features for TensorFlow model input."""
    y, sr = librosa.load(file_path, duration=30)  # Load first 30 seconds for consistency

    # Extract audio features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Aggregate features (mean and std for each)
    features = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
        tempo
    ])
    return features

# -------------------------------
# Dataset Preparation
# -------------------------------
def prepare_dataset(csv_path, audio_folder):
    """Prepare features and labels for model training."""
    data = pd.read_csv(csv_path)  # CSV with columns: 'file', 'label'
    X, y = [], []
    for _, row in data.iterrows():
        file_path = os.path.join(audio_folder, row['file'])
        features = extract_features(file_path)
        X.append(features)
        y.append(row['label'])
    return np.array(X), np.array(y)

# -------------------------------
# Build the TensorFlow Model
# -------------------------------
def build_model(input_shape, num_classes):
    """Build a simple TensorFlow model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # For multi-class classification
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------
# Train the Model
# -------------------------------
def train_model(csv_path, audio_folder, output_model_path):
    """Train the model and save it."""
    # Load dataset
    X, y = prepare_dataset(csv_path, audio_folder)

    # Build and train the model
    model = build_model(X.shape[1], num_classes=len(np.unique(y)))
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

# -------------------------------
# Predict Sentiment
# -------------------------------
def predict_sentiment(model_path, audio_path):
    """Predict the sentiment of a song."""
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Extract features from audio
    features = extract_features(audio_path).reshape(1, -1)

    # Predict sentiment
    prediction = model.predict(features)
    sentiment = np.argmax(prediction)
    print(f"Predicted sentiment: {sentiment}")
    return sentiment

# -------------------------------
# Live Microphone Input
# -------------------------------
def record_audio(duration=10, sr=22050):
    """Record audio from the microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten(), sr

# -------------------------------
# Main Program
# -------------------------------
if __name__ == "__main__":
    # Define paths
    csv_path = "dataset.csv"  # CSV file with 'file' and 'label' columns
    audio_folder = "audio_files"  # Folder containing audio files
    model_path = "sentiment_model.h5"  # Path to save/load the model

    # Choose mode: 'train', 'predict', or 'microphone'
    mode = input("Choose mode (train/predict/microphone): ").strip().lower()

    if mode == "train":
        # Train the model
        train_model(csv_path, audio_folder, model_path)
    elif mode == "predict":
        # Predict sentiment from a file
        audio_path = input("Enter path to audio file: ").strip()
        predict_sentiment(model_path, audio_path)
    elif mode == "microphone":
        # Record and predict sentiment from live audio
        audio, sr = record_audio()
        sf.write("temp_audio.wav", audio, sr)  # Save temporary audio file
        predict_sentiment(model_path, "temp_audio.wav")
    else:
        print("Invalid mode. Choose 'train', 'predict', or 'microphone'.")
