import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and preprocess the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['seeds'] = df['seeds'].apply(lambda x: eval(x))  # Convert string to list
    df['emotion_tags'] = df['seeds'].apply(lambda tags: ','.join(tags))  # Join tags for encoding
    return df

# Extract audio features using librosa
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(40)

# Prepare features and labels
def prepare_data(df, audio_dir, save_path='prepared_data.npz'):
    if os.path.exists(save_path):
        print("Loading prepared data...")
        data = np.load(save_path, allow_pickle=True)
        return data['features'], data['labels'], data['label_classes']

    print("Extracting audio features...")
    audio_features = []
    for preview in df['audio_previews']:
        file_path = os.path.join(audio_dir, preview)
        audio_features.append(extract_audio_features(file_path))
    audio_features = np.array(audio_features)

    metadata_features = df[['valence_tags', 'arousal_tags', 'dominance_tags']].values
    all_features = np.hstack((audio_features, metadata_features))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['emotion_tags'])
    labels = to_categorical(labels)

    np.savez_compressed(save_path, features=all_features, labels=labels, label_classes=label_encoder.classes_)
    return all_features, labels, label_encoder.classes_

# Build the RNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main script
if __name__ == "__main__":
    dataset_path = 'filtered_dataset.csv'
    audio_dir = ''

    # Load and preprocess dataset
    print("Loading dataset...")
    df = load_dataset(dataset_path)

    # Prepare features and labels
    print("Preparing data...")
    features, labels, label_classes = prepare_data(df, audio_dir)

    # Reshape features for RNN
    time_steps = 1  # Use 1 time step for static features
    features = features.reshape(features.shape[0], time_steps, features.shape[1])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build the model
    print("Building model...")
    model = build_model(X_train.shape[1:], y_train.shape[1])

    # Add checkpoints
    checkpoint_path = "model_checkpoints/song_sentiment_epoch-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Save the final model
    model.save('song_sentiment_rnn_model.h5')
    print("Model saved as song_sentiment_rnn_model.h5")
