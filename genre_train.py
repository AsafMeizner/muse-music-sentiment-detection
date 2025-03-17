import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.image import resize # type: ignore

# =============================================================================
# Create a directory to save model and plots
# =============================================================================
RESULTS_DIR = "./genre_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# =============================================================================
# Configuration
# =============================================================================
# Set the dataset directory.
# Assumes your dataset is arranged as:
# ./GTZAN-Dataset/genres_original/<genre_name>/<audio_file>.wav
data_dir = "./GTZAN-Dataset/genres_original"
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
target_shape = (150, 150)  # Resize mel-spectrogram to this shape

# =============================================================================
# Data Loading and Preprocessing Function
# =============================================================================
def load_and_preprocess_data(data_dir, classes, target_shape=(150, 150)):
    data = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load the audio file
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    # Set up chunk parameters: 4-second chunks with 2-second overlap
                    chunk_duration = 4
                    overlap_duration = 2
                    chunk_samples = chunk_duration * sample_rate
                    overlap_samples = overlap_duration * sample_rate
                    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

                    for i in range(num_chunks):
                        start = i * (chunk_samples - overlap_samples)
                        end = start + chunk_samples
                        chunk = audio_data[start:end]
                        # Compute mel spectrogram
                        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                        # Expand dims to add a channel dimension
                        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
                        # Resize spectrogram to target shape using tf.image.resize
                        mel_spectrogram_resized = resize(mel_spectrogram, target_shape).numpy()
                        data.append(mel_spectrogram_resized)
                        labels.append(i_class)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return np.array(data), np.array(labels)

# =============================================================================
# Load Data
# =============================================================================
data, labels = load_and_preprocess_data(data_dir, classes, target_shape)
print("data.shape:", data.shape)
print("labels.shape:", labels.shape)

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(classes))

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

# =============================================================================
# Build the CNN Model
# =============================================================================
model = tf.keras.models.Sequential()

# Input shape inferred from X_train[0]: e.g. (150, 150, 1)
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=X_train[0].shape))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units=1200, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(units=len(classes), activation='softmax'))

model.summary()

# =============================================================================
# Compile the Model
# =============================================================================
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =============================================================================
# Set up Callbacks for Checkpointing and Early Stopping
# =============================================================================
checkpoint = ModelCheckpoint(os.path.join(RESULTS_DIR, "best_model.keras"),
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# =============================================================================
# Train the Model
# =============================================================================
training_history = model.fit(X_train, Y_train,
                             epochs=30,
                             batch_size=32,
                             validation_data=(X_test, Y_test),
                             callbacks=[checkpoint, earlystop])

# Save the final model in .keras format
model.save(os.path.join(RESULTS_DIR, "final_model.keras"))

# =============================================================================
# Plot and Save Training Curves
# =============================================================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(training_history.history['loss'], label='Train Loss')
plt.plot(training_history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_history.history['accuracy'], label='Train Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "training_curves.png")
plt.savefig(plot_path)
plt.show()

print("Model and training plots have been saved to", RESULTS_DIR)
