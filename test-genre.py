import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.image import resize

# ================================
# Configuration
# ================================
# Path to the GTZAN dataset (local folder)
data_dir = "./GTZAN-Dataset/genres_original"
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
target_shape = (150, 150)  # same as used during training
RESULTS_DIR = "./genre_results"  # folder where plots will be saved

# Path to the saved model (change as needed)
model_path = os.path.join(RESULTS_DIR, "best_model.keras")

# ================================
# Data Loading and Preprocessing Function
# ================================
def load_and_preprocess_data(data_dir, classes, target_shape=(150,150)):
    """
    Loads the GTZAN dataset from data_dir.
    For each genre folder, it splits each audio file into 4-second chunks (with 2-second overlap),
    computes the mel-spectrogram, resizes it to target_shape, and returns the image data and labels.
    """
    data = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load the audio file (use native sampling rate)
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    # Parameters for chunking: 4-second chunks with 2-second overlap
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
                        # Expand dims for channel dimension
                        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
                        # Resize spectrogram to target shape using tf.image.resize
                        mel_resized = resize(mel_spectrogram, target_shape).numpy()
                        data.append(mel_resized)
                        labels.append(i_class)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return np.array(data), np.array(labels)

# ================================
# Load Data and Prepare Test Set
# ================================
print("Loading and preprocessing data...")
data, labels = load_and_preprocess_data(data_dir, classes, target_shape)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# One-hot encode labels
labels_cat = to_categorical(labels, num_classes=len(classes))

# Split the dataset (use 20% as test set)
_, X_test, _, Y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)
print("Test set shape:", X_test.shape)

# ================================
# Load the Saved Model
# ================================
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)
model.summary()

# ================================
# Evaluate the Model on Test Data
# ================================
print("Evaluating model on test set...")
score = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss: {score[0]:.4f}, Test Accuracy: {score[1]*100:.2f}%")

# ================================
# Generate Predictions and Confusion Matrix
# ================================
print("Generating predictions...")
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(Y_test, axis=1)

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)
report = classification_report(true_labels, pred_labels, target_names=classes)
print("Classification Report:\n", report)

# ================================
# Plot Confusion Matrix
# ================================
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalize the confusion matrix.
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_path:
        plt.savefig(save_path)
    plt.show()

cm_save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plot_confusion_matrix(cm, classes, save_path=cm_save_path)

# ================================
# Optionally: Save the Classification Report to a Text File
# ================================
report_save_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_save_path, "w") as f:
    f.write(report)
print("Confusion matrix and classification report saved to", RESULTS_DIR)
