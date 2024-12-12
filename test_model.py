# test_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore

#############################################
# Custom Attention Layer (Same as in training)
#############################################
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(self.units, activation='tanh')
        self.output_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch, time, features)
        scores = self.output_dense(self.score_dense(inputs))  # (batch, time, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)     # (batch, time, 1)
        weighted_sum = inputs * attention_weights
        context_vector = tf.reduce_sum(weighted_sum, axis=1)  # (batch, features)
        return context_vector

#############################################
# TensorFlow Dataset Creation
#############################################
def create_tf_dataset(features, labels=None, batch_size=32, shuffle=False):
    """
    Creates a TensorFlow dataset from features and labels.

    Parameters:
        features (np.ndarray): Audio features.
        labels (np.ndarray, optional): Emotion labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Prepared dataset.
    """
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#############################################
# Visualization Functions
#############################################
def plot_predicted_vs_actual(y_true, y_pred, scaler, emotion_names=['Valence', 'Arousal', 'Dominance']):
    """
    Plots scatter plots of Actual vs. Predicted values for each emotion dimension.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        scaler (MinMaxScaler): Scaler used during training.
        emotion_names (list): List of emotion dimension names.
    """
    # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    for i, emotion in enumerate(emotion_names):
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_true_inv[:, i], y=y_pred_inv[:, i], alpha=0.5)
        plt.plot([y_true_inv[:, i].min(), y_true_inv[:, i].max()],
                 [y_true_inv[:, i].min(), y_true_inv[:, i].max()],
                 'r--')
        plt.xlabel(f'Actual {emotion}')
        plt.ylabel(f'Predicted {emotion}')
        plt.title(f'Actual vs Predicted {emotion}')
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f'actual_vs_predicted_{emotion.lower()}.png'
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

def plot_error_distribution(y_true, y_pred, scaler, emotion_names=['Valence', 'Arousal', 'Dominance']):
    """
    Plots histograms of prediction errors for each emotion dimension.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        scaler (MinMaxScaler): Scaler used during training.
        emotion_names (list): List of emotion dimension names.
    """
    # Inverse transform to original scale
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    errors = y_pred_inv - y_true_inv

    for i, emotion in enumerate(emotion_names):
        plt.figure(figsize=(8,4))
        sns.histplot(errors[:, i], kde=True, bins=30, color='skyblue')
        plt.xlabel(f'Prediction Error for {emotion}')
        plt.title(f'Error Distribution for {emotion}')
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f'error_distribution_{emotion.lower()}.png'
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    # Paths and constants
    test_data_path = 'test_data.npz'
    model_path = 'model_checkpoints/crnn_attention_best.keras'  # Use the best model checkpoint
    scaler_path = 'test_data_scaler.pkl'  # Adjust if different

    try:
        # Check that data, model, and scaler exist
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found at {test_data_path}.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")

        # Load test data
        print(f"Loading test data from {test_data_path}...")
        data = np.load(test_data_path, allow_pickle=True)
        X_test, y_test = data['features'], data['labels']
        print(f"Test data loaded: {X_test.shape[0]} samples.")

        # Add channel dimension
        X_test = X_test[..., np.newaxis]

        # Create test dataset
        print("Creating TensorFlow test dataset...")
        test_dataset = create_tf_dataset(X_test, y_test, batch_size=32, shuffle=False)

        # Load the trained model with custom_objects handled by registration
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)

        # Re-compile the model to specify loss and metrics
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model loaded and compiled.")

        # Load the scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded.")

        # Make predictions
        print("Making predictions on the test set...")
        y_pred = model.predict(test_dataset)
        y_pred = y_pred.reshape(-1, 3)  # Ensure shape is (num_samples, 3)
        print("Predictions completed.")

        # Verify shapes
        if y_test.shape[0] != y_pred.shape[0]:
            raise ValueError("Mismatch between number of predictions and number of true labels.")
        print("Shapes verified.")

        # Plot Predicted vs Actual
        print("Generating Predicted vs Actual plots...")
        plot_predicted_vs_actual(y_test, y_pred, scaler)

        # Plot Error Distribution
        print("Generating Error Distribution plots...")
        plot_error_distribution(y_test, y_pred, scaler)

        print("All visualizations saved successfully.")

    except Exception as e:
        print(f"Error during testing: {e}")
