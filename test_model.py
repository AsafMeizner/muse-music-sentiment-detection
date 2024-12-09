import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the prepared data
save_path = 'prepared_data_emotion.npz'  # Ensure this is the correct path to the prepared data
data = np.load(save_path, allow_pickle=True)
features = data['features']
labels = data['labels']

# The model was trained with features having an extra channel dimension
features = features[..., np.newaxis]

# Split into train/test (use same split ratio and random_state as training script)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Load the trained model
model_path = 'song_emotion_model.keras'  # Ensure this is the correct path to the model
model = tf.keras.models.load_model(model_path)

# Evaluate on the test data
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict on test data
predictions = model.predict(X_test)

# predictions and y_test are scaled between [0, 1]
# The model directly outputs scaled values as we used MinMaxScaler during training.

# For plotting, let's just compare actual vs predicted for a subset of the test set
num_samples_to_plot = 50  # Adjust as needed
actual = y_test[:num_samples_to_plot]
predicted = predictions[:num_samples_to_plot]

# Create subplots for valence, arousal, and dominance
emotions = ['Valence', 'Arousal', 'Dominance']
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
fig.suptitle('Actual vs. Predicted Emotion Values', fontsize=16)

for i, ax in enumerate(axes):
    ax.plot(actual[:, i], label='Actual', marker='o')
    ax.plot(predicted[:, i], label='Predicted', marker='x')
    ax.set_title(f'{emotions[i]}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Scaled Value [0, 1]')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Optionally, we can also do a scatter plot to see how well predictions correlate with actual values
fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
fig2.suptitle('Actual vs. Predicted Scatter Plots', fontsize=16)
for i, ax in enumerate(axes2):
    ax.scatter(actual[:, i], predicted[:, i], alpha=0.7)
    ax.set_title(f'{emotions[i]}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    # Plot a line y=x for reference
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
