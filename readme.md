# Multi-Task Music Emotion & Genre Model

This project implements a **multi-task** deep learning model that:
1. **Classifies** music tracks into one of several top genres.
2. **Regresses** three continuous emotional dimensions: **Valence**, **Arousal**, and **Dominance**.

---

## **1. Data Preparation**

1. **Chunking (5s each)**  
   - Split each 30s preview into multiple 5s segments to increase sample diversity.

2. **Waveform Augmentation** (Training Only)  
   - Random **time-stretch**, **pitch shift**, and **additive noise** to improve robustness.

3. **Mel-Spectrogram Extraction**  
   - **128 Mel bins** → Convert to dB scale → **Normalize** each spectrogram.

4. **SpecAugment** (Training Only)  
   - Random **frequency** and **time** masking to prevent overfitting.

5. **Padding/Truncation**  
   - Fixed final shape `[128 x 300]`.

6. **Label Encoding**  
   - **Regression**: V/A/D values (scaled to `[0, 1]`).  
   - **Genre**: Top genres encoded via a label encoder.

---


## **2. Model Architecture**

The architecture is using **CRNN** and comprises:

**Input Layer**  
- Shape: `[128, 300, 1]` (Mel-spectrogram with 1 channel)

**Conv Block 1**  
1. **Conv2D(64, (3,3), padding='same')**  
2. **BatchNormalization**  
3. **ReLU Activation**  
4. **MaxPooling2D(pool_size=(2,2))**  
5. **Dropout(0.3)**

**Conv Block 2**  
1. **Conv2D(128, (3,3), padding='same')**  
2. **BatchNormalization**  
3. **ReLU Activation**  
4. **MaxPooling2D(pool_size=(2,2))**  
5. **Dropout(0.3)**

**Conv Block 3**  
1. **Conv2D(128, (3,3), padding='same')**  
2. **BatchNormalization**  
3. **ReLU Activation**  
4. **MaxPooling2D(pool_size=(2,2))**  
5. **Dropout(0.3)**

**Reshaping for LSTM**  
1. **Permute((2, 1, 3))** – rearranges `[freq, time, channels]` to `[time, freq, channels]`  
2. **Reshape((time_steps, features))** – flattens `freq * channels` into a single dimension

**BiLSTM**  
- **Bidirectional(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))**

**Fully-Connected Layer**  
1. **Dense(256, activation='relu')**  
2. **Dropout(0.4)**

**Multi-Task Outputs**  
1. **Regression Head**: `Dense(3, activation='linear', name='reg_output')`  
   - Predicts **Valence, Arousal, Dominance**  
2. **Classification Head**: `Dense(num_genres, activation='softmax', name='class_output')`  
   - Predicts **Genre** probabilities

---

## **3. Technologies Used**

- **TensorFlow/Keras** for the core deep learning model.
- **Librosa** for audio loading and Mel-spectrogram extraction.
- **scikit-learn** for label encoding and scaling (MinMaxScaler).
- **pandas, numpy** for data handling and array ops.
- **matplotlib, seaborn** for plots (e.g., confusion matrix, scatter plots).
- **tqdm** for progress bars during feature extraction.

---

## **4. Training**

- **Loss Functions**  
  - **Mean Squared Error (MSE)** for the regression head.  
  - **Sparse Categorical Crossentropy** for the classification head.
- **Optimizer**: **Adam** with an **Exponential Decay** learning rate schedule.
- **Callbacks**:  
  - **EarlyStopping** (to stop training if val loss stops improving).  
  - **ModelCheckpoint** (to save the best weights).  
  - **ReduceLROnPlateau** (to lower learning rate when training plateaus).

---

## **5. Results**

1. **Regression**  
   - Evaluated with **Mean Absolute Error** (MAE) on valence, arousal, and dominance.
2. **Classification**  
   - **Genre** accuracy and detailed classification report (precision, recall, F1).
3. **Top-K Predictions**  
   - We can retrieve the top 3 most likely genres per track from the softmax output.

---

## **6. Summary**

This model combines **CNN**-based feature extraction with a **BiLSTM** for temporal context, ending in two separate heads for **emotion regression** and **genre classification**. By chunking audio and applying augmentations, the model learns robust representations that help it excel at both continuous (V/A/D) and categorical (genre) tasks.
