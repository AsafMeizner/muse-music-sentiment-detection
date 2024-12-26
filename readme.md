# Multi-Task Music Emotion & Genre Model

This project implements a **multi-task** deep learning model that:
1. **Classifies** music tracks into one of several top genres.
2. **Regresses** three continuous emotional dimensions: **Valence**, **Arousal**, and **Dominance**.

---

## **1. Data Explanation**

I use the **MUSE** dataset, which has approximately 4000 songs. Each entry in the dataset contains:

- **lastfm_url**: A reference URL on Last.fm.  
- **track**: The track title.  
- **artist**: The artist name.  
- **seeds**: A list of descriptive emotional tags, e.g., `['aggressive', 'fun']`.  
- **number_of_emotion_tags**: How many emotion tags were applied.  
- **valence_tags**, **arousal_tags**, **dominance_tags**: Numerical values indicating each dimension of emotion for the track.  
- **mbid**: MusicBrainz ID for the track.  
- **spotify_id**: Spotify track ID.  
- **genre**: The top-level genre classification for the track.  
- **audio_previews**: A local path or filename to the 30-second audio clip (`.mp3`) for that track.

Example rows:
| lastfm_url                                                                 | track      | artist | seeds                                 | number_of_emotion_tags | valence_tags      | arousal_tags     | dominance_tags    | mbid                                 | spotify_id                      | genre    | audio_previews                                |
|---------------------------------------------------------------------------|------------|--------|---------------------------------------|------------------------|-------------------|------------------|-------------------|--------------------------------------|----------------------------------|----------|-----------------------------------------------|
| [Bamboo Banga](https://www.last.fm/music/m.i.a./_/bamboo%2bbanga)          | Bamboo Banga | M.I.A. | `['aggressive', 'fun', 'sexy', 'energetic']` | 13                     | 6.555071428571428 | 5.537214285714287 | 5.691357142857143 | 99dd2c8c-e7c1-413e-8ea4-4497a00ffa18 | 6tqFC1DIOphJkCwrjVzPmg          | hip-hop  | [Audio Preview](audio_previews\6tqFC1DIOphJkCwrjVzPmg.mp3) |
| [Die MF Die](https://www.last.fm/music/dope/_/die%2bmf%2bdie)              | Die MF Die | Dope   | `['aggressive']`                      | 7                      | 3.771176470588235 | 5.348235294117648 | 5.441764705882353 | b9eb3484-5e0e-4690-ab5a-ca91937032a5 | 5bU4KX47KqtDKKaLM4QCzh          | metal    | [Audio Preview](audio_previews\5bU4KX47KqtDKKaLM4QCzh.mp3) |

---

## **2. Data Preparation**

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

## **3. Model Architecture**

The architecture is using a **CRNN** (Convolutional Recurrent Neural Network) and comprises:

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

## **4. Technologies Used**

- **TensorFlow/Keras** for the core deep learning model.
- **Librosa** for audio loading and Mel-spectrogram extraction.
- **scikit-learn** for label encoding and scaling (MinMaxScaler).
- **pandas, numpy** for data handling and array ops.
- **matplotlib, seaborn** for plots (e.g., confusion matrix, scatter plots).
- **tqdm** for progress bars during feature extraction.

---

## **5. Training**

- **Loss Functions**  
  - **Mean Squared Error (MSE)** for the regression head.  
  - **Sparse Categorical Crossentropy** for the classification head.

- **Optimizer**  
  - **Adam** with an **Exponential Decay** learning rate schedule.

- **Callbacks**  
  - **EarlyStopping** (to stop training if val loss stops improving).  
  - **ModelCheckpoint** (to save the best weights).  
  - **ReduceLROnPlateau** (to lower learning rate when training plateaus).

---

## **6. Results**

1. **Regression**  
   - Evaluated with **Mean Absolute Error** (MAE) on valence, arousal, and dominance.

2. **Classification**  
   - **Genre** accuracy and detailed classification report (precision, recall, F1).

3. **Top-K Predictions**  
   - We can retrieve the top 3 most likely genres per track from the softmax output.

---

## **7. Summary**

This model combines **CNN**-based feature extraction with a **BiLSTM** for temporal context, ending in two separate heads for **emotion regression** and **genre classification**. By chunking audio and applying augmentations, the model learns robust representations that help it excel at both continuous (V/A/D) and categorical (genre) tasks.