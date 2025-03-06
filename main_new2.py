import os
import warnings
import numpy as np
import pandas as pd
import ast
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import concurrent.futures

# Suppress known warnings
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set.")

# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
SR = 22050                    # Sampling rate (Hz)
SEGMENT_DURATION = 10         # Each segment is 10 seconds
HOP_LENGTH = 512              # Hop length for STFT
N_MELS = 128                  # Number of mel bands
N_MFCC = 20                   # Number of MFCC coefficients (we use 20; 20*3 = 60)
N_CHROMA = 12                 # Chroma features (12*3 = 36)
N_TONNETZ = 6                 # Tonnetz features (6*3 = 18)
AGG_FEATURE_DIM = 498         # Total aggregated feature vector dimension

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# =============================================================================
# 1. Load CSV, Filter, and Expand Data
# =============================================================================
df = pd.read_csv("filtered_dataset.csv")
df = df.drop_duplicates()

# Exclude "indie" and keep only top 10 genres
df = df[df['genre'].str.lower() != "indie"]
genre_counts = df['genre'].value_counts()
top_genres = genre_counts.nlargest(10).index.tolist()
df = df[df['genre'].isin(top_genres)]

# Process seeds: convert string to list and keep top 20 seeds.
df['seeds_list'] = df['seeds'].apply(lambda x: ast.literal_eval(x))
all_seeds = [seed for seeds in df['seeds_list'] for seed in seeds]
seed_counts = Counter(all_seeds)
top20_seeds = [seed for seed, _ in seed_counts.most_common(20)]
df['seeds_list'] = df['seeds_list'].apply(lambda seeds: [s for s in seeds if s in top20_seeds])
df = df[df['seeds_list'].map(len) > 0]

# Encode genre labels.
genre_encoder = LabelEncoder()
df['genre_encoded'] = genre_encoder.fit_transform(df['genre'])
num_genres = len(genre_encoder.classes_)

# Encode seeds.
seed_mlb = MultiLabelBinarizer(classes=sorted(top20_seeds))
seeds_encoded = seed_mlb.fit_transform(df['seeds_list'])
num_seed_labels = seeds_encoded.shape[1]
df['seeds_encoded'] = list(seeds_encoded)

# Continuous targets.
df['valence'] = df['valence_tags'].astype(float)
df['arousal'] = df['arousal_tags'].astype(float)
df['dominance'] = df['dominance_tags'].astype(float)

# Expand each song into non-overlapping 10-second segments.
def expand_dataframe(df):
    rows = []
    for idx, row in df.iterrows():
        file_path = row['audio_previews']
        try:
            duration = librosa.get_duration(path=file_path, sr=SR)
        except Exception as e:
            print(f"Error getting duration for {file_path}: {e}")
            duration = SEGMENT_DURATION
        num_segments = int(duration // SEGMENT_DURATION) if duration >= SEGMENT_DURATION else 1
        for i in range(num_segments):
            offset = i * SEGMENT_DURATION
            new_row = {
                'audio_previews': file_path,
                'offset': offset,
                'genre_encoded': row['genre_encoded'],
                'seeds_encoded': row['seeds_encoded'],
                'valence': row['valence'],
                'arousal': row['arousal'],
                'dominance': row['dominance']
            }
            rows.append(new_row)
    return pd.DataFrame(rows)

expanded_df = expand_dataframe(df)
print("Expanded dataset size:", len(expanded_df))

train_df, val_df = train_test_split(expanded_df, test_size=0.2, random_state=42)

train_paths = train_df['audio_previews'].values
train_offsets = train_df['offset'].values.astype(np.float32)
train_genres = train_df['genre_encoded'].values
train_seeds = np.stack(train_df['seeds_encoded'].values)
train_valence = train_df['valence'].values
train_arousal = train_df['arousal'].values
train_dominance = train_df['dominance'].values

val_paths = val_df['audio_previews'].values
val_offsets = val_df['offset'].values.astype(np.float32)
val_genres = val_df['genre_encoded'].values
val_seeds = np.stack(val_df['seeds_encoded'].values)
val_valence = val_df['valence'].values
val_arousal = val_df['arousal'].values
val_dominance = val_df['dominance'].values

# =============================================================================
# 2. Aggregated Feature Extraction Function (498-D vector)
# =============================================================================
def get_aggregated_features(file_path, offset=0, duration=SEGMENT_DURATION, sr=SR):
    try:
        y, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration)
    except Exception as e:
        print(f"Error loading {file_path} at offset {offset}: {e}")
        return None
    if len(y) < duration * sr:
        pad_length = int(duration * sr) - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')
    
    # MFCC features (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = np.concatenate([mfcc_mean, mfcc_min, mfcc_max])
    
    # Mel spectrogram features (128 mel bands)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    melspec_mean = melspec.mean(axis=1)
    melspec_min = melspec.min(axis=1)
    melspec_max = melspec.max(axis=1)
    melspec_feature = np.concatenate([melspec_mean, melspec_min, melspec_max])
    
    # Chroma features (12 bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_feature = np.concatenate([chroma_mean, chroma_min, chroma_max])
    
    # Tonnetz features (6 dims) on harmonic component.
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1)
    tonnetz_min = tonnetz.min(axis=1)
    tonnetz_max = tonnetz.max(axis=1)
    tonnetz_feature = np.concatenate([tonnetz_mean, tonnetz_min, tonnetz_max])
    
    # Concatenate in order: chroma, melspec, mfcc, tonnetz → total dimension 36+384+60+18 = 498.
    feature = np.concatenate([chroma_feature, melspec_feature, mfcc_feature, tonnetz_feature])
    return feature.astype(np.float32)

# =============================================================================
# 3. tf.data Pipeline Function
# =============================================================================
def load_and_preprocess_agg(audio_path, offset, genre, seeds, valence, arousal, dominance):
    def _extract(path, off):
        path = path.numpy()
        off = off.numpy().item()
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        feat = get_aggregated_features(path, offset=off)
        return feat
    feature = tf.py_function(func=_extract, inp=[audio_path, offset], Tout=tf.float32)
    feature.set_shape([AGG_FEATURE_DIM])
    labels = {
        'genre': genre,
        'seeds': seeds,
        'sentiments': tf.stack([valence, arousal, dominance], axis=-1)
    }
    return feature, labels

def get_dataset(paths, offsets, genres, seeds, valence, arousal, dominance):
    ds = tf.data.Dataset.from_tensor_slices((paths, offsets, genres, seeds, valence, arousal, dominance))
    ds = ds.map(load_and_preprocess_agg, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = get_dataset(train_paths, train_offsets, train_genres, train_seeds, train_valence, train_arousal, train_dominance)
val_dataset = get_dataset(val_paths, val_offsets, val_genres, val_seeds, val_valence, val_arousal, val_dominance)

# =============================================================================
# 4. Build Separate Models
# =============================================================================
def build_genre_model(input_dim, num_genres):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(300, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(200, activation='relu')(x)
    outputs = layers.Dense(num_genres, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_seeds_model(input_dim, num_seed_labels):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(300, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(200, activation='relu')(x)
    outputs = layers.Dense(num_seed_labels, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_sentiments_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='tanh')(x)  # tanh to constrain in [-1, 1]
    return models.Model(inputs=inputs, outputs=outputs)

# Build models.
genre_model = build_genre_model(AGG_FEATURE_DIM, num_genres)
seeds_model = build_seeds_model(AGG_FEATURE_DIM, num_seed_labels)
sentiments_model = build_sentiments_model(AGG_FEATURE_DIM)

genre_model.summary()
seeds_model.summary()
sentiments_model.summary()

# =============================================================================
# 5. Training Functions for Parallel Execution
# =============================================================================
def train_genre_model():
    # Rebuild dataset inside the process.
    ds_train = get_dataset(train_paths, train_offsets, train_genres, train_seeds, train_valence, train_arousal, train_dominance).map(lambda feat, lab: (feat, lab['genre']))
    ds_val = get_dataset(val_paths, val_offsets, val_genres, val_seeds, val_valence, val_arousal, val_dominance).map(lambda feat, lab: (feat, lab['genre']))
    model = build_genre_model(AGG_FEATURE_DIM, num_genres)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint("best_genre_model.keras", monitor='val_accuracy', mode='max', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=NUM_EPOCHS, callbacks=cb)
    model.save("best_genre_model.keras")
    return history.history

def train_seeds_model():
    ds_train = get_dataset(train_paths, train_offsets, train_genres, train_seeds, train_valence, train_arousal, train_dominance).map(lambda feat, lab: (feat, lab['seeds']))
    ds_val = get_dataset(val_paths, val_offsets, val_genres, val_seeds, val_valence, val_arousal, val_dominance).map(lambda feat, lab: (feat, lab['seeds']))
    model = build_seeds_model(AGG_FEATURE_DIM, num_seed_labels)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint("best_seeds_model.keras", monitor='val_accuracy', mode='max', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=NUM_EPOCHS, callbacks=cb)
    model.save("best_seeds_model.keras")
    return history.history

def train_sentiments_model():
    ds_train = get_dataset(train_paths, train_offsets, train_genres, train_seeds, train_valence, train_arousal, train_dominance).map(lambda feat, lab: (feat, lab['sentiments']))
    ds_val = get_dataset(val_paths, val_offsets, val_genres, val_seeds, val_valence, val_arousal, val_dominance).map(lambda feat, lab: (feat, lab['sentiments']))
    model = build_sentiments_model(AGG_FEATURE_DIM)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mae'])
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint("best_sentiments_model.keras", monitor='val_loss', mode='min', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=NUM_EPOCHS, callbacks=cb)
    model.save("best_sentiments_model.keras")
    return history.history

# =============================================================================
# 6. Parallel Training Using ProcessPoolExecutor
# =============================================================================
if __name__ == "__main__":
    # Ensure TensorFlow does not allocate all GPU memory in each process.
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_genre = executor.submit(train_genre_model)
        future_seeds = executor.submit(train_seeds_model)
        future_sentiments = executor.submit(train_sentiments_model)
        
        genre_history = future_genre.result()
        seeds_history = future_seeds.result()
        sentiments_history = future_sentiments.result()
    
    # Save training curves.
    def plot_and_save_history(history, title_prefix, filename_prefix):
        hist = history
        epochs_range = range(len(hist['loss']))
        
        plt.figure(figsize=(16,10))
        plt.subplot(1,2,1)
        plt.plot(epochs_range, hist['loss'], label='Train Loss')
        plt.plot(epochs_range, hist['val_loss'], label='Val Loss')
        plt.title(f'{title_prefix} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1,2,2)
        if 'accuracy' in hist:
            plt.plot(epochs_range, hist['accuracy'], label='Train Acc')
            plt.plot(epochs_range, hist['val_accuracy'], label='Val Acc')
            plt.title(f'{title_prefix} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        elif 'mae' in hist:
            plt.plot(epochs_range, hist['mae'], label='Train MAE')
            plt.plot(epochs_range, hist['val_mae'], label='Val MAE')
            plt.title(f'{title_prefix} MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_training.png")
        plt.show()
    
    plot_and_save_history(genre_history, "Genre Model", "genre_model")
    plot_and_save_history(seeds_history, "Seeds Model", "seeds_model")
    plot_and_save_history(sentiments_history, "Sentiments Model", "sentiments_model")
    
    # =============================================================================
    # 7. Evaluation and Graphs on Validation Set for Genre and Sentiments
    # =============================================================================
    def evaluate_genre_model(model_path, dataset):
        model = tf.keras.models.load_model(model_path)
        y_true = []
        y_pred = []
        for feat, lab in dataset:
            preds = model.predict(feat)
            y_true.extend(lab.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, target_names=genre_encoder.classes_)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Genre Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(genre_encoder.classes_))
        plt.xticks(tick_marks, genre_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, genre_encoder.classes_)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("genre_confusion_matrix.png")
        plt.show()
        print("Genre Classification Report:\n", cr)
    
    genre_val_ds = val_dataset.map(lambda feat, lab: (feat, lab['genre']))
    evaluate_genre_model("best_genre_model.keras", genre_val_ds)
    
    def evaluate_sentiments_model(model_path, dataset):
        model = tf.keras.models.load_model(model_path)
        y_true = []
        y_pred = []
        for feat, lab in dataset:
            preds = model.predict(feat)
            y_true.extend(lab.numpy())
            y_pred.extend(preds)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sentiments = ['Valence', 'Arousal', 'Dominance']
        plt.figure(figsize=(18,5))
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                     [y_true[:, i].min(), y_true[:, i].max()], 'r--')
            plt.xlabel("True " + sentiments[i])
            plt.ylabel("Predicted " + sentiments[i])
            plt.title(sentiments[i] + " Regression")
        plt.tight_layout()
        plt.savefig("sentiments_regression.png")
        plt.show()
    
    sentiments_val_ds = val_dataset.map(lambda feat, lab: (feat, lab['sentiments']))
    evaluate_sentiments_model("best_sentiments_model.keras", sentiments_val_ds)
    
    # =============================================================================
    # 8. Inference Function
    # =============================================================================
    def predict_song(file_path, offset=0):
        feat = get_aggregated_features(file_path, offset=offset)
        if feat is None:
            return None
        feat = np.expand_dims(feat, axis=0)
        genre_pred = genre_model.predict(feat)
        seeds_pred = seeds_model.predict(feat)
        sentiments_pred = sentiments_model.predict(feat)
        
        genre_idx = np.argmax(genre_pred, axis=1)[0]
        genre_label = genre_encoder.inverse_transform([genre_idx])[0]
        seeds_binary = (seeds_pred[0] > 0.5).astype(int)
        seeds_labels = [s for s, flag in zip(seed_mlb.classes_, seeds_binary) if flag == 1]
        sentiments = sentiments_pred[0]
        return {
            'genre': genre_label,
            'seeds': seeds_labels,
            'valence': sentiments[0],
            'arousal': sentiments[1],
            'dominance': sentiments[2]
        }
    
    # =============================================================================
    # Example Usage:
    # =============================================================================
    # Uncomment the lines below and set a valid file path to test inference.
    # result = predict_song("audio_previews/6tqFC1DIOphJkCwrjVzPmg.mp3", offset=0)
    # print(result)
