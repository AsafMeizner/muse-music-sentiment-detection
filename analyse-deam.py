import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# -------------------------------------------------------------------------
# 0) PREPARE OUTPUT DIRECTORY FOR PLOTS
#    (So we can save each figure as a PNG.)
# -------------------------------------------------------------------------
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1) SET YOUR DEAM DATASET FOLDER HERE
# -------------------------------------------------------------------------
base_path = './deam-dataset'  # <-- Modify if needed

# -------------------------------------------------------------------------
# 2) BUILD PATHS FOR ANNOTATIONS AND AUDIO
# -------------------------------------------------------------------------
static_annotations_path = os.path.join(
    base_path,
    'DEAM_Annotations',
    'annotations',
    'annotations averaged per song',
    'song_level',
    'static_annotations_averaged_songs_1_2000.csv'
)

arousal_annotations_path = os.path.join(
    base_path,
    'DEAM_Annotations',
    'annotations',
    'annotations averaged per song',
    'dynamic (per second annotations)',
    'arousal.csv'
)

valence_annotations_path = os.path.join(
    base_path,
    'DEAM_Annotations',
    'annotations',
    'annotations averaged per song',
    'dynamic (per second annotations)',
    'valence.csv'
)

audio_folder = os.path.join(base_path, 'DEAM_audio', 'MEMD_audio')

# -------------------------------------------------------------------------
# 3) LOAD THE DATA
# -------------------------------------------------------------------------
static_annotations = pd.read_csv(static_annotations_path)
# Some columns may have leading spaces; remove them:
static_annotations.columns = static_annotations.columns.str.strip()

dynamic_annotations_arousal = pd.read_csv(arousal_annotations_path)
dynamic_annotations_valence = pd.read_csv(valence_annotations_path)

# -------------------------------------------------------------------------
# 4) EXPLORE THE STATIC ANNOTATIONS
# -------------------------------------------------------------------------
print("=== Static Annotations: First 5 Rows ===")
print(static_annotations.head(), "\n")

print("=== Static Annotations: Info ===")
print(static_annotations.info(), "\n")

print("=== Static Annotations: Describe ===")
print(static_annotations.describe(), "\n")

print("=== Columns ===")
print(static_annotations.columns)

# -------------------------------------------------------------------------
# 5) VISUALIZE BASIC DISTRIBUTIONS USING MATPLOTLIB (Saving as PNG)
# -------------------------------------------------------------------------
# 5a) Distribution of Valence Mean
plt.figure(figsize=(6, 4))
plt.hist(static_annotations['valence_mean'], bins=20, alpha=0.7)
plt.title('Valence Mean Distribution')
plt.xlabel('Valence Mean')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'valence_mean_distribution.png'))
plt.show()

# 5b) Distribution of Arousal Mean
plt.figure(figsize=(6, 4))
plt.hist(static_annotations['arousal_mean'], bins=20, alpha=0.7)
plt.title('Arousal Mean Distribution')
plt.xlabel('Arousal Mean')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'arousal_mean_distribution.png'))
plt.show()

# 5c) Scatterplot: Arousal Mean vs Valence Mean
plt.figure(figsize=(6, 6))
plt.scatter(
    static_annotations['arousal_mean'],
    static_annotations['valence_mean'],
    alpha=0.5
)
plt.title('Arousal Mean vs Valence Mean')
plt.xlabel('Arousal Mean')
plt.ylabel('Valence Mean')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'arousal_vs_valence_scatter.png'))
plt.show()

# 5d) Valence Std Distribution
plt.figure(figsize=(6, 4))
plt.hist(static_annotations['valence_std'], bins=20, alpha=0.7)
plt.title('Valence Std Distribution')
plt.xlabel('Valence Std')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'valence_std_distribution.png'))
plt.show()

# 5e) Arousal Std Distribution
plt.figure(figsize=(6, 4))
plt.hist(static_annotations['arousal_std'], bins=20, alpha=0.7)
plt.title('Arousal Std Distribution')
plt.xlabel('Arousal Std')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'arousal_std_distribution.png'))
plt.show()

# -------------------------------------------------------------------------
# 6) SPLIT SONGS INTO LOW / MEDIUM / HIGH FOR AROUSAL AND VALENCE
# -------------------------------------------------------------------------
arousal_low_threshold = 2.0
arousal_high_threshold = 7.5

valence_low_threshold = 2.0
valence_high_threshold = 7.5

# Arousal categories
arousal_low = static_annotations[static_annotations['arousal_mean'] < arousal_low_threshold]
arousal_high = static_annotations[static_annotations['arousal_mean'] > arousal_high_threshold]
arousal_medium = static_annotations[
    (static_annotations['arousal_mean'] >= arousal_low_threshold) &
    (static_annotations['arousal_mean'] <= arousal_high_threshold)
]

# Valence categories
valence_low = static_annotations[static_annotations['valence_mean'] < valence_low_threshold]
valence_high = static_annotations[static_annotations['valence_mean'] > valence_high_threshold]
valence_medium = static_annotations[
    (static_annotations['valence_mean'] >= valence_low_threshold) &
    (static_annotations['valence_mean'] <= valence_high_threshold)
]

print("\n=== AR Items ===")
print(f"Arousal Low: {len(arousal_low)} | Medium: {len(arousal_medium)} | High: {len(arousal_high)}")

print("=== VAL Items ===")
print(f"Valence  Low: {len(valence_low)}  | Medium: {len(valence_medium)}  | High: {len(valence_high)}")

# -------------------------------------------------------------------------
# 7) HELPER FUNCTION: PLAY AUDIO + SHOW WAVEFORM & SPECTROGRAM
# -------------------------------------------------------------------------
def play_audio_with_visuals(file_path):
    """
    Loads an audio file, plots its waveform and spectrogram,
    and returns an inline audio player.
    """
    print(f"\nPlaying audio from: {file_path}")
    if not os.path.exists(file_path):
        print("Audio file does NOT exist. Check your path!")
        return

    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    
    # 7a) Waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    # Save + show
    waveform_png = os.path.join(plots_dir, f"waveform_{os.path.basename(file_path)}.png")
    plt.savefig(waveform_png)
    plt.show()

    # 7b) Spectrogram (log-scale)
    plt.figure(figsize=(10, 3))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, x_axis='time', y_axis='log'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Log Scale)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    # Save + show
    spectrogram_png = os.path.join(plots_dir, f"spectrogram_{os.path.basename(file_path)}.png")
    plt.savefig(spectrogram_png)
    plt.show()

    # 7c) Audio playback (if interactive environment allows)
    display(Audio(file_path))

# -------------------------------------------------------------------------
# 8) DEMO: PICK ONE SONG FROM EACH AROUSAL CATEGORY AND PLAY
# -------------------------------------------------------------------------
if not arousal_low.empty:
    ex = arousal_low.iloc[0]
    print("\n=== [AROUSAL LOW EXAMPLE] ===")
    print(ex)
    audio_path = os.path.join(audio_folder, f"{int(ex['song_id'])}.mp3")
    play_audio_with_visuals(audio_path)

if not arousal_medium.empty:
    ex = arousal_medium.iloc[0]
    print("\n=== [AROUSAL MEDIUM EXAMPLE] ===")
    print(ex)
    audio_path = os.path.join(audio_folder, f"{int(ex['song_id'])}.mp3")
    play_audio_with_visuals(audio_path)

if not arousal_high.empty:
    ex = arousal_high.iloc[0]
    print("\n=== [AROUSAL HIGH EXAMPLE] ===")
    print(ex)
    audio_path = os.path.join(audio_folder, f"{int(ex['song_id'])}.mp3")
    play_audio_with_visuals(audio_path)

# -------------------------------------------------------------------------
# 9) DEMO DYNAMIC ANNOTATIONS FOR A CHOSEN SONG (First Valence High)
# -------------------------------------------------------------------------
if not valence_high.empty:
    chosen_song_id = int(valence_high.iloc[0]['song_id'])
    print(f"\n=== DYNAMIC ANNOTATIONS for song_id={chosen_song_id} ===")

    # Filter dynamic annotations for that ID
    val_df = dynamic_annotations_valence[dynamic_annotations_valence['song_id'] == chosen_song_id]
    ar_df = dynamic_annotations_arousal[dynamic_annotations_arousal['song_id'] == chosen_song_id]

    time_cols = [c for c in val_df.columns if c.startswith('sample_')]
    if len(val_df) == 1 and len(ar_df) == 1:
        time_values = np.array([
            int(col.split('_')[1][:-2]) / 1000.0
            for col in time_cols
        ])

        val_values = val_df[time_cols].values[0]
        ar_values = ar_df[time_cols].values[0]

        # Dynamic Valence
        plt.figure(figsize=(10, 3))
        plt.plot(time_values, val_values, label='Valence')
        plt.title(f'Dynamic Valence (song_id={chosen_song_id})')
        plt.xlabel('Time (s)')
        plt.ylabel('Valence')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fname_val = f"dynamic_valence_{chosen_song_id}.png"
        plt.savefig(os.path.join(plots_dir, fname_val))
        plt.show()

        # Dynamic Arousal
        plt.figure(figsize=(10, 3))
        plt.plot(time_values, ar_values, label='Arousal', color='red')
        plt.title(f'Dynamic Arousal (song_id={chosen_song_id})')
        plt.xlabel('Time (s)')
        plt.ylabel('Arousal')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fname_ar = f"dynamic_arousal_{chosen_song_id}.png"
        plt.savefig(os.path.join(plots_dir, fname_ar))
        plt.show()

        # Optional: playback
        chosen_path = os.path.join(audio_folder, f"{chosen_song_id}.mp3")
        play_audio_with_visuals(chosen_path)
    else:
        print("Error: Found more than one row for that song_id, or none at all!")

# -------------------------------------------------------------------------
# 10) DIMENSION-BASED EMOTION CLASSIFICATION (Russell’s Circumplex style)
# -------------------------------------------------------------------------
# We'll map each of the 11 emotions to a (val_center, aro_center).
EMOTION_CENTERS = {
    "H": ("Erotic, desirous",        (6.5, 5.0)),
    "J": ("Joyful, cheerful",        (7.5, 6.0)),
    "A": ("Amusing",                 (7.0, 7.5)),
    "G": ("Energizing, pump-up",     (8.0, 8.0)),
    "I": ("Indignant, defiant",      (4.0, 6.5)),
    "B": ("Annoying",                (4.5, 5.5)),
    "L": ("Scary, fearful",          (3.5, 7.5)),
    "C": ("Anxious, tense",          (3.5, 6.5)),
    "M": ("Triumphant, heroic",      (7.8, 6.8)),
    "F": ("Dreamy",                  (7.0, 3.0)),
    "K": ("Sad, depressing",         (2.5, 2.5)),
}

def classify_emotion_by_centroid(valence, arousal, centers_dict):
    best_label = None
    best_dist = float('inf')
    for code, (_, (vc, ac)) in centers_dict.items():
        dist = np.sqrt((valence - vc)**2 + (arousal - ac)**2)
        if dist < best_dist:
            best_dist = dist
            best_label = code
    return best_label

def assign_emotion_label(row):
    v = row["valence_mean"]
    a = row["arousal_mean"]
    return classify_emotion_by_centroid(v, a, EMOTION_CENTERS)

# Apply classification
static_annotations["emotion_label"] = static_annotations.apply(assign_emotion_label, axis=1)

# -------------------------------------------------------------------------
# 11) PLOT VALENCE–AROUSAL SCATTER WITH EMOTION LABELS
#     (Legend + text annotations show code + emotion name)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Custom colors for each label
emotion_colors = {
    "H": "red",
    "J": "orange",
    "A": "yellow",
    "G": "green",
    "I": "blue",
    "B": "purple",
    "L": "pink",
    "C": "cyan",
    "M": "magenta",
    "F": "gray",
    "K": "brown",
}

# Plot each emotion label in a distinct color
for code, (emotion_name, (vc, ac)) in EMOTION_CENTERS.items():
    # Subset of songs labeled with this emotion code
    subset = static_annotations[static_annotations["emotion_label"] == code]
    # Legend entry = "Code (Name)"
    legend_label = f"{code} ({emotion_name})"
    
    plt.scatter(
        subset["valence_mean"],
        subset["arousal_mean"],
        c=emotion_colors.get(code, "black"),
        label=legend_label,
        alpha=0.6,
        s=30
    )
    # Place text near the emotion center (vc, ac), showing "Code - Name"
    plt.text(
        vc + 0.05,        # offset text slightly so it doesn't overlap the center
        ac + 0.05,
        f"{code} - {emotion_name}",
        fontsize=8,
        alpha=0.8
    )

plt.title("Valence vs. Arousal, with Emotion Labels (Dimensional Approach)")
plt.xlabel("Valence Mean")
plt.ylabel("Arousal Mean")
plt.grid(alpha=0.3)
# Place legend outside the plot on the right
plt.legend(title="Emotion Codes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
save_scatter = os.path.join(plots_dir, "valence_arousal_emotion_classification.png")
plt.savefig(save_scatter)
plt.show()

# Show final distribution counts
counts = static_annotations["emotion_label"].value_counts()
print("\n=== Song Count per Emotion Category ===")
for code in counts.index:
    full_name = EMOTION_CENTERS[code][0]
    num = counts[code]
    print(f"  {code} ({full_name}): {num} songs")

print("\nAnalysis complete. All plots saved in the 'plots' folder.")
