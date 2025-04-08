import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from tensorflow.image import resize # type: ignore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 22050
GENRE_SEGMENT_DURATION = 4  # 4 seconds per segment for genre prediction (same as used in training)
SENTIMENT_SEGMENT_DURATION = 10  # 10 seconds per segment for sentiment prediction
TARGET_FRAMES = 200  # For sentiment prediction
N_MELS = 128  # Number of mel bands to generate

def save_upload_to_temp(upload_file):
    """Save an uploaded file to a temporary file and return the path"""
    try:
        # Create a temporary file with the correct extension
        suffix = os.path.splitext(upload_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write uploaded file content to the temp file
            content = upload_file.file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        logger.error(f"Error saving upload to temp file: {e}")
        raise e

def load_audio(file_path, sr=SAMPLE_RATE):
    """Load audio file and return audio data and sample rate"""
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise e

def segment_audio_for_genre(y, sr, segment_duration=GENRE_SEGMENT_DURATION):
    """
    Segment audio into chunks for genre prediction
    Returns list of segments with 2 seconds overlap
    """
    segments = []
    overlap_duration = segment_duration / 2  # 50% overlap
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    
    # Calculate number of segments
    total_samples = len(y)
    if total_samples < segment_samples:
        # If audio is shorter than segment duration, pad it
        y = np.pad(y, (0, segment_samples - total_samples), 'constant')
        segments.append(y)
    else:
        # Create overlapping segments
        start = 0
        while start + segment_samples <= total_samples:
            segment = y[start:start + segment_samples]
            segments.append(segment)
            start += segment_samples - overlap_samples
    
    return segments

def segment_audio_for_sentiment(y, sr, segment_duration=SENTIMENT_SEGMENT_DURATION):
    """
    Segment audio into chunks for sentiment prediction
    Returns list of segments with no overlap for sentiment analysis
    """
    segments = []
    segment_samples = int(segment_duration * sr)
    
    # Calculate number of segments
    total_samples = len(y)
    if total_samples < segment_samples:
        # If audio is shorter than segment duration, pad it
        y = np.pad(y, (0, segment_samples - total_samples), 'constant')
        segments.append(y)
    else:
        # Create non-overlapping segments
        num_segments = total_samples // segment_samples
        for i in range(num_segments):
            start = i * segment_samples
            segment = y[start:start + segment_samples]
            segments.append(segment)
    
    return segments

def compute_mel_spectrogram_for_genre(y, sr, target_shape=(150, 150)):
    """
    Compute mel spectrogram for genre prediction and resize to target shape
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Add channel dimension
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    # Resize spectrogram to target shape
    mel_spec_resized = resize(mel_spec_db, target_shape).numpy()
    return mel_spec_resized

def compute_mel_spectrogram_for_sentiment(y, sr, target_frames=TARGET_FRAMES):
    """
    Compute mel spectrogram for sentiment prediction with fixed time dimension
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    # Convert to dB scale
    S_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Fix spectrogram length
    n_mels, T = S_db.shape
    if T >= target_frames:
        S_fixed = S_db[:, :target_frames]
    else:
        pad_width = target_frames - T
        S_fixed = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
    
    # Transpose to (time, frequency) format and add channel dimension
    S_fixed = S_fixed.T
    S_fixed = np.expand_dims(S_fixed, -1)
    return S_fixed

def scale_sentiment_label(label):
    """Scale sentiment label from original scale (1-9) to model scale"""
    return (label - 5.0) * 0.05

def invert_sentiment_label(scaled):
    """Convert scaled sentiment value back to original scale (1-9)"""
    return (scaled / 0.05) + 5.0 