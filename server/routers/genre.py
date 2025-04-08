from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import os
import tempfile
import time
import tensorflow as tf
from typing import Dict, List
import logging

from models.model_loader import model_loader
from utils.audio_processing import (
    save_upload_to_temp, 
    load_audio, 
    segment_audio_for_genre,
    compute_mel_spectrogram_for_genre
)

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict the genre of an uploaded audio file (.mp3 or .wav)
    
    The function:
    1. Saves the uploaded file to a temporary location
    2. Loads the audio file and segments it
    3. Generates mel-spectrograms for each segment
    4. Predicts genre for each segment
    5. Aggregates results and returns prediction
    """
    start_time = time.time()
    
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename.lower()
    if not (filename.endswith('.mp3') or filename.endswith('.wav')):
        raise HTTPException(status_code=400, detail="File must be MP3 or WAV format")
    
    try:
        # Save upload to temp file
        temp_path = await save_upload_to_temp(file)
        logger.info(f"Saved upload to {temp_path}")
        
        # Load audio
        y, sr = load_audio(temp_path)
        
        # Load model (lazy loading)
        model = model_loader.load_genre_model()
        genre_classes = model_loader.genre_classes
        
        # Segment audio
        segments = segment_audio_for_genre(y, sr)
        logger.info(f"Audio segmented into {len(segments)} chunks")
        
        # Process each segment
        segment_predictions = []
        segment_genres = []
        
        for i, segment in enumerate(segments):
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram_for_genre(segment, sr, target_shape=(150, 150))
            
            # Add batch dimension
            mel_spec = np.expand_dims(mel_spec, axis=0)
            
            # Predict
            segment_pred = model.predict(mel_spec, verbose=0)[0]
            segment_predictions.append(segment_pred)
            
            # Get top genre for this segment
            segment_genre = genre_classes[np.argmax(segment_pred)]
            segment_genres.append(segment_genre)
        
        # Compute average prediction across all segments
        avg_prediction = np.mean(np.array(segment_predictions), axis=0)
        
        # Calculate confidence and top genres
        sorted_indices = np.argsort(avg_prediction)[::-1]
        top_genres = [genre_classes[i] for i in sorted_indices[:3]]
        top_confidences = [float(avg_prediction[i]) for i in sorted_indices[:3]]
        
        # Count occurrences of each genre prediction in segments
        genre_counts = {}
        for genre in segment_genres:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1
        
        # Calculate time taken
        elapsed_time = time.time() - start_time
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Return result
        return {
            "predicted_genre": top_genres[0],
            "confidence": top_confidences[0],
            "top_genres": [
                {"genre": genre, "confidence": conf} 
                for genre, conf in zip(top_genres, top_confidences)
            ],
            "segment_analysis": {
                "num_segments": len(segments),
                "segment_distribution": {
                    genre: count/len(segments) for genre, count in genre_counts.items()
                },
            },
            "processing_time": elapsed_time
        }
    
    except Exception as e:
        logger.error(f"Error predicting genre: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting genre: {str(e)}")

async def save_upload_to_temp(upload_file):
    """Save the uploaded file to a temporary location"""
    try:
        suffix = os.path.splitext(upload_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await upload_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise e 