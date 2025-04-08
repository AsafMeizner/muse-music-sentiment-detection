from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import os
import tempfile
import time
from typing import Dict, List, Any
import logging

from models.model_loader import model_loader
from utils.audio_processing import (
    save_upload_to_temp,
    load_audio,
    segment_audio_for_sentiment, 
    compute_mel_spectrogram_for_sentiment,
    invert_sentiment_label
)
from utils.emotion_classification import get_emotion_predictions

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=Dict[str, Any])
async def predict_sentiment(file: UploadFile = File(...)):
    """
    Predict sentiment (valence and arousal) and emotions from an audio file.
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="File must be WAV or MP3 format")

    try:
        # Save uploaded file to temporary location
        temp_path = await save_upload_to_temp(file)
        logger.info(f"Saved upload to {temp_path}")
        
        # Load audio
        y, sr = load_audio(temp_path)
        
        # Segment audio 
        segments = segment_audio_for_sentiment(y, sr)
        logger.info(f"Audio segmented into {len(segments)} chunks for sentiment analysis")
        
        # Process each segment
        segment_results = []
        mel_spectrograms = []
        total_duration = len(y) / sr
        
        for i, segment in enumerate(segments):
            # Compute mel spectrogram for sentiment
            mel_spec = compute_mel_spectrogram_for_sentiment(segment, sr)
            mel_spectrograms.append(mel_spec)
            
            # Calculate time range for this segment
            segment_duration = len(segment) / sr
            start_time_sec = i * segment_duration
            end_time_sec = start_time_sec + segment_duration
            
            segment_results.append({
                "segment_index": i,
                "time_range": {
                    "start": start_time_sec,
                    "end": end_time_sec
                }
            })
        
        # Load model (lazy loading)
        model = model_loader.load_sentiment_model()
        
        # Predict
        predictions = model.predict(np.array(mel_spectrograms), verbose=0)
        
        # Add predictions to segment results
        for i, prediction in enumerate(predictions):
            # Convert scaled predictions back to original scale (1-9)
            valence = float(invert_sentiment_label(prediction[0]))
            arousal = float(invert_sentiment_label(prediction[1]))
            
            segment_results[i].update({
                "valence": valence,
                "arousal": arousal,
                "scaled_prediction": {
                    "valence": float(prediction[0]),
                    "arousal": float(prediction[1])
                }
            })
        
        # Calculate average sentiment
        avg_valence = np.mean([seg["valence"] for seg in segment_results])
        avg_arousal = np.mean([seg["arousal"] for seg in segment_results])
        
        # Get emotion predictions based on average sentiment
        emotion_predictions = get_emotion_predictions(avg_valence, avg_arousal)
        
        # Calculate time taken
        elapsed_time = time.time() - start_time
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            "overall_sentiment": {
                "valence": float(avg_valence),
                "arousal": float(avg_arousal)
            },
            "emotions": emotion_predictions,
            "segment_analysis": segment_results,
            "audio_info": {
                "duration": total_duration,
                "num_segments": len(segments)
            },
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        logger.error(f"Error predicting sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting sentiment: {str(e)}")

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