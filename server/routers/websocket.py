from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import numpy as np
import os
import json
import tempfile
import base64
import asyncio
import time
import logging
from typing import Dict, List

from models.model_loader import model_loader
from utils.audio_processing import (
    load_audio,
    segment_audio_for_genre,
    segment_audio_for_sentiment,
    compute_mel_spectrogram_for_genre,
    compute_mel_spectrogram_for_sentiment,
    invert_sentiment_label,
    SAMPLE_RATE
)

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total active: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

# Create connection manager instance
manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio analysis
    
    Receives audio data in base64 format, processes it, and returns analysis results
    
    Message format (from client):
    {
        "type": "audio_data",
        "data": "<base64-encoded-audio>",
        "format": "mp3" or "wav",
        "analysis_type": "genre", "sentiment", or "combined"
    }
    """
    await manager.connect(websocket)
    try:
        # Load models in advance to avoid delay during processing
        genre_model = model_loader.load_genre_model()
        sentiment_model = model_loader.load_sentiment_model()
        genre_classes = model_loader.genre_classes
        
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            # Check message type
            if message.get("type") != "audio_data":
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid message type. Expected 'audio_data'."
                })
                continue
            
            # Check required fields
            if not all(k in message for k in ["data", "format", "analysis_type"]):
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Missing required fields: data, format, or analysis_type"
                })
                continue
            
            # Get message data
            audio_base64 = message["data"]
            audio_format = message["format"].lower()
            analysis_type = message["analysis_type"].lower()
            
            # Validate format
            if audio_format not in ["mp3", "wav"]:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid audio format. Supported formats: mp3, wav"
                })
                continue
            
            # Validate analysis type
            if analysis_type not in ["genre", "sentiment", "combined"]:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid analysis type. Supported types: genre, sentiment, combined"
                })
                continue
            
            try:
                # Process the audio data
                start_time = time.time()
                
                # Decode base64 data
                audio_bytes = base64.b64decode(audio_base64)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name
                
                # Load audio
                y, sr = load_audio(temp_path)
                
                # Process based on analysis type
                result = {}
                
                if analysis_type in ["genre", "combined"]:
                    # Perform genre analysis
                    segments = segment_audio_for_genre(y, sr)
                    
                    segment_predictions = []
                    segment_genres = []
                    
                    for segment in segments:
                        mel_spec = compute_mel_spectrogram_for_genre(segment, sr, target_shape=(150, 150))
                        mel_spec = np.expand_dims(mel_spec, axis=0)
                        segment_pred = genre_model.predict(mel_spec, verbose=0)[0]
                        segment_predictions.append(segment_pred)
                        segment_genre = genre_classes[np.argmax(segment_pred)]
                        segment_genres.append(segment_genre)
                    
                    avg_prediction = np.mean(np.array(segment_predictions), axis=0)
                    sorted_indices = np.argsort(avg_prediction)[::-1]
                    top_genres = [genre_classes[i] for i in sorted_indices[:3]]
                    top_confidences = [float(avg_prediction[i]) for i in sorted_indices[:3]]
                    
                    genre_counts = {}
                    for genre in segment_genres:
                        if genre in genre_counts:
                            genre_counts[genre] += 1
                        else:
                            genre_counts[genre] = 1
                    
                    result["genre"] = {
                        "predicted_genre": top_genres[0],
                        "confidence": top_confidences[0],
                        "top_genres": [
                            {"genre": genre, "confidence": conf} 
                            for genre, conf in zip(top_genres, top_confidences)
                        ],
                        "segment_distribution": {
                            genre: count/len(segments) for genre, count in genre_counts.items()
                        }
                    }
                
                if analysis_type in ["sentiment", "combined"]:
                    # Perform sentiment analysis
                    segments = segment_audio_for_sentiment(y, sr)
                    
                    segment_results = []
                    total_duration = len(y) / sr
                    
                    for i, segment in enumerate(segments):
                        mel_spec = compute_mel_spectrogram_for_sentiment(segment, sr)
                        mel_spec = np.expand_dims(mel_spec, axis=0)
                        prediction = sentiment_model.predict(mel_spec, verbose=0)[0]
                        
                        valence = float(invert_sentiment_label(prediction[0]))
                        arousal = float(invert_sentiment_label(prediction[1]))
                        
                        segment_duration = len(segment) / sr
                        start_time_sec = i * segment_duration
                        end_time_sec = start_time_sec + segment_duration
                        
                        segment_results.append({
                            "segment_index": i,
                            "time_range": {
                                "start": start_time_sec,
                                "end": end_time_sec
                            },
                            "valence": valence,
                            "arousal": arousal
                        })
                    
                    avg_valence = np.mean([seg["valence"] for seg in segment_results])
                    avg_arousal = np.mean([seg["arousal"] for seg in segment_results])
                    
                    result["sentiment"] = {
                        "overall": {
                            "valence": float(avg_valence),
                            "arousal": float(avg_arousal)
                        },
                        "segments": segment_results
                    }
                
                # Add audio info and processing time
                elapsed_time = time.time() - start_time
                result["audio_info"] = {
                    "duration": len(y) / sr,
                    "sample_rate": sr
                }
                result["processing_time"] = elapsed_time
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Send result
                await manager.send_json(websocket, {
                    "type": "analysis_result",
                    "analysis_type": analysis_type,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}")
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Error processing audio data: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket) 