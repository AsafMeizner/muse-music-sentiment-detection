from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, Any

from routers import genre, sentiment, websocket
from utils.emotion_classification import get_emotion_predictions

app = FastAPI(
    title="Music Analysis API",
    description="API for analyzing music genre and emotional content (valence/arousal)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(genre.router, prefix="/genre", tags=["genre"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(websocket.router, prefix="/realtime", tags=["realtime"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Music Analysis API",
        "docs": "/docs",
        "endpoints": {
            "genre": "/genre/predict",
            "sentiment": "/sentiment/predict",
            "combined": "/analysis/combined",
            "websocket": "/realtime/ws"
        }
    }

# Combined analysis endpoint
@app.post("/analysis/combined", response_model=Dict[str, Any])
async def analyze_combined(file: UploadFile = File(...)):
    """
    Analyze an audio file and return both genre prediction and sentiment analysis with emotions
    """
    try:
        # Get genre prediction
        genre_result = await genre.predict_genre(file)
        
        # Reset file position for sentiment analysis
        await file.seek(0)
        
        # Get sentiment analysis
        sentiment_result = await sentiment.predict_sentiment(file)
        
        # Extract average sentiment for emotion classification
        avg_valence = sentiment_result["overall_sentiment"]["valence"]
        avg_arousal = sentiment_result["overall_sentiment"]["arousal"]
        
        # Get emotion predictions
        emotion_predictions = get_emotion_predictions(avg_valence, avg_arousal)
        
        return {
            "genre": genre_result,
            "sentiment": {
                "overall": sentiment_result["overall_sentiment"],
                "emotions": emotion_predictions,
                "segments": sentiment_result["segment_analysis"]
            },
            "audio_info": sentiment_result["audio_info"],
            "processing_time": sentiment_result["processing_time"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 