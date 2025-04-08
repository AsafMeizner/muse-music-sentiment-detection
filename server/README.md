# Music Analysis API Server

A FastAPI application for analyzing music by genre and emotional content (valence/arousal).

## Features

- **Genre Classification**: Predict music genre from audio files
- **Sentiment Analysis**: Analyze valence and arousal emotional dimensions
- **Combined Analysis**: Get both genre and sentiment in one request
- **Real-time WebSocket API**: Streaming analysis for real-time applications

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the server

Run the following command from the `server` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

### API Documentation

Interactive API documentation is available at http://localhost:8000/docs

### API Endpoints

#### Genre Prediction

```
POST /genre/predict
```

Upload an audio file (MP3 or WAV) to get genre predictions.

#### Sentiment Analysis

```
POST /sentiment/predict
```

Upload an audio file to get valence and arousal predictions for each segment.

#### Combined Analysis

```
POST /analysis/combined
```

Upload an audio file to get both genre and sentiment predictions.

#### WebSocket Real-time Analysis

Connect to the WebSocket endpoint for real-time analysis:

```
ws://localhost:8000/realtime/ws
```

Message format for WebSocket:
```json
{
  "type": "audio_data",
  "data": "<base64-encoded-audio>",
  "format": "mp3",
  "analysis_type": "combined"
}
```

## Examples

### HTTP Request Example (Python)

```python
import requests

# Predict genre for an audio file
url = "http://localhost:8000/genre/predict"
files = {"file": open("song.mp3", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### WebSocket Example (JavaScript)

```javascript
const socket = new WebSocket('ws://localhost:8000/realtime/ws');

socket.onopen = () => {
  console.log('WebSocket connected');
  
  // Read file and convert to base64
  const fileReader = new FileReader();
  fileReader.onload = () => {
    const base64 = fileReader.result.split(',')[1];
    
    // Send data for analysis
    socket.send(JSON.stringify({
      type: 'audio_data',
      data: base64,
      format: 'mp3',
      analysis_type: 'combined'
    }));
  };
  
  // Read a file input from a form
  const fileInput = document.getElementById('audioFile');
  fileReader.readAsDataURL(fileInput.files[0]);
};

socket.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Analysis result:', response);
};
```

## Model Information

- **Genre Model**: CNN trained on GTZAN dataset with 10 genre classes
- **Sentiment Model**: VGG-like CNN trained on DEAM dataset for valence/arousal prediction 