import numpy as np

# Emotion centers from DEAM analysis
EMOTION_CENTERS = {
    "H": ("Erotic, desirous", (6.5, 5.0)),
    "J": ("Joyful, cheerful", (7.5, 6.0)),
    "A": ("Amusing", (7.0, 7.5)),
    "G": ("Energizing, pump-up", (8.0, 8.0)),
    "I": ("Indignant, defiant", (4.0, 6.5)),
    "B": ("Annoying", (4.5, 5.5)),
    "L": ("Scary, fearful", (3.5, 7.5)),
    "C": ("Anxious, tense", (3.5, 6.5)),
    "M": ("Triumphant, heroic", (7.8, 6.8)),
    "F": ("Dreamy", (7.0, 3.0)),
    "K": ("Sad, depressing", (2.5, 2.5)),
}

def classify_emotion_by_centroid(valence, arousal, centers_dict=EMOTION_CENTERS):
    """
    Classify emotion based on valence and arousal values using centroid-based classification.
    Returns a list of top emotions sorted by distance to the centroid.
    """
    distances = []
    for code, (name, (vc, ac)) in centers_dict.items():
        dist = np.sqrt((valence - vc)**2 + (arousal - ac)**2)
        distances.append((code, name, dist))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[2])
    
    # Return top 3 emotions with their distances
    top_emotions = []
    for code, name, dist in distances[:3]:
        top_emotions.append({
            "code": code,
            "name": name,
            "distance": float(dist)
        })
    
    return top_emotions

def get_emotion_predictions(valence, arousal):
    """
    Get emotion predictions for given valence and arousal values.
    Returns both the top emotion and a list of top 3 emotions with their distances.
    """
    top_emotions = classify_emotion_by_centroid(valence, arousal)
    
    return {
        "top_emotion": top_emotions[0],
        "top_emotions": top_emotions
    } 