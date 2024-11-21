import pandas as pd
import random
import requests
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# Replace these with your Spotify API credentials
SPOTIPY_CLIENT_ID = "584d6c6043e6476389fc218168cf3a4e"
SPOTIPY_CLIENT_SECRET = "f6f7cb30d99643bda29c9d55d4a9b043"

# Authenticate with Spotify API
sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

def choose_random_song(csv_path):
    """Selects a random song from the dataset."""
    df = pd.read_csv(csv_path)
    random_entry = df.sample(1).iloc[0]
    
    song_data = {
        "track": random_entry['track'],
        "artist": random_entry['artist'],
        "lastfm_url": random_entry['lastfm_url'],
        "spotify_id": random_entry.get('spotify_id', None)
    }
    print(f"Selected: {song_data['track']} by {song_data['artist']}")
    return song_data

def download_audio(song_data):
    """Downloads audio preview from Spotify."""
    spotify_id = song_data.get("spotify_id")
    if spotify_id:
        # Fetch track details using Spotify ID
        track = sp.track(spotify_id)
        preview_url = track.get('preview_url')
        
        if preview_url:
            # Download the preview
            response = requests.get(preview_url)
            file_name = f"{song_data['track']} - {song_data['artist']}.mp3"
            file_name = file_name.replace('/', '_')  # Sanitize file name
            
            with open(file_name, 'wb') as f:
                f.write(response.content)
            
            print(f"Audio preview downloaded: {file_name}")
        else:
            print("No audio preview available on Spotify.")
    else:
        print("Spotify ID not available for this track.")

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = "muse_dataset.csv"
    
    # Choose a random song and download its audio preview
    song = choose_random_song(csv_file_path)
    download_audio(song)