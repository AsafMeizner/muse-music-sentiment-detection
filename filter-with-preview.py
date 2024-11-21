import pandas as pd
import os
import time
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from requests.exceptions import HTTPError

# Spotify API credentials
SPOTIPY_CLIENT_ID = "584d6c6043e6476389fc218168cf3a4e"
SPOTIPY_CLIENT_SECRET = "f6f7cb30d99643bda29c9d55d4a9b043"

# Authenticate with Spotify API
sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

def check_audio_sample(spotify_id):
    """Check if a song has an audio sample on Spotify."""
    try:
        track = sp.track(spotify_id)
        return track.get('preview_url') is not None
    except HTTPError as e:
        if e.response.status_code == 429:  # Rate limit error
            retry_after = int(e.response.headers.get("Retry-After", 1))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            return check_audio_sample(spotify_id)  # Retry the request
        else:
            print(f"Error checking Spotify ID {spotify_id}: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error for Spotify ID {spotify_id}: {e}")
        return False

def filter_songs_with_audio_samples(input_csv, output_csv):
    """Filter songs with audio samples and write them directly to the output file."""
    # Load the dataset
    data = pd.read_csv(input_csv)
    
    # Check if the output file already exists
    if os.path.exists(output_csv):
        processed_data = pd.read_csv(output_csv)
        processed_spotify_ids = set(processed_data['spotify_id'])
    else:
        # Initialize an empty CSV if it doesn't exist
        with open(output_csv, 'w') as f:
            data.iloc[0:0].to_csv(f, index=False)  # Write header only
        processed_spotify_ids = set()

    # Process songs
    for index, row in data.iterrows():
        spotify_id = row.get('spotify_id')
        if spotify_id in processed_spotify_ids:
            continue  # Skip already processed songs
        
        if spotify_id and check_audio_sample(spotify_id):
            # Append the valid song to the output file
            with open(output_csv, 'a') as f:
                row.to_frame().T.to_csv(f, index=False, header=False)
            
            print(f"Added: {row['track']} - {row['artist']} with Spotify ID {spotify_id}")
        
        processed_spotify_ids.add(spotify_id)  # Mark as processed

    print(f"Filtering completed. Filtered dataset saved to {output_csv}")

# Input and output file paths
input_csv = "muse_dataset.csv"  # Original dataset
output_csv = "filtered_dataset.csv"  # File for songs with audio samples

# Run the filtering process
filter_songs_with_audio_samples(input_csv, output_csv)