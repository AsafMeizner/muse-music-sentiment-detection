import pandas as pd
import os
import time
import requests
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
        preview_url = track.get('preview_url')
        return preview_url
    except HTTPError as e:
        if e.response.status_code == 429:  # Rate limit error
            retry_after = int(e.response.headers.get("Retry-After", 1))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            return check_audio_sample(spotify_id)  # Retry the request
        else:
            print(f"Error checking Spotify ID {spotify_id}: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error for Spotify ID {spotify_id}: {e}")
        return None

def download_audio_preview(preview_url, output_folder, spotify_id):
    """Download audio preview and return the local file path."""
    try:
        response = requests.get(preview_url, stream=True)
        if response.status_code == 200:
            local_filename = os.path.join(output_folder, f"{spotify_id}.mp3")
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return local_filename
        else:
            print(f"Failed to download preview for {spotify_id}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading preview for {spotify_id}: {e}")
        return None

def filter_songs_with_audio_samples(input_csv, output_csv, previews_folder):
    """Filter songs with audio samples, download previews, and update the output CSV dynamically."""
    # Ensure the previews folder exists
    os.makedirs(previews_folder, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(input_csv, encoding="utf-8")
    print(f"Processing {len(data)} songs from the input CSV.")

    # Check if the output file already exists and load processed Spotify IDs
    processed_spotify_ids = set()
    if os.path.exists(output_csv):
        processed_data = pd.read_csv(output_csv, encoding="utf-8")
        processed_spotify_ids = set(processed_data['spotify_id'])
    else:
        # Initialize an empty CSV if it doesn't exist
        with open(output_csv, 'w', encoding="utf-8") as f:
            data.iloc[0:0].to_csv(f, index=False)  # Write header only

    # Process songs
    for index, row in data.iterrows():
        print(f"Processing song {index + 1}/{len(data)}: {row['track']} - {row['artist']}")
        spotify_id = row.get('spotify_id')
        if spotify_id in processed_spotify_ids:
            print(f"Skipping already processed song: {spotify_id}")
            continue  # Skip already processed songs
        
        preview_url = check_audio_sample(spotify_id)
        if preview_url:
            local_file_path = download_audio_preview(preview_url, previews_folder, spotify_id)
            if local_file_path:
                # Add the local file path to the row
                row['preview_path'] = local_file_path
                
                # Append the valid song to the output file
                with open(output_csv, 'a', encoding="utf-8") as f:
                    row.to_frame().T.to_csv(f, index=False, header=f.tell() == 0)  # Write header only once
                print(f"Added: {row['track']} - {row['artist']} with Spotify ID {spotify_id}")
        
        processed_spotify_ids.add(spotify_id)  # Mark as processed

    print(f"Filtering completed. Filtered dataset saved to {output_csv}")

# Input and output file paths
input_csv = "muse_dataset.csv"  # Original dataset
output_csv = "filtered_dataset.csv"  # File for songs with audio samples
previews_folder = "audio_previews"  # Folder to save audio previews

# Run the filtering process
filter_songs_with_audio_samples(input_csv, output_csv, previews_folder)
