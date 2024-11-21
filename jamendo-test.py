import pandas as pd
import requests
import random
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Your Jamendo API credentials
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")
JAMENDO_CLIENT_SECRET = os.getenv("JAMENDO_CLIENT_SECRET")

# Load the MuSe dataset
def load_dataset(csv_path):
    """Load the MuSe dataset from a CSV file."""
    try:
        data = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

# Select a random song from the dataset
def get_random_song(data):
    """Get a random song from the dataset."""
    random_index = random.randint(0, len(data) - 1)
    song = data.iloc[random_index]
    print(f"Randomly selected song: {song['track']} by {song['artist']}")
    return song

# Search for the song on Jamendo
def search_jamendo(track, artist):
    """Search for a track on Jamendo."""
    url = f"https://api.jamendo.com/v3.0/tracks?client_id={JAMENDO_CLIENT_ID}&search={track} {artist}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            track_info = data["results"][0]
            print(f"Found track on Jamendo: {track_info['name']} by {track_info['artist_name']}")
            print(f"Audio Preview URL: {track_info['audio']}")
            return track_info
    print("No matching track found on Jamendo.")
    return None

# Download audio
def download_audio(audio_url, output_filename):
    """Download the audio file from the provided URL."""
    try:
        response = requests.get(audio_url)
        if response.status_code == 200:
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"Audio downloaded successfully: {output_filename}")
        else:
            print(f"Failed to download audio: {response.status_code}")
    except Exception as e:
        print(f"Error during download: {e}")

# Main script
if __name__ == "__main__":
    dataset_path = "muse_dataset.csv"  # Path to your MuSe dataset CSV file
    data = load_dataset(dataset_path)

    if data is not None:
        # Get a random song from the dataset
        random_song = get_random_song(data)

        # Search for the song on Jamendo
        track_name = random_song['track']
        artist_name = random_song['artist']
        jamendo_track = search_jamendo(track_name, artist_name)

        # Download the audio if found
        if jamendo_track and "audio" in jamendo_track:
            audio_url = jamendo_track["audio"]
            output_file = f"{track_name.replace(' ', '_')}_{artist_name.replace(' ', '_')}.mp3"
            download_audio(audio_url, output_file)
        else:
            print("No audio available for the selected song.")
