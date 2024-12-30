import os
import time
import requests
import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from requests.exceptions import HTTPError

# ---------------------------------------------
# Spotify API credentials
# ---------------------------------------------
SPOTIPY_CLIENT_ID = "584d6c6043e6476389fc218168cf3a4e"
SPOTIPY_CLIENT_SECRET = "f6f7cb30d99643bda29c9d55d4a9b043"

# Authenticate with Spotify API
sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

def get_tracks_in_batches(spotify_ids):
    """
    Takes a list of up to 50 Spotify IDs, calls sp.tracks() once.
    Returns a list of track data. If a 429 rate limit is hit, we back off and retry.
    """
    try:
        results = sp.tracks(spotify_ids)  # sp.tracks() can handle up to 50 IDs at once
        return results.get('tracks', [])
    except HTTPError as e:
        # 429 => Rate limited
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 2))
            print(f"[Batch] 429 Rate limit. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            return get_tracks_in_batches(spotify_ids)  # retry
        else:
            print(f"[Batch] HTTP Error: {e}")
            return []
    except Exception as ex:
        print(f"[Batch] Unexpected Error: {ex}")
        return []

def download_audio_preview(preview_url, output_folder, spotify_id):
    """Download audio preview and return the local file path, or None if failed."""
    try:
        response = requests.get(preview_url, stream=True)
        if response.status_code == 200:
            local_filename = os.path.join(output_folder, f"{spotify_id}.mp3")
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return local_filename
        else:
            print(f"Failed to download preview for {spotify_id}. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading preview for {spotify_id}: {e}")
        return None

def filter_songs_with_audio_samples(
    input_csv,
    filtered_csv,
    no_preview_csv,
    previews_folder,
    batch_size=50
):
    """
    1) Reads the dataset from input_csv.
    2) Batches Spotify IDs in chunks of `batch_size` (up to 50).
    3) Calls sp.tracks() once per batch => reduces API calls drastically.
    4) For each track in a batch:
       - If preview_url, downloads => writes row to 'filtered_csv'
       - Else => writes row to 'no_preview_csv'
    5) Skips IDs already present in either CSV (processed).
    6) Adds a brief sleep after each batch to reduce rate-limiting risk.
    """
    # Ensure the previews folder exists
    os.makedirs(previews_folder, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(input_csv, encoding="utf-8")
    print(f"Processing {len(data)} songs from the input CSV.")

    # --------------------------------------------
    # Load or initialize "filtered_csv" (has preview)
    # --------------------------------------------
    processed_spotify_ids_with_preview = set()
    if os.path.exists(filtered_csv):
        existing_filtered = pd.read_csv(filtered_csv, encoding="utf-8")
        processed_spotify_ids_with_preview = set(existing_filtered['spotify_id'].unique())
    else:
        # Initialize an empty CSV if it doesn't exist
        with open(filtered_csv, 'w', encoding="utf-8") as f:
            data.iloc[0:0].to_csv(f, index=False)  # write header only

    # --------------------------------------------
    # Load or initialize "no_preview_csv" (no preview)
    # --------------------------------------------
    processed_spotify_ids_no_preview = set()
    if os.path.exists(no_preview_csv):
        existing_no_preview = pd.read_csv(no_preview_csv, encoding="utf-8")
        processed_spotify_ids_no_preview = set(existing_no_preview['spotify_id'].unique())
    else:
        # Initialize an empty CSV if it doesn't exist
        with open(no_preview_csv, 'w', encoding="utf-8") as f:
            data.iloc[0:0].to_csv(f, index=False)  # write header only

    # Combined set => skip if already processed
    already_processed = processed_spotify_ids_with_preview.union(processed_spotify_ids_no_preview)

    # Filter out rows that have already been processed in either CSV
    df_to_process = data[~data['spotify_id'].isin(already_processed)]
    print(f"{len(df_to_process)} songs remain to be processed.")

    # Convert df_to_process to a list of dicts for easy iteration
    rows_to_process = df_to_process.to_dict(orient='records')

    # --------------------------------------------
    # Process in batches
    # --------------------------------------------
    for i in range(0, len(rows_to_process), batch_size):
        batch = rows_to_process[i : i + batch_size]
        spotify_ids_batch = [row['spotify_id'] for row in batch]

        print(f"\nProcessing batch of {len(batch)} songs (indices {i} to {i+len(batch)-1}).")

        # Call sp.tracks() once for up to 50 IDs
        tracks_data = get_tracks_in_batches(spotify_ids_batch)

        # Dictionary for quick lookup by Spotify ID
        # tracks_data is a list of dicts; each dict has 'id' and 'preview_url', etc.
        track_data_map = {t['id']: t for t in tracks_data if t}

        # For each row in the batch, see if there's a preview
        for row in batch:
            spotify_id = row['spotify_id']
            track_info = track_data_map.get(spotify_id)

            if track_info and track_info.get('preview_url'):
                # There's a valid preview
                preview_url = track_info['preview_url']
                local_file_path = download_audio_preview(preview_url, previews_folder, spotify_id)
                if local_file_path:
                    row['preview_path'] = local_file_path
                    # Append row to filtered_csv
                    with open(filtered_csv, 'a', encoding="utf-8") as f:
                        pd.DataFrame([row]).to_csv(f, index=False, header=False)
                    processed_spotify_ids_with_preview.add(spotify_id)
                    print(f"  -> Downloaded preview for {row.get('track')} ({spotify_id})")
                else:
                    # Could not download => treat as no preview
                    with open(no_preview_csv, 'a', encoding="utf-8") as f:
                        pd.DataFrame([row]).to_csv(f, index=False, header=False)
                    processed_spotify_ids_no_preview.add(spotify_id)
            else:
                # No track_info or no preview_url => no preview
                with open(no_preview_csv, 'a', encoding="utf-8") as f:
                    pd.DataFrame([row]).to_csv(f, index=False, header=False)
                processed_spotify_ids_no_preview.add(spotify_id)

        # Optional: short sleep to avoid bursts
        time.sleep(0.05)

    print("\nProcessing complete.")
    print(f"Tracks with previews so far: {len(processed_spotify_ids_with_preview)}")
    print(f"Tracks without previews so far: {len(processed_spotify_ids_no_preview)}")

# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------
if __name__ == "__main__":
    input_csv = "muse_dataset.csv"
    filtered_csv = "filtered_dataset.csv"
    no_preview_csv = "no_preview_dataset.csv"
    previews_folder = "audio_previews"

    filter_songs_with_audio_samples(
        input_csv=input_csv,
        filtered_csv=filtered_csv,
        no_preview_csv=no_preview_csv,
        previews_folder=previews_folder,
        batch_size=50  # up to 50 for sp.tracks()
    )
