import os
import time
import random
import requests
import pandas as pd

from dotenv import load_dotenv
from requests.exceptions import HTTPError

# ---------------------------------
# Spotipy
# ---------------------------------
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

# ---------------------------------
# Jamendo credentials
# ---------------------------------
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "")
# Jamendo does not strictly require a client secret for basic track searching,
# but you can store one if needed
JAMENDO_CLIENT_SECRET = os.getenv("JAMENDO_CLIENT_SECRET", "")

# ---------------------------------
# Authenticate with Spotify
# ---------------------------------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

# -------------------------------------------------------------------
# Utility to batch-fetch multiple Spotify IDs at once
# -------------------------------------------------------------------
def get_tracks_in_batches(spotify_ids):
    """Fetch up to 50 tracks from Spotify in a single call to sp.tracks()."""
    try:
        results = sp.tracks(spotify_ids)  # can handle up to 50 IDs
        return results.get('tracks', [])
    except HTTPError as e:
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 2))
            print(f"[Spotify] 429 Rate limit. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            return get_tracks_in_batches(spotify_ids)
        else:
            print(f"[Spotify] HTTP Error: {e}")
            return []
    except Exception as ex:
        print(f"[Spotify] Unexpected Error: {ex}")
        return []

# -------------------------------------------------------------------
# Jamendo Searching
# -------------------------------------------------------------------
def search_jamendo(track_name, artist_name):
    """
    Search for the track on Jamendo using the track + artist name.
    Returns a dict with track info if found, otherwise None.
    """
    if not JAMENDO_CLIENT_ID:
        # No credentials => skip
        return None

    # Basic search endpoint
    base_url = "https://api.jamendo.com/v3.0/tracks"
    query_params = {
        "client_id": JAMENDO_CLIENT_ID,
        "search": f"{track_name} {artist_name}",
        "limit": "1",
        # Possibly also add "include": "musicinfo,licenses" etc. if needed
    }

    try:
        response = requests.get(base_url, params=query_params)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                return data["results"][0]  # Only the first match
    except Exception as e:
        print(f"[Jamendo] Error searching for {track_name} - {artist_name}: {e}")

    return None

# -------------------------------------------------------------------
# Download audio preview
# -------------------------------------------------------------------
def download_audio_preview(preview_url, output_folder, filename_prefix):
    """
    Attempt to download the preview from preview_url.
    Returns the local file path if successful, else None.
    """
    try:
        response = requests.get(preview_url, stream=True)
        if response.status_code == 200:
            # remove forbidden characters from filename
            safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in "._-")
            local_filename = os.path.join(output_folder, f"{safe_prefix}.mp3")
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return local_filename
        else:
            print(f"[Download] Failed to download preview. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"[Download] Error downloading preview: {e}")
        return None

# -------------------------------------------------------------------
# Main function: Attempt Spotify, then fallback Jamendo
# -------------------------------------------------------------------
def gather_audio_previews(
    input_csv,
    filtered_csv,
    no_preview_csv,
    previews_folder,
    batch_size=50,
    total_duration=30
):
    """
    1) Read input_csv
    2) For rows with a valid 'spotify_id':
         - Attempt to batch fetch from Spotify.
         - If preview exists => download => append row to filtered_csv
         - If no preview => fallback to Jamendo search
    3) For rows with no 'spotify_id':
         - Go directly to Jamendo search
    4) If no preview from either => record in no_preview_csv
    5) Skip rows if their 'spotify_id' or (track+artist) is in either CSV.
    """
    os.makedirs(previews_folder, exist_ok=True)

    # Load the dataset
    df_input = pd.read_csv(input_csv, encoding="utf-8", low_memory=False)
    print(f"Loaded input dataset: {len(df_input)} rows.")

    # -------------------------------------------
    # Load or init "filtered_csv"
    # -------------------------------------------
    processed_spotify_ids_with_preview = set()
    processed_tuples_with_preview = set()  # (track, artist) or some combo for no-spotify fallback
    if os.path.exists(filtered_csv):
        df_filtered_existing = pd.read_csv(filtered_csv, encoding="utf-8")
        # build sets
        # for safety, track both spotify_id and track-artist combos
        if 'spotify_id' in df_filtered_existing.columns:
            processed_spotify_ids_with_preview = set(df_filtered_existing['spotify_id'].dropna().unique())
        # also track track+artist combos as fallback
        df_filtered_existing['track_artist'] = df_filtered_existing.apply(
            lambda r: (str(r['track']), str(r['artist'])), axis=1
        )
        processed_tuples_with_preview = set(df_filtered_existing['track_artist'].unique())
    else:
        with open(filtered_csv, 'w', encoding="utf-8") as f:
            df_input.iloc[0:0].to_csv(f, index=False)  # write header only

    # -------------------------------------------
    # Load or init "no_preview_csv"
    # -------------------------------------------
    processed_spotify_ids_no_preview = set()
    processed_tuples_no_preview = set()
    if os.path.exists(no_preview_csv):
        df_no_preview_existing = pd.read_csv(no_preview_csv, encoding="utf-8")
        if 'spotify_id' in df_no_preview_existing.columns:
            processed_spotify_ids_no_preview = set(df_no_preview_existing['spotify_id'].dropna().unique())
        df_no_preview_existing['track_artist'] = df_no_preview_existing.apply(
            lambda r: (str(r['track']), str(r['artist'])), axis=1
        )
        processed_tuples_no_preview = set(df_no_preview_existing['track_artist'].unique())
    else:
        with open(no_preview_csv, 'w', encoding="utf-8") as f:
            df_input.iloc[0:0].to_csv(f, index=False)  # write header only

    # Combined sets => skip if in either
    already_processed_spotify_ids = processed_spotify_ids_with_preview.union(processed_spotify_ids_no_preview)
    already_processed_tuples = processed_tuples_with_preview.union(processed_tuples_no_preview)

    # Filter the main DF to skip processed
    def is_processed(row):
        sid = str(row.get('spotify_id', ''))
        t_a = (str(row.get('track', '')), str(row.get('artist', '')))
        return (sid in already_processed_spotify_ids) or (t_a in already_processed_tuples)

    df_to_process = df_input[~df_input.apply(is_processed, axis=1)]
    rows_to_process = df_to_process.to_dict(orient='records')
    print(f"{len(rows_to_process)} rows remain to be processed (Spotify+Jamendo).")

    # -----------------------------------------------------------------
    # 1) BATCH SPOTIFY LOOKUPS
    # -----------------------------------------------------------------
    # We'll gather all rows that have a 'spotify_id', do them in chunks of batch_size.
    rows_with_spotify_id = [r for r in rows_to_process if pd.notnull(r.get('spotify_id', None))]
    print(f"Found {len(rows_with_spotify_id)} rows with a valid Spotify ID to process in batches.")

    i = 0
    while i < len(rows_with_spotify_id):
        batch = rows_with_spotify_id[i : i + batch_size]
        i += batch_size

        # gather spotify IDs in this batch
        sid_batch = [str(r['spotify_id']) for r in batch]
        # call sp.tracks() once
        tracks_data = get_tracks_in_batches(sid_batch)

        # build dict for quick lookup
        track_data_map = {t['id']: t for t in tracks_data if t}

        for row in batch:
            sid = str(row['spotify_id'])
            track_title = str(row.get('track', 'UnknownTrack'))
            artist_name = str(row.get('artist', 'UnknownArtist'))

            # see if track_data_map has a preview
            track_info = track_data_map.get(sid)
            if track_info and track_info.get('preview_url'):
                # Download from Spotify
                preview_url = track_info['preview_url']
                dl_path = download_audio_preview(
                    preview_url,
                    previews_folder,
                    filename_prefix=f"{sid}_{track_title}_{artist_name}"
                )
                if dl_path:
                    row['preview_path'] = dl_path
                    # append row to filtered_csv
                    with open(filtered_csv, 'a', encoding="utf-8") as f:
                        pd.DataFrame([row]).to_csv(f, index=False, header=False)
                    processed_spotify_ids_with_preview.add(sid)
                    processed_tuples_with_preview.add((track_title, artist_name))
                    print(f"[Spotify] {track_title} - {artist_name} => preview downloaded.")
                else:
                    # fallback to jamendo
                    fallback_jamendo(row, filtered_csv, no_preview_csv, previews_folder,
                                     processed_spotify_ids_with_preview,
                                     processed_tuples_with_preview,
                                     processed_spotify_ids_no_preview,
                                     processed_tuples_no_preview)
            else:
                # no preview => fallback jamendo
                fallback_jamendo(row, filtered_csv, no_preview_csv, previews_folder,
                                 processed_spotify_ids_with_preview,
                                 processed_tuples_with_preview,
                                 processed_spotify_ids_no_preview,
                                 processed_tuples_no_preview)

        # optional small sleep after batch
        time.sleep(0.1)

    # -----------------------------------------------------------------
    # 2) For rows with NO spotify_id => go directly to jamendo
    # -----------------------------------------------------------------
    rows_no_spotify_id = [r for r in rows_to_process if pd.isnull(r.get('spotify_id', None))]
    print(f"{len(rows_no_spotify_id)} rows have no Spotify ID, going directly to Jamendo...")

    for row in rows_no_spotify_id:
        fallback_jamendo(row, filtered_csv, no_preview_csv, previews_folder,
                         processed_spotify_ids_with_preview,
                         processed_tuples_with_preview,
                         processed_spotify_ids_no_preview,
                         processed_tuples_no_preview)

    print("\nGathering complete.")
    print(f"Total with previews so far: {len(processed_spotify_ids_with_preview)}")
    print(f"Total with no previews so far: {len(processed_spotify_ids_no_preview)}")

# -------------------------------------------------------------------
# fallback_jamendo: tries Jamendo if Spotify preview isn't found
# -------------------------------------------------------------------
def fallback_jamendo(row, filtered_csv, no_preview_csv, previews_folder,
                     processed_spotify_ids_with_preview, processed_tuples_with_preview,
                     processed_spotify_ids_no_preview, processed_tuples_no_preview):
    """
    Attempt Jamendo search if Spotify preview not found.
    If Jamendo has an audio link => download => filtered_csv
    Else => no_preview_csv
    """
    sid = str(row.get('spotify_id', ''))
    track_title = str(row.get('track', 'UnknownTrack'))
    artist_name = str(row.get('artist', 'UnknownArtist'))

    jamendo_info = search_jamendo(track_title, artist_name)
    if jamendo_info and 'audio' in jamendo_info:
        jamendo_url = jamendo_info['audio']
        dl_path = download_audio_preview(
            jamendo_url,
            previews_folder,
            filename_prefix=f"jamendo_{track_title}_{artist_name}"
        )
        if dl_path:
            row['preview_path'] = dl_path
            with open(filtered_csv, 'a', encoding="utf-8") as f:
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
            processed_spotify_ids_with_preview.add(sid)
            processed_tuples_with_preview.add((track_title, artist_name))
            print(f"[Jamendo] {track_title} - {artist_name} => preview downloaded.")
        else:
            # cannot download jamendo => no preview
            with open(no_preview_csv, 'a', encoding="utf-8") as f:
                pd.DataFrame([row]).to_csv(f, index=False, header=False)
            processed_spotify_ids_no_preview.add(sid)
            processed_tuples_no_preview.add((track_title, artist_name))
    else:
        # jamendo had no result
        with open(no_preview_csv, 'a', encoding="utf-8") as f:
            pd.DataFrame([row]).to_csv(f, index=False, header=False)
        processed_spotify_ids_no_preview.add(sid)
        processed_tuples_no_preview.add((track_title, artist_name))

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    input_csv = "muse_dataset.csv"           # your big dataset (90k songs)
    filtered_csv = "filtered_dataset.csv"    # songs with previews
    no_preview_csv = "no_preview_dataset.csv"
    previews_folder = "audio_previews"

    gather_audio_previews(
        input_csv=input_csv,
        filtered_csv=filtered_csv,
        no_preview_csv=no_preview_csv,
        previews_folder=previews_folder,
        batch_size=50,          # up to 50 for sp.tracks()
        total_duration=30       # 30s preview is standard, though not used here
    )
