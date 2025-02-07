import os
import sys
import pandas as pd
from mutagen.id3 import ID3, ID3NoHeaderError, TIT2, TPE1, TCON
from mutagen.mp3 import MP3
from mutagen import MutagenError
import logging
import subprocess

# ---------------------------
# Configuration
# ---------------------------

# Path to your CSV file
CSV_PATH = 'filtered_dataset.csv'  # Ensure this is correct

# Base directory for audio files (relative to CSV_PATH or absolute)
BASE_DIR = '.'  # '.' means current directory

# Whether to re-encode MP3 files using ffmpeg
REENCODE = True  # Set to True to re-encode corrupted files

# Metadata columns in your CSV to add as ID3 tags
METADATA_COLS = ['track', 'artist', 'genre']

# Log file name
LOG_FILE = 'fix_metadata.log'

# ---------------------------
# Logging Configuration
# ---------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# ---------------------------
# Function Definitions
# ---------------------------

def is_id3_tag_corrupted(file_path):
    """
    Checks if the MP3 file has corrupted or malformed ID3 tags.
    Returns True if corrupted, False otherwise.
    """
    try:
        audio = MP3(file_path, ID3=ID3)
        if audio.tags is not None:
            # Attempt to access tags
            _ = audio.tags.keys()
        return False  # If no exception, assume tags are fine
    except (ID3NoHeaderError, MutagenError) as e:
        # ID3NoHeaderError means no tags, which is not corrupted
        # Any other MutagenError likely indicates corrupted tags
        if isinstance(e, ID3NoHeaderError):
            return False
        logger.warning(f"Corrupted ID3 tags detected in {file_path}: {e}")
        return True

def remove_id3_tags(file_path):
    """
    Removes all ID3 tags from the specified MP3 file.
    """
    try:
        audio = MP3(file_path, ID3=ID3)
        audio.delete()
        audio.save()
        logger.info(f"Removed corrupted ID3 tags from: {file_path}")
    except ID3NoHeaderError:
        logger.info(f"No ID3 tags to remove in: {file_path}")
    except MutagenError as e:
        logger.error(f"Error removing ID3 tags from {file_path}: {e}")

def add_id3_tags(file_path, metadata):
    """
    Adds ID3 tags to the specified MP3 file using the provided metadata dictionary.
    """
    try:
        audio = MP3(file_path, ID3=ID3)
        try:
            audio.add_tags()
        except MutagenError:
            pass  # Tags already exist

        # Add or update tags
        if 'track' in metadata and metadata['track']:
            audio.tags.add(TIT2(encoding=3, text=str(metadata['track'])))  # Track as Title
        if 'artist' in metadata and metadata['artist']:
            audio.tags.add(TPE1(encoding=3, text=str(metadata['artist'])))  # Artist
        if 'genre' in metadata and metadata['genre']:
            audio.tags.add(TCON(encoding=3, text=str(metadata['genre'])))    # Genre

        audio.save()
        logger.info(f"Added/Updated ID3 tags to: {file_path}")
    except MutagenError as e:
        logger.error(f"Error adding ID3 tags to {file_path}: {e}")

def reencode_mp3(file_path):
    """
    Re-encodes an MP3 file using ffmpeg to fix potential audio corruption.
    Requires ffmpeg to be installed and accessible via the command line.
    """
    try:
        # Define a temporary file path
        temp_file = file_path + ".temp.mp3"

        # ffmpeg command to re-encode the MP3
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite without asking
            '-i', file_path,
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',  # High quality
            temp_file
        ]

        logger.info(f"Re-encoding file: {file_path}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed for {file_path}: {result.stderr}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

        # Replace the original file with the re-encoded file
        os.replace(temp_file, file_path)
        logger.info(f"Re-encoded and replaced: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error re-encoding {file_path}: {e}")
        return False

def fix_corrupted_mp3s(csv_path, base_dir, reencode=False):
    """
    Processes MP3 files listed in the CSV:
    1. Identifies corrupted ID3 tags.
    2. Optionally re-encodes the audio to fix potential corruption.
    3. Removes corrupted ID3 tags.
    4. Re-adds clean metadata from the CSV.
    5. Removes entries from CSV and deletes audio files if fixing fails.
    
    Parameters:
    - csv_path: Path to 'filtered_dataset.csv'.
    - base_dir: Base directory where MP3 files are located.
    - reencode: Boolean flag to re-encode MP3 files using ffmpeg.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV file: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}")
        sys.exit(1)

    required_audio_col = 'audio_previews'
    if required_audio_col not in df.columns:
        logger.error(f"CSV file must contain '{required_audio_col}' column.")
        sys.exit(1)

    total_files = len(df)
    logger.info(f"Total files to process: {total_files}")

    # To keep track of indices to drop
    indices_to_drop = []

    # Iterate through each row in the CSV with a counter
    for count, (index, row) in enumerate(df.iterrows(), 1):
        audio_path = row['audio_previews']
        # Construct absolute path
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base_dir, audio_path)

        logger.info(f"Processing file {count} of {total_files}: {audio_path}")

        if not os.path.isfile(audio_path):
            logger.warning(f"File does not exist: {audio_path}")
            # Optionally, remove the entry if file doesn't exist
            indices_to_drop.append(index)
            continue

        # Check if the MP3 file has corrupted ID3 tags
        if is_id3_tag_corrupted(audio_path):
            logger.info(f"Corruption detected in: {audio_path}")

            # Optional: Re-encode the MP3 file to fix potential audio corruption
            if reencode:
                success = reencode_mp3(audio_path)
                if not success:
                    logger.error(f"Re-encoding failed for: {audio_path}")
                    # Remove entry and delete file
                    indices_to_drop.append(index)
                    try:
                        os.remove(audio_path)
                        logger.info(f"Deleted corrupted file: {audio_path}")
                    except Exception as del_err:
                        logger.error(f"Failed to delete {audio_path}: {del_err}")
                    continue  # Skip further processing for this file

            # After re-encoding, check again if it's still corrupted
            if is_id3_tag_corrupted(audio_path):
                logger.error(f"Corruption still exists after re-encoding: {audio_path}")
                # Remove entry and delete file
                indices_to_drop.append(index)
                try:
                    os.remove(audio_path)
                    logger.info(f"Deleted corrupted file: {audio_path}")
                except Exception as del_err:
                    logger.error(f"Failed to delete {audio_path}: {del_err}")
                continue  # Skip further processing for this file

            # Now remove corrupted ID3 tags
            remove_id3_tags(audio_path)

            # Prepare metadata dictionary
            metadata = {}
            for col in METADATA_COLS:
                if col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        metadata[col] = value
                    else:
                        metadata[col] = None

            # Add clean ID3 tags if metadata is available
            if any(metadata.values()):
                add_id3_tags(audio_path, metadata)
            else:
                logger.info(f"No metadata to add for: {audio_path}")
        else:
            logger.info(f"No corruption detected in: {audio_path}")

    # Remove problematic entries from the DataFrame
    if indices_to_drop:
        df = df.drop(indices_to_drop)
        logger.info(f"Removed {len(indices_to_drop)} problematic entries from CSV.")

    # Save the updated DataFrame back to the CSV
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Updated CSV file saved: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save updated CSV file {csv_path}: {e}")

def main():
    """
    Main function to execute the script.
    """
    fix_corrupted_mp3s(CSV_PATH, BASE_DIR, reencode=REENCODE)

if __name__ == "__main__":
    main()
