import os
import sys
from pydub import AudioSegment
import concurrent.futures

MAX_DURATION_MS = 10 * 60 * 1000  # 20 minutes
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")

def process_audio_file(file_path):
    """
    Load an audio file, trim it if longer than 20 minutes,
    and overwrite the original file.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Skipping (unsupported extension): {file_path}"

    try:
        # Load audio
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        return f"Error loading {file_path}: {e}"

    # Trim if necessary
    if len(audio) > MAX_DURATION_MS:
        trimmed_audio = audio[:MAX_DURATION_MS]
        try:
            trimmed_audio.export(file_path, format=ext.replace(".", ""))
            return f"Trimmed: {os.path.basename(file_path)}"
        except Exception as e:
            return f"Error exporting {file_path}: {e}"
    else:
        return f"No change (under 20 min): {os.path.basename(file_path)}"

def main():
    audio_dir = "audio_previews"
    
    # Collect audio files (full paths)
    audio_files = []
    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        if os.path.isfile(file_path):
            # Check if it's a supported extension before adding
            if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(file_path)

    # Sort by file size (descending), so largest files get processed first
    audio_files.sort(key=lambda f: os.path.getsize(f), reverse=True)

    print(f"Found {len(audio_files)} audio files in '{audio_dir}'.")
    print("Starting parallel processing...")

    # max_workers defaults to a sensible value based on your CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # This will submit tasks in descending file-size order
        results = executor.map(process_audio_file, audio_files)

        # Process the results in the same order we submitted them
        for result in results:
            print(result)

if __name__ == "__main__":
    main()
