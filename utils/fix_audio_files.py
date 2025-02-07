import os
import subprocess
import concurrent.futures
import logging

# Configuration
AUDIO_DIR = "audio_previews"  # Directory containing trimmed audio files
OUTPUT_DIR = "audio_previews_fixed"  # Directory to save re-encoded files
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
LOG_FILE = "metadata_fixing.log"

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def fix_metadata(file_path, output_dir):
    """
    Re-encode the audio file using ffmpeg to fix metadata issues.
    Saves the fixed file in the output directory.
    """
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Re-encode the file without copying metadata
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', file_path,
            '-map_metadata', '-1',  # Strip metadata
            '-c:a', 'copy',  # Copy audio codec (no re-encoding)
            output_path
        ], check=True)
        
        # Optionally, replace the original file with the fixed file
        os.replace(output_path, file_path)
        logging.info(f"Fixed metadata: {filename}")
        return f"Fixed metadata: {filename}"
        
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed for {filename}: {e}")
        return f"ffmpeg failed for {filename}: {e}"
    except Exception as e:
        logging.error(f"Unexpected error for {filename}: {e}")
        return f"Unexpected error for {filename}: {e}"

def main():
    audio_dir = AUDIO_DIR
    output_dir = OUTPUT_DIR
    
    # Collect audio files with supported extensions
    audio_files = [
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, f)) and
           os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    
    print(f"Found {len(audio_files)} audio files in '{audio_dir}'.")
    print("Starting metadata fixing...")
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all tasks
        futures = {executor.submit(fix_metadata, file_path, output_dir): file_path for file_path in audio_files}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                logging.error(f"Unhandled exception for {file_path}: {e}")
                print(f"Unhandled exception for {file_path}: {e}")
    
    print("Metadata fixing completed. Check the log file for details.")

if __name__ == "__main__":
    main()
