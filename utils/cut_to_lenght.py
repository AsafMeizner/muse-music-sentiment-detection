import os
import subprocess
import concurrent.futures
import logging
from functools import partial

# Configuration
AUDIO_DIR = "audio_previews"            # Directory containing audio files
OUTPUT_DIR = "audio_previews_fixed"     # Temporary directory for fixed files
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
LOG_FILE = "fix_metadata.log"

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def fix_metadata(file_path, output_dir):
    """
    Fixes metadata by stripping existing metadata using ffmpeg.
    Saves the fixed file to the output directory and replaces the original.
    """
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Re-encode the file without copying metadata
        # Using '-c copy' to avoid re-encoding; if issues persist, remove '-c copy' to force re-encoding
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', file_path,
            '-map_metadata', '-1',  # Strip metadata
            '-c:a', 'copy',         # Copy audio codec (no re-encoding)
            output_path
        ], check=True)
        
        # Atomically replace the original file with the fixed file
        os.replace(output_path, file_path)
        logging.info(f"Fixed metadata: {filename}")
        return f"Fixed metadata: {filename}"
        
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed for {filename}: {e.stderr.decode() if e.stderr else 'No error message'}")
        return f"ffmpeg failed for {filename}: {e.stderr.decode() if e.stderr else 'No error message'}"
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
    
    total_files = len(audio_files)
    print(f"Found {total_files} audio files in '{audio_dir}'.")
    
    if total_files == 0:
        print("No audio files found to process.")
        return
    
    print("Starting metadata fixing with multiprocessing...")
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Partial function to fix metadata with fixed output directory
        fix_func = partial(fix_metadata, output_dir=output_dir)
        
        # Submit all tasks
        futures = {executor.submit(fix_func, file_path): file_path for file_path in audio_files}
        
        # Process the results as they complete
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                filename = os.path.basename(file_path)
                logging.error(f"Unhandled exception for {filename}: {e}")
                print(f"Unhandled exception for {filename}: {e}")
    
    print("Metadata fixing completed. Check 'fix_metadata.log' for details.")

if __name__ == "__main__":
    main()
