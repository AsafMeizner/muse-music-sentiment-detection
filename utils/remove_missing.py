import os
import sys
import pandas as pd
import argparse
import shutil
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove rows from CSV where audio files are missing.")
    parser.add_argument(
        '--csv_path',
        type=str,
        default='filtered_dataset.csv',
        help='Path to the input CSV file.'
    )
    parser.add_argument(
        '--audio_base_dir',
        type=str,
        default='.',
        help='Base directory where audio files are stored. Can be absolute or relative.'
    )
    parser.add_argument(
        '--output_csv_path',
        type=str,
        default=None,
        help='Path to save the cleaned CSV file. If not specified, overwrites the original CSV.'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='If set, creates a backup of the original CSV before overwriting.'
    )
    return parser.parse_args()

def create_backup(csv_path):
    backup_path = csv_path + '.backup'
    try:
        shutil.copy(csv_path, backup_path)
        print(f"Backup created at: {backup_path}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    CSV_PATH = args.csv_path
    AUDIO_BASE_DIR = args.audio_base_dir
    OUTPUT_CSV_PATH = args.output_csv_path if args.output_csv_path else CSV_PATH
    BACKUP = args.backup

    # Verify CSV file exists
    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file does not exist at path: {CSV_PATH}")
        sys.exit(1)
    
    # Read the CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded CSV file with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if 'audio_previews' column exists
    if 'audio_previews' not in df.columns:
        print("Error: 'audio_previews' column not found in the CSV.")
        sys.exit(1)
    
    # Initialize counters
    total_files = len(df)
    missing_files = 0
    missing_indices = []
    missing_entries = []

    # Iterate over the DataFrame and check file existence
    print("Checking for missing audio files...")
    for index, row in df.iterrows():
        audio_path = row['audio_previews']
        # Construct absolute path if needed
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(AUDIO_BASE_DIR, audio_path)
        
        # Normalize path for the current OS
        audio_path = os.path.normpath(audio_path)
        
        if not os.path.isfile(audio_path):
            missing_files += 1
            missing_indices.append(index)
            missing_entries.append(row.to_dict())
            print(f"Missing file: {audio_path} (Row {index})")
    
    print(f"\nTotal files checked: {total_files}")
    print(f"Missing files found: {missing_files}")

    if missing_files == 0:
        print("No missing audio files detected. No changes made to the CSV.")
        sys.exit(0)
    
    # Optionally, create a backup before modifying the CSV
    if BACKUP and OUTPUT_CSV_PATH == CSV_PATH:
        print("Creating a backup of the original CSV...")
        create_backup(CSV_PATH)
    
    # Remove the rows with missing files
    cleaned_df = df.drop(index=missing_indices)
    removed_percentage = (missing_files / total_files) * 100
    print(f"Removed {missing_files} rows ({removed_percentage:.2f}%) from the CSV.")

    # Save the cleaned DataFrame to CSV
    try:
        cleaned_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Cleaned CSV saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving cleaned CSV: {e}")
        sys.exit(1)
    
    # Optionally, save details of removed entries
    removed_entries_path = Path(OUTPUT_CSV_PATH).with_suffix('.removed_entries.csv')
    try:
        removed_df = pd.DataFrame(missing_entries)
        removed_df.to_csv(removed_entries_path, index=False)
        print(f"Details of removed entries saved to: {removed_entries_path}")
    except Exception as e:
        print(f"Error saving removed entries: {e}")
        # Not exiting since main task is done

if __name__ == "__main__":
    main()
