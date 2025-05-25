import os
from pathlib import Path
from PIL import Image
import shutil

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def clean_directory(directory):
    deleted_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = Path(root) / file
            if not is_valid_image(filepath):
                print(f"Deleting corrupt file: {filepath}")
                os.remove(filepath)
                deleted_files += 1
    return deleted_files

if __name__ == "__main__":
    data_dirs = ["data/Train_Data", "data/Test_Data"]
    total_deleted = 0
    
    for dir in data_dirs:
        print(f"Cleaning {dir}...")
        deleted = clean_directory(dir)
        total_deleted += deleted
        print(f"Deleted {deleted} corrupt files from {dir}")
    
    print(f"\nTotal deleted files: {total_deleted}")