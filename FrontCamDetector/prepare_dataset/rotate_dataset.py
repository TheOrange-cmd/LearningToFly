'''
This script processes our aggregrated dataset from the CyberZoo project.
It flattens the directory structure, rotates images, and saves them in a new directory, to be processed by MiDaS in another script.
Author: Daniel Rugge (2025) created with help of Claude Sonnet 3.5
'''

from pathlib import Path
from PIL import Image
import re
from collections import defaultdict
import shutil

def is_sequence_folder(folder_name):
    """Identify folders with YYYYMMDD-HHMMSS pattern"""
    return re.fullmatch(r"\d{8}-\d{6}", folder_name)

def flatten_and_process(root_dir, output_dir):
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # Track sequence folders to handle duplicates
    sequence_counts = defaultdict(int)
    
    # Walk through all directories
    for path in root_path.glob('**/*'):
        if path.is_dir() and is_sequence_folder(path.name):
            # Get all image files in sequence folder
            images = sorted([f for f in path.glob('*') 
                           if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
            
            if not images:
                continue
                
            # Create output folder (handle name conflicts)
            base_name = path.name
            while True:
                output_seq = output_path / f"{base_name}_{sequence_counts[base_name]}" if sequence_counts[base_name] else output_path / base_name
                if not output_seq.exists():
                    output_seq.mkdir(parents=True, exist_ok=True)
                    break
                sequence_counts[base_name] += 1
                
            print(f"Processing {path} => {output_seq.name}")
            
            # Process images
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        rotated = img.rotate(90, expand=True)
                        rotated.save(output_seq / img_path.name)
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {str(e)}")
                    
            sequence_counts[base_name] += 1
    
# Create output directory
output_path = Path("dataset_rotated")
output_path.mkdir(parents=True, exist_ok=True)

# Process all sequences
flatten_and_process(Path("datasets\cyberzoo"), output_path)
print("\nProcessing complete! New structure:")
print(f"Total sequences: {len(list(output_path.glob('*')))}")