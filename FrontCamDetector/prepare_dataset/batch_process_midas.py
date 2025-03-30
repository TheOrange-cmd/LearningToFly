'''
This script generates depth maps for a dataset of images using the MiDaS model.
It processes each sequence in the dataset, running the MiDaS model on the images and saving the resulting depth maps in a specified output directory. The script handles multiple sequences.
Author: Daniel Rugge (2025) created with help of Claude Sonnet 3.5
'''


import subprocess
from pathlib import Path
import argparse

def process_depth_maps(input_root, output_root, model_type):
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Find all sequence folders
    sequence_folders = [f for f in input_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(sequence_folders)} sequences to process")
    
    for seq_folder in sequence_folders:
        # Create corresponding output folder
        relative_path = seq_folder.relative_to(input_path)
        output_seq_path = output_path / relative_path
        output_seq_path.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "python", 
            "run.py",
            "--model_type", model_type,
            "--input_path", str(seq_folder),
            "--output_path", str(output_seq_path)
        ]
        
        print(f"\nProcessing: {seq_folder.name}")
        print(" ".join(cmd))
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {seq_folder.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {seq_folder.name}: {e}")
        except Exception as e:
            print(f"Unexpected error with {seq_folder.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process depth maps')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to rotated_dataset folder')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to depth_dataset output folder')
    parser.add_argument('--model', type=str, default="dpt_beit_large_512",
                      help='MiDaS model type (default: dpt_beit_large_512)')
    
    args = parser.parse_args()
    
    # Create output root if it doesn't exist
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    process_depth_maps(args.input, args.output, args.model)
    print("\nProcessing complete! Check output folder for results.")