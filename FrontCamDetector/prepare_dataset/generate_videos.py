'''
This script generates side-by-side comparison videos of RGB images and their corresponding depth maps. It processes each sequence in the dataset, ensuring that the images and depth maps are aligned and resized correctly. The resulting videos are saved in a specified output directory.
Author: Daniel Rugge (2025) created with help of Claude Sonnet 3.5
'''

import cv2
import numpy as np
from pathlib import Path
import imageio

def create_comparison_video(image_dir, depth_dir, output_path, fps=30):
    # Get sorted list of frames
    image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".png"]])
    depth_files = sorted([f for f in depth_dir.glob("*") if f.suffix.lower() in [".png", ".jpg"]])
    
    # Verify matching files
    assert len(image_files) == len(depth_files), "Mismatched frame counts"
    
    # Create video writer
    writer = imageio.get_writer(str(output_path), fps=fps, macro_block_size=None)
    
    for img_path, depth_path in zip(image_files, depth_files):
        # Load images
        img = cv2.imread(str(img_path))
        depth = cv2.imread(str(depth_path))
        
        # Resize depth to match image dimensions
        if img.shape != depth.shape:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
            
        # Create side-by-side view
        combined = np.hstack([img, depth])
        
        # Add frame info overlay
        text = f"Frame: {img_path.stem}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Convert color space for imageio
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(combined)
    
    writer.close()

def process_all_sequences(rotated_dataset, depth_dataset, output_dir):
    for seq_folder in rotated_dataset.iterdir():
        if seq_folder.is_dir():
            depth_folder = depth_dataset / seq_folder.name
            if depth_folder.exists():
                output_path = output_dir / f"{seq_folder.name}_comparison.mp4"
                create_comparison_video(seq_folder, depth_folder, output_path)
                print(f"Created: {output_path.name}")

if __name__ == "__main__":
    rotated_dataset = Path("C:\\Users\\danie\\Documents\\GitHub\\Autonomous_Flight\\dataset_rotated")
    depth_dataset = Path("C:\\Users\\danie\\Documents\\GitHub\\Autonomous_Flight\\dataset_depth")
    output_dir = Path("C:\\Users\\danie\\Documents\\GitHub\\Autonomous_Flight\\comparison_videos")
    output_dir.mkdir(exist_ok=True)
    
    process_all_sequences(rotated_dataset, depth_dataset, output_dir)