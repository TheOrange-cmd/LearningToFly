# preprocess_images.py
import h5py
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_and_save(original_h5_path, output_h5_path):
    with h5py.File(original_h5_path, 'r') as f_orig:
        image_paths = [x.decode() for x in f_orig['image_paths']]
        danger_values = f_orig['danger_values'][:]
        num_columns = f_orig.attrs.get('num_columns', 5)
        
        with h5py.File(output_h5_path, 'w') as f_new:
            # Create datasets
            preprocessed = f_new.create_dataset(
                'preprocessed_images',
                shape=(len(image_paths), 3, 240, 240),
                dtype=np.uint8
            )
            f_new.create_dataset('danger_values', data=danger_values)
            f_new.attrs['num_columns'] = num_columns
            
            for i, path in enumerate(tqdm(image_paths)):
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Skipping invalid image {path}")
                    continue
                
                # Preprocessing steps
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                h, w = img.shape[:2]
                y_start = (h - 240) // 2
                x_start = (w - 240) // 2
                cropped = img[y_start:y_start+240, x_start:x_start+240]
                yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                yuv = yuv.transpose(2, 0, 1)  # CHW format
                
                preprocessed[i] = yuv

# Example usage
preprocess_and_save('C:\\Users\\danie\\Documents\\GitHub\\paparazzi\\FrontCamDetector\\danger_labels\\danger_values.h5', 'C:\\Users\\danie\\Documents\\GitHub\\paparazzi\\FrontCamDetector\\preprocessed_labels\\danger_values.h5')