import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def reshape_uyvy_to_yuv(uyvy_data, downscale_factor=2):
    """
    Reshape 120x120 UYVY data into YUV channels with optional downscaling
    Input: raw UYVY data (120x240 uint8 array, since each pixel needs 2 bytes)
    Output: (3, output_height, output_width) float32 array with separate Y, U, V channels, normalized to [0,1]
    """
    if downscale_factor not in [2, 4]:
        raise ValueError("Downscale factor must be 2 or 4")
        
    height, width = 120, 120
    out_height = height // downscale_factor
    out_width = width // downscale_factor
    
    # Reshape input to 2D array where each row represents pixels in UYVY format
    uyvy = uyvy_data.reshape(height, -1)
    
    # Initialize output arrays
    y = np.zeros((out_height, out_width), dtype=np.float32)
    u = np.zeros((out_height, out_width), dtype=np.float32)
    v = np.zeros((out_height, out_width), dtype=np.float32)
    
    # Process blocks
    for out_y in range(out_height):
        for out_x in range(out_width):
            y_sum = 0
            u_sum = 0
            v_sum = 0
            count = 0
            
            # Process each pixel in the block
            for dy in range(downscale_factor):
                in_y = out_y * downscale_factor + dy
                for dx in range(downscale_factor):
                    in_x = out_x * downscale_factor + dx
                    
                    # Calculate indices in UYVY format
                    uyvy_idx = in_x * 2
                    
                    # Extract YUV values
                    if dx % 2 == 0:  # Even pixels
                        y_val = uyvy[in_y, uyvy_idx + 1]
                        u_val = uyvy[in_y, uyvy_idx]
                        v_val = uyvy[in_y, uyvy_idx + 2]
                    else:  # Odd pixels
                        y_val = uyvy[in_y, uyvy_idx + 1]
                        u_val = uyvy[in_y, uyvy_idx - 2]
                        v_val = uyvy[in_y, uyvy_idx]
                    
                    y_sum += y_val
                    u_sum += u_val
                    v_sum += v_val
                    count += 1
            
            # Average and normalize
            y[out_y, out_x] = (y_sum / count) / 255.0
            u[out_y, out_x] = (u_sum / count) / 255.0 - 0.5
            v[out_y, out_x] = (v_sum / count) / 255.0 - 0.5
    
    # Stack channels
    yuv = np.stack([y, u, v])
    
    return yuv

def rotate_image(image, angle):
    """
    Rotate YUV image with border replication
    image shape: (3, 120, 120) float32 array with values in [0,1]
    """
    # Transpose image to (120, 120, 3) for OpenCV rotation
    image = image.transpose(1, 2, 0)
    height, width = image.shape[:2]
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Create padded image
    padding_color = np.median(image.reshape(-1, 3), axis=0).astype(np.float32)
    square_image = np.full((diagonal, diagonal, 3), padding_color, dtype=np.float32)
    
    # Place the original image
    start_y = (diagonal - height) // 2
    start_x = (diagonal - width) // 2
    square_image[start_y:start_y+height, start_x:start_x+width] = image
    
    # Rotate image
    M = cv2.getRotationMatrix2D((diagonal/2, diagonal/2), angle, 1.0)
    rotated_image = cv2.warpAffine(square_image, M, (diagonal, diagonal), 
                                  borderMode=cv2.BORDER_REPLICATE)
    
    # Crop back to original size
    final_start_y = (diagonal - height) // 2
    final_start_x = (diagonal - width) // 2
    final_image = rotated_image[final_start_y:final_start_y+height, 
                               final_start_x:final_start_x+width]
    
    # Transpose back to (3, 120, 120)
    final_image = final_image.transpose(2, 0, 1)
    
    return final_image

def adjust_brightness_contrast(image, brightness, contrast):
    """
    Adjust brightness and contrast for float32 image
    image shape: (3, 120, 120) with values in [0,1]
    """
    # Only adjust Y channel (luminance)
    y_channel = image[0].copy()
    
    # Apply contrast
    contrast_factor = (1 + contrast)
    y_channel = (y_channel - 0.5) * contrast_factor + 0.5
    
    # Apply brightness (scaled to [-0.5, 0.5] range for float32)
    brightness_factor = brightness / 255.0
    y_channel = y_channel + brightness_factor
    
    # Clip values to [0,1]
    y_channel = np.clip(y_channel, 0, 1)
    
    # Update Y channel
    image[0] = y_channel
    
    return image

def visualize_rotations(image, angles=[15, 30, 45, 60, 75, 90]):
    """
    Takes a single UYVY image and shows the results of different rotations
    Returns: Original and rotated images, with padding percentages
    """
    results = []
    yuv = reshape_uyvy_to_yuv(image)
    
    # First add original
    results.append({
        'angle': 0,
        'image': yuv,
        'padding_percent': 0
    })
    
    for angle in angles:
        rotated_img = rotate_image(yuv, angle)
        
        results.append({
            'angle': angle,
            'image': rotated_img,
        })
    
    return results

# Test function
def test_rotations():
    # Load a single image from your dataset
    boundaries_dir = 'path/to/boundaries'
    test_image_path = os.path.join(boundaries_dir, os.listdir(boundaries_dir)[0])
    
    with open(test_image_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(120, 240)
    
    results = visualize_rotations(raw_data)
    
    # Print statistics and display images
    for result in results:
        print(f"Rotation: {result['angle']}°")
        print(f"Padding: {result['padding_percent']:.1f}%")
        print("---")

def prepare_datasets_for_configs(configs, data_dir):
    # Create a dictionary to store unique dataset configurations
    dataset_configs = {}
    
    for config in configs:
        # Create a key based on dataset-specific parameters
        dataset_key = (
            config['num_rotations'],
            config['augment_brightness_contrast'],
            config['downscale_factor']
        )
        
        # Only prepare dataset if we haven't seen these parameters before
        if dataset_key not in dataset_configs:
            print(f"\nPreparing dataset with parameters:")
            print(f"- num_rotations: {config['num_rotations']}")
            print(f"- augment_brightness_contrast: {config['augment_brightness_contrast']}")
            print(f"- downscale_factor: {config['downscale_factor']}")
            
            images, labels, sources = prepare_dataset(
                data_dir,
                num_rotations=config['num_rotations'],
                augment_brightness_contrast=config['augment_brightness_contrast'],
                downscale_factor=config['downscale_factor']
            )
            
            dataset_configs[dataset_key] = (images, labels, sources)
    
    return dataset_configs

def prepare_dataset(base_dir, num_rotations=2, augment_brightness_contrast=True, downscale_factor=2, debug=False):
    print("Loading and preprocessing dataset...")
    
    # Define allowed rotation angles (5 degrees around multiples of 90)
    rotation_angles = [5, 85, 95, 175, 185, 265, 275, 355]
    
    boundaries_dir = os.path.join(base_dir, 'boundaries')
    confirmed_floor_dir = os.path.join(base_dir, 'confirmed_floor')
    unlabeled_dir = os.path.join(base_dir, 'unlabeled')
    
    images = []
    labels = []
    sources = []
    
    if debug:
        end = 10
    else:
        end = None
    
    # Process images from boundaries directory
    for img_path in tqdm(os.listdir(boundaries_dir)[0:end], desc="Processing boundaries"):
        with open(os.path.join(boundaries_dir, img_path), 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(120, 240)
        
        img = reshape_uyvy_to_yuv(raw_data, downscale_factor=downscale_factor)
        
        # Add original image
        images.append(img)
        labels.append(1)
        sources.append('boundaries')
        
        # Pick num_rotations random angles and augment brightness/contrast
        for angle in np.random.choice(rotation_angles, num_rotations, replace=False):
            augmented = img.copy()
            rotated_img = rotate_image(augmented, angle)
            
            if augment_brightness_contrast:
                brightness = np.random.uniform(-30, 30)  # Still use pixel values for easier understanding
                contrast = np.random.uniform(-0.3, 0.3)
                rotated_img = adjust_brightness_contrast(rotated_img, brightness, contrast)
            
            images.append(rotated_img)
            labels.append(1)
            sources.append('boundaries')
    
    # Process confirmed_floor and unlabeled
    for directory, label in [(confirmed_floor_dir, 0), (unlabeled_dir, 0)]:
        for img_path in tqdm(os.listdir(directory)[0:end], desc=f"Processing {os.path.basename(directory)}"):
            with open(os.path.join(directory, img_path), 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(120, 240)
            
            img = reshape_uyvy_to_yuv(raw_data, downscale_factor=downscale_factor)
            images.append(img)
            labels.append(label)
            sources.append(os.path.basename(directory))

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    sources = np.array(sources)

    print("Label distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    
    # Shuffle all arrays together
    shuffle_idx = np.random.permutation(len(labels))
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]
    sources = sources[shuffle_idx]

    return np.array(images), np.array(labels), np.array(sources)

class BorderDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].copy())  # Already float32 [0,1]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label

# class BorderDataset(Dataset):
#     def __init__(self, images, labels):
#         self.images = images  
#         self.labels = labels.squeeze() if isinstance(labels, np.ndarray) else labels
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img = torch.from_numpy(self.images[idx].copy())
#         label = torch.tensor([self.labels[idx]], dtype=torch.float32)
#         #  reshape to [1,1] tensor
#         label = label.view(1, 1)
#         print(f"Dataset output shapes - img: {img.shape}, label: {label.shape}")
#         return img, label