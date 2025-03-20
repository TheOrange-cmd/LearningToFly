import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from label_generator import danger_level
import h5py  

class DangerDataset(Dataset):
    def __init__(self, h5_path, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        
        with h5py.File(h5_path, 'r') as f:
            self.image_paths = [x.decode() for x in f['image_paths']]
            self.labels = f['danger_values'][:]
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load raw UYVY image (520x480 bytes)
        with open(self.image_paths[idx], 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(520, 480)
        
        # Crop center 240 rows (original height 520)
        cropped = raw_data[140:380, :]  # 240x480
        
        # Convert to YUV and downscale
        yuv = reshape_uyvy_to_yuv(cropped, self.downscale_factor)
        
        return (
            torch.from_numpy(yuv.copy()).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

def reshape_uyvy_to_yuv(uyvy_data, downscale_factor=2):
    """Process 240x480 UYVY input (240x240 image)"""
    height, width = 240, 240  # After cropping
    out_h = height // downscale_factor
    out_w = width // downscale_factor
    
    y = np.zeros((out_h, out_w), np.float32)
    u = np.zeros_like(y)
    v = np.zeros_like(y)
    
    for oy in range(out_h):
        for ox in range(out_w):
            # Process downscale_factor x downscale_factor block
            y_sum = u_sum = v_sum = 0
            for dy in range(downscale_factor):
                iy = oy * downscale_factor + dy
                for dx in range(downscale_factor):
                    ix = ox * downscale_factor + dx
                    byte_pos = ix * 2
                    y_val = uyvy_data[iy, byte_pos + 1]
                    uv_byte = uyvy_data[iy, byte_pos - (dx % 2)*2]
                    u_val = uv_byte[0]
                    v_val = uv_byte[2] if dx % 2 == 0 else uv_byte[0]
                    
                    y_sum += y_val
                    u_sum += u_val
                    v_sum += v_val
            
            y[oy, ox] = y_sum / (downscale_factor**2) / 255
            u[oy, ox] = u_sum / (downscale_factor**2) / 255 - 0.5
            v[oy, ox] = v_sum / (downscale_factor**2) / 255 - 0.5
    
    return np.stack([y, u, v])

def rotate_image(image, angle):
    """
    Rotate YUV image with border replication.
    
    Args:
        image: (3, H, W) float32 array with Y, U, V channels.
        angle: Rotation angle in degrees.
    
    Returns:
        rotated_image: (3, H, W) float32 array with rotated YUV channels.
    """
    # Transpose image to (H, W, 3) for OpenCV rotation
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
    
    # Transpose back to (3, H, W)
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
        
        # Generate regression targets using danger_level function
        # Assuming you have a function to generate 5 targets
        regression_targets = generate_regression_targets(img)  # Implement this function
        
        # Add original image
        images.append(img)
        labels.append(regression_targets)
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
            labels.append(regression_targets)  # Use the same targets for augmented images
            sources.append('boundaries')
    
    # Process confirmed_floor and unlabeled
    for directory, label in [(confirmed_floor_dir, 0), (unlabeled_dir, 0)]:
        for img_path in tqdm(os.listdir(directory)[0:end], desc=f"Processing {os.path.basename(directory)}"):
            with open(os.path.join(directory, img_path), 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(120, 240)
            
            img = reshape_uyvy_to_yuv(raw_data, downscale_factor=downscale_factor)
            regression_targets = generate_regression_targets(img)  # Implement this function
            images.append(img)
            labels.append(regression_targets)
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

def generate_regression_targets(image):
    # Assuming image is of shape (3, height, width)
    height, width = image.shape[1], image.shape[2]
    
    # Split the image into 5 columns
    column_width = width // 5
    targets = []
    
    for i in range(5):
        x1 = i * column_width
        x2 = (i + 1) * column_width
        y1 = 0
        y2 = height
        
        # Calculate danger level for this column
        danger = danger_level(x1, x2, y1, y2, label=1, width=width, height=height)
        targets.append(danger)
    
    return np.array(targets)

class BorderDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  
        self.labels = labels 
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].copy())  # Already float32 [0,1]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Shape (5,)
        return img, label