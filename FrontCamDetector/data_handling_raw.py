import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# from label_generator import danger_level
import h5py  
import matplotlib.pyplot as plt
from pathlib import Path

class DangerDataset(Dataset):
    def __init__(self, h5_path, downscale_factor=2, training=True, debug=False):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.training = training
        
        with h5py.File(h5_path, 'r') as f:
            self.preprocessed_images = f['preprocessed_images'][:]  # Shape: (N, 3, 240, 240)
            self.danger_values = f['danger_values'][:]  # Shape: (N, 5)
        
        self.valid_indices = list(range(len(self.preprocessed_images)))
        if debug:
            self.valid_indices = self.valid_indices[:100]
        
        self.calculate_label_statistics()
    
    def calculate_label_statistics(self):
        """Calculate statistics of danger values for potential weighting"""
        values = self.danger_values[self.valid_indices]
        self.label_mean = np.mean(values)
        self.label_std = np.std(values)
        
        # Calculate histogram for weighting
        hist, _ = np.histogram(values.flatten(), bins=10, range=(0, 1))
        self.label_weights = 1.0 / (hist + 1)
        self.label_weights = self.label_weights / np.sum(self.label_weights)
        
        print("Label statistics:")
        print(f"Mean: {self.label_mean:.3f}")
        print(f"Std: {self.label_std:.3f}")
        print(f"Training with {len(self)} samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        yuv = self.preprocessed_images[real_idx]  # Shape: (3, 240, 240)
        yuv = yuv.transpose(1, 2, 0).astype(np.uint8)  # Convert to HWC for OpenCV
        
        # Data augmentation
        if self.training:
            if np.random.random() < 0.5:
                angle = np.random.choice([5, 85, 95, 175, 185, 265, 275, 355])
                yuv = rotate_image(yuv.transpose(2, 0, 1), angle).transpose(1, 2, 0)
            
            if np.random.random() < 0.5:
                brightness = np.random.uniform(-30, 30)
                contrast = np.random.uniform(-0.3, 0.3)
                yuv = adjust_brightness_contrast(yuv.transpose(2, 0, 1), brightness, contrast).transpose(1, 2, 0)
        
        # Downscale
        if self.downscale_factor > 1:
            h, w = yuv.shape[:2]
            new_size = (w // self.downscale_factor, h // self.downscale_factor)
            yuv = cv2.resize(yuv, new_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        yuv = yuv.transpose(2, 0, 1).astype(np.float32)
        yuv[0] /= 255.0  # Y channel
        yuv[1:] = yuv[1:] / 255.0 - 0.5  # U/V
        
        return (
            torch.from_numpy(yuv).float(),
            torch.tensor(self.danger_values[real_idx], dtype=torch.float32)
        )
    
    def get_temporal_pair(self, idx, window=1):
        """
        Get temporally adjacent frames for time series validation
        
        Args:
            idx: Index of current frame
            window: Number of frames to include before/after
        
        Returns:
            list: List of (image, label) pairs for temporal sequence
        """
        real_idx = self.valid_indices[idx]
        base_path = Path(self.image_paths[real_idx])
        
        # Try to find adjacent frames
        sequence = []
        current_frame_num = int(base_path.stem.split('_')[-1])
        
        for offset in range(-window, window + 1):
            target_frame = current_frame_num + offset
            target_path = base_path.parent / f"{base_path.stem[:-len(str(current_frame_num))]}{target_frame}{base_path.suffix}"
            
            if target_path.exists() and str(target_path) in self.image_paths:
                target_idx = self.image_paths.index(str(target_path))
                if target_idx in self.valid_indices:
                    sequence.append(self.__getitem__(self.valid_indices.index(target_idx)))
        
        return sequence
    
    def visualize_processing(self, idx, num_augmented=3, save_path=None):
        """
        Visualize original and augmented images with their YUV channels and danger values
        
        Args:
            idx: Index of image to visualize
            num_augmented: Number of augmented versions to show
            save_path: If provided, save visualization to this path
        """
        real_idx = self.valid_indices[idx]
        
        # Load original image
        orig_img = cv2.imread(self.image_paths[real_idx])
        orig_rotated = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
        
        # Get center crop
        h, w = orig_rotated.shape[:2]
        y_start = (h - 240) // 2
        x_start = (w - 240) // 2
        orig_cropped = orig_rotated[y_start:y_start+240, x_start:x_start+240]
        
        # Create figure
        num_cols = 4  # Original + YUV channels
        num_rows = num_augmented + 1  # Original + augmented versions
        plt.figure(figsize=(15, 4*num_rows))
        
        # Function to plot single image set (original/augmented + YUV channels)
        def plot_image_set(img_bgr, row, title):
            # Convert BGR to YUV
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            
            # Plot BGR image
            plt.subplot(num_rows, num_cols, row * num_cols + 1)
            plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            plt.title(f'{title} (RGB)')
            plt.axis('off')
            
            # Plot YUV channels
            channel_names = ['Y channel', 'U channel', 'V channel']
            for i in range(3):
                plt.subplot(num_rows, num_cols, row * num_cols + i + 2)
                plt.imshow(img_yuv[:, :, i], cmap='gray')
                plt.title(f'{title} ({channel_names[i]})')
                plt.axis('off')
        
        # Plot original image and its channels
        plot_image_set(orig_cropped, 0, 'Original')
        
        # Get danger values
        danger_values = self.danger_values[real_idx]
        
        # Plot danger values on the first row
        plt.figtext(0.02, 0.98 - 0/num_rows, 
                    f'Danger values: {", ".join([f"{x:.2f}" for x in danger_values])}',
                    fontsize=10)
        
        # Generate and plot augmented versions
        for i in range(num_augmented):
            # Temporarily set training mode to True to enable augmentation
            orig_training = self.training
            self.training = True
            
            # Get augmented image
            aug_img, _ = self.__getitem__(idx)
            
            # Convert tensor back to BGR for visualization
            aug_img = aug_img.numpy().transpose(1, 2, 0)  # CHW -> HWC
            # Denormalize
            aug_img[..., 0] = aug_img[..., 0] * 255.0  # Y channel
            aug_img[..., 1:] = (aug_img[..., 1:] + 0.5) * 255.0  # UV channels
            aug_img = aug_img.astype(np.uint8)
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_YUV2BGR)
            
            # Plot augmented image and its channels
            plot_image_set(aug_img_bgr, i+1, f'Augmented {i+1}')
            
            # Reset training mode
            self.training = orig_training
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_batch(self, batch_size=4, save_path=None):
        """
        Visualize a batch of images and their labels
        
        Args:
            batch_size: Number of images to visualize
            save_path: If provided, save visualization to this path
        """
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        plt.figure(figsize=(15, 4*batch_size))
        for i, idx in enumerate(indices):
            img, label = self[idx]
            
            # Convert tensor to numpy and denormalize
            img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
            img[..., 0] = img[..., 0] * 255.0  # Y channel
            img[..., 1:] = (img[..., 1:] + 0.5) * 255.0  # UV channels
            img = img.astype(np.uint8)
            
            # Convert YUV to BGR to RGB for display
            img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            plt.subplot(batch_size, 1, i+1)
            plt.imshow(img_rgb)
            plt.title(f'Danger values: {", ".join([f"{x:.2f}" for x in label.numpy()])}')
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

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

def prepare_datasets_for_configs(configs, h5_path, debug=False):
    """Create dataset configurations for each unique set of parameters"""
    dataset_configs = {}
    
    for config in configs:
        # Create key based on parameters that affect dataset generation
        dataset_key = (
            config['downscale_factor'],
            config['augment_brightness_contrast']  # Determines if training mode is enabled
        )
        
        if dataset_key not in dataset_configs:
            print(f"\nPreparing dataset with parameters:")
            print(f"- downscale_factor: {config['downscale_factor']}")
            print(f"- training_mode: {config['augment_brightness_contrast']}")
            
            # Create the dataset instance
            dataset = DangerDataset(
                h5_path=h5_path,
                downscale_factor=config['downscale_factor'],
                training=config['augment_brightness_contrast'],
                debug=debug
            )
            
            # Store the dataset instance
            dataset_configs[dataset_key] = dataset
    
    return dataset_configs

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