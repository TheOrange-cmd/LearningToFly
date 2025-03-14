import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

def rotate_image_with_mask(image, angle):
    """
    Rotate image with padding and return both rotated image and mask
    """
    height, width = image.shape[:2]
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Create padded image
    padding_color = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    square_image = np.full((diagonal, diagonal, 3), padding_color, dtype=np.uint8)
    
    # Create binary mask (1 for real image, 0 for padding)
    mask = np.zeros((diagonal, diagonal), dtype=np.float32)
    
    # Place the original image and update mask
    start_y = (diagonal - height) // 2
    start_x = (diagonal - width) // 2
    square_image[start_y:start_y+height, start_x:start_x+width] = image
    mask[start_y:start_y+height, start_x:start_x+width] = 1.0
    
    # Rotate both image and mask
    M = cv2.getRotationMatrix2D((diagonal/2, diagonal/2), angle, 1.0)
    rotated_image = cv2.warpAffine(square_image, M, (diagonal, diagonal), 
                                  borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask, M, (diagonal, diagonal),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Crop back to original size
    final_start_y = (diagonal - height) // 2
    final_start_x = (diagonal - width) // 2
    final_image = rotated_image[final_start_y:final_start_y+height, 
                               final_start_x:final_start_x+width]
    final_mask = rotated_mask[final_start_y:final_start_y+height,
                             final_start_x:final_start_x+width]
    
    return final_image, final_mask

def preprocess_image(img):
    """
    Preprocess image for the model.
    """
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure float32 and normalize to [0,1]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
        
    # Transpose to (C, H, W)
    img = img.transpose(2, 0, 1)
    
    return img

def adjust_brightness_contrast(image, brightness, contrast):
    """
    Adjust brightness and contrast of an image.
    """
    # Ensure float32
    image = image.astype(np.float32)
    
    # Convert to [0,1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply contrast first
    mean = np.mean(image)
    image = (1 + contrast) * (image - mean) + mean
    
    # Then brightness
    image = image + brightness
    
    # Clip to [0,1]
    image = np.clip(image, 0, 1)
    
    # Convert back to [0,255] for OpenCV operations
    image = (image * 255).astype(np.uint8)
    
    return image

class BorderDataset(Dataset):
    def __init__(self, images, masks, labels):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Add debugging in __init__
        unique, counts = np.unique(labels, return_counts=True)
        # print(f"Dataset initialized with label distribution: {dict(zip(unique, counts))}")
        
        # Check first few samples
        # print("First 5 labels:", labels[:5])
        # print("Last 5 labels:", labels[-5:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].copy())
        mask = torch.from_numpy(self.masks[idx].copy())
        img = self.normalize(img)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        # Add occasional debug print
        # if idx % 500 == 0:
        #     print(f"\nDataset access - idx: {idx}")
        #     print(f"Label at idx {idx}: {self.labels[idx]}")
        #     print(f"Surrounding labels: {self.labels[max(0,idx-2):min(len(self.labels),idx+3)]}")
        
        return img, mask, label


def prepare_dataset(base_dir, num_rotations=9, augment_brightness_contrast=True):
    print("Loading and preprocessing dataset...")
    
    boundaries_dir = os.path.join(base_dir, 'boundaries')
    confirmed_floor_dir = os.path.join(base_dir, 'confirmed_floor')
    unlabeled_dir = os.path.join(base_dir, 'unlabeled')
    
    images = []
    masks = []
    labels = []
    sources = []  # Track where each image comes from
    
    # Process images from boundaries directory
    for img_path in tqdm(os.listdir(boundaries_dir), desc="Processing boundaries"):
        img = cv2.imread(os.path.join(boundaries_dir, img_path))
        if img is None:
            continue
            
        # Add original image
        orig_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        images.append(preprocess_image(img))
        masks.append(orig_mask)
        labels.append(1)  # Let's verify what this should be
        sources.append('boundaries')
        
        # Add augmented versions
        for _ in range(num_rotations):
            augmented = img.copy()
            angle = np.random.uniform(0, 360)
            rotated_img, rotated_mask = rotate_image_with_mask(augmented, angle)
            
            if augment_brightness_contrast:
                brightness = np.random.uniform(-0.3, 0.3)
                contrast = np.random.uniform(-0.3, 0.3)
                rotated_img = adjust_brightness_contrast(rotated_img, brightness, contrast)
            
            images.append(preprocess_image(rotated_img))
            masks.append(rotated_mask)
            labels.append(1)  # Let's verify what this should be
            sources.append('boundaries')
    
    # Process confirmed floor images
    for img_path in tqdm(os.listdir(confirmed_floor_dir), desc="Processing confirmed_floor"):
        img = cv2.imread(os.path.join(confirmed_floor_dir, img_path))
        if img is None:
            continue
        
        orig_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        images.append(preprocess_image(img))
        masks.append(orig_mask)
        labels.append(0)  # Let's verify what this should be
        sources.append('confirmed_floor')
    
    # Process unlabeled images
    for img_path in tqdm(os.listdir(unlabeled_dir), desc="Processing unlabeled"):
        img = cv2.imread(os.path.join(unlabeled_dir, img_path))
        if img is None:
            continue
        
        orig_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        images.append(preprocess_image(img))
        masks.append(orig_mask)
        labels.append(0)  # Let's verify what this should be
        sources.append('unlabeled')

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    labels = np.array(labels)
    sources = np.array(sources)

    unique, counts = np.unique(labels, return_counts=True)

    print("Label distribution:", dict(zip(*np.unique(labels, return_counts=True))))
    
    # Shuffle all arrays together before splitting
    shuffle_idx = np.random.permutation(len(labels))
    images = images[shuffle_idx]
    masks = masks[shuffle_idx]
    labels = labels[shuffle_idx]
    sources = sources[shuffle_idx]
    
    # Print shuffled label distribution in chunks
    # for i in range(0, len(labels), 100):
    #     chunk = labels[i:i+100]
    #     print(f"Labels in chunk {i}-{i+100}: {np.unique(chunk)}")
    
    # print("Label range:", labels.min(), labels.max())
    # print("Dataset size:", len(images), "images")

    return images, masks, labels, sources
