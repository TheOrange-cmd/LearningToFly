from torch.utils.data import Dataset
import torchvision.io
import glob
import matplotlib.pyplot as plt
import torch
from CNN import cell_contains_bbox, convert_coordinates
import albumentations as A
import time


width = 520
height = 240

grid_lines = [0,0.2, 0.4,0.6, 0.8,1]
grid_lines = [line*width for line in grid_lines]

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, width, height, grid_lines):
        self.image_paths = sorted(glob.glob(image_dir + "/*.jpg"))
        self.label_paths = sorted(glob.glob(image_dir.replace("images" , "labels") + "/*.txt"))
        self.width = width
        self.height = height
        self.grid_lines = grid_lines

    def __len__(self):
        return len(self.image_paths)

    def get_bboxes(self, path):
        bboxes = open(path, "r").readlines()
        bboxes = [{"label": int(bbox.split(" ")[0]),
                   "x": float(bbox.split(" ")[1]),
                   "y": float(bbox.split(" ")[2]),
                   "w": float(bbox.split(" ")[3]),
                   "h": float(bbox.split(" ")[4])} for bbox in bboxes]
        return bboxes

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx]).float().cuda()
        if image.shape[-2:] != (self.width, self.height):
            raise ValueError(f"Image shape {image.shape} of {self.image_paths[idx]} is not equal to the expected shape ({self.width}, {self.height})")
        bboxes = self.get_bboxes(self.label_paths[idx])
        cell_danger_levels = torch.tensor(cell_contains_bbox(bboxes, self.grid_lines, self.width, self.height)).float()
        return image, cell_danger_levels, bboxes


dataset = CustomImageDataset("./SmallConvNetwork/dataset/images/train", width, height, grid_lines)

sample = 50

transform = A.Compose(
    [
        A.Affine(p=1, scale=(1.3,1.5), translate_percent=(-0.1, 0.1), shear=(-2, 2)),
    ],
    bbox_params=A.BboxParams(format='yolo'))

for sample in range(len(dataset)):
    # Get the original image and bounding boxes
    original_image = torch.rot90(dataset[sample][0], k=1, dims=[1,2]).cpu().permute(1, 2, 0).numpy()/255
    bboxes = [[bbox["x"], bbox["y"], bbox["w"], bbox["h"], str(bbox["label"])] for bbox in dataset[sample][2]]

    # Apply transformation
    start = time.time()
    transformed = transform(image=original_image, bboxes=bboxes)
    transformed_image = transformed['image']
    print(f"transform took {time.time()-start} seconds")
    transformed_bboxes = transformed['bboxes']

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot original image and bounding boxes
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        x = x*width
        y = y*height
        w = w*width
        h = h*height
        x1 = x - w/2
        x2 = x + w/2
        y1 = y - h/2
        y2 = y + h/2
        ax1.plot([x1, x1, x2, x2, x1], [y2, y1, y1, y2, y2], color='red')

    # Plot transformed image and bounding boxes
    ax2.imshow(transformed_image)
    ax2.set_title("Transformed Image")
    for bbox in transformed_bboxes:
        x, y, w, h = bbox[:4]
        x = x*width
        y = y*height
        w = w*width
        h = h*height
        x1 = x - w/2
        x2 = x + w/2
        y1 = y - h/2
        y2 = y + h/2
        ax2.plot([x1, x1, x2, x2, x1], [y2, y1, y1, y2, y2], color='red')
    ax2.set_ylim([height, 0])
    ax2.set_xlim([0, width])
    ax1.set_ylim([height, 0])
    ax1.set_xlim([0, width])
    plt.tight_layout()
    plt.show()