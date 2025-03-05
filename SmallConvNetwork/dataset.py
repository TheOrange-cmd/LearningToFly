import torchvision.io
import glob
import torch
from torch.utils.data import Dataset
from utils import *


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, width, height, grid_lines, danger_levels, device="cpu"):
        self.image_paths = sorted(glob.glob(image_dir + "/*.jpg"))
        self.label_paths = sorted(glob.glob(image_dir.replace("images" , "labels") + "/*.txt"))
        # print(f"[CustomImageDataset] Found {len(self.image_paths)} images and {len(self.label_paths)} labels")
        self.width = width
        self.height = height
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.device = device

    def __len__(self):
        return len(self.image_paths)
    
    def get_bboxes(self, path):
        bboxes = open(path, "r").readlines()
        bboxes = [{"label": int(bbox.split(" ")[0]),
                   "x": int(float(bbox.split(" ")[1])*self.width),
                   "y": int(float(bbox.split(" ")[2])*self.height),
                   "w": int(float(bbox.split(" ")[3])*self.width),
                   "h": int(float(bbox.split(" ")[4])*self.height)} for bbox in bboxes]
        return bboxes

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx]).float().to(self.device) / 255
        bboxes = self.get_bboxes(self.label_paths[idx])
        cell_danger_levels = torch.tensor(
            cell_contains_bbox(bboxes, self.grid_lines, self.width, self.height, self.danger_levels),
            device=self.device
        ).float()
        return image, cell_danger_levels