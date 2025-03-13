import glob
import numpy as np
import torchvision.io
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


def read_bboxes(path):
    bboxes = open(path, "r").readlines()
    bboxes_processed = np.zeros((len(bboxes), 5))

    for i, bbox in enumerate(bboxes):
        bboxes_split = bbox.split(" ")
        x = float(bboxes_split[1])
        y = float(bboxes_split[2])
        w = float(bboxes_split[3])
        h = float(bboxes_split[4])
        label = int(bboxes_split[0])
        bboxes_processed[i,:] = [x, y, w, h, label]

    return bboxes_processed

def convert_coordinates(bbox):
    # Convert from x, y, w, h to x1, y1, x2, y2
    x1 = bbox[0] - bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    x2 = bbox[0] + bbox[2]/2
    y2 = bbox[1] + bbox[3]/2
    label = int(bbox[4])
    return x1, x2, y1, y2, label

def danger_level(x1, x2, y1, y2, label, min_danger=0.0):
    # 0 : Panel
    # 1 : Plant
    # 2 : Pole
    # 3 : blocks

    rel_height = y2 - y1
    rel_width = x2 - x1

    # Panel
    if label == 0:
        width_distance = rel_width / 0.25
        height_distance = rel_height / 0.9
        danger = min(max([width_distance, height_distance]), 1)

    # Plant
    if label == 1: # Plant
        width_distance = rel_width / 0.2
        height_distance = rel_height / 0.6
        danger = min(max([width_distance, height_distance]), 1)

    # Pole or blocks
    if (label == 2) or (label == 3):
        width_distance = rel_width / 0.15
        height_distance = min(rel_height / 1.1, 0.7)
        danger = min(max([width_distance, height_distance]), 1)

    return max(min_danger, danger)

def generate_danger_level_list(bboxes, grid_lines, width, height):
    danger_list = [0 for _ in range(len(grid_lines)-1)]

    # loop over all bboxes
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, x2, y1, y2, label = convert_coordinates(bbox)

        # calculate how dangerous given box is
        box_danger = danger_level(x1, x2, y1, y2, label)
        for j in range(len(grid_lines)-1):
            left = grid_lines[j]
            right = grid_lines[j+1]
            # update the danger_list iff the box is in the given column and danger value is greater than current for that column
            if ((left <= x1 <= right) or (left <= x2 <= right)) or (x1 <= left and x2 >= right):
                danger_list[j] = max(danger_list[j], box_danger)
    return danger_list

if __name__ == "__main__":
    # Test the danger_level function
    test_images_dir = "./SmallConvNetwork/dataset/images/val/*.jpg"
    for image_path in sorted(glob.glob(test_images_dir)):
        print(f"plotting image: {image_path}")
        colors = [(0, "green"), (0.5, "orange"), (1.0, "red")]
        cmap = LinearSegmentedColormap.from_list("traffic_light", colors)

        # Rotate image to display it correctly. Divide by 255 to scale pixel values to [0, 1].
        plt.imshow(torch.rot90(torchvision.io.read_image(image_path), k=1, dims=[1,2]).permute(1, 2, 0)/255)
        ax = plt.gca()
        labels = read_bboxes(image_path.replace("images","labels").replace(".jpg", ".txt"))
        for i in range(labels.shape[0]):
            bbox = labels[i]
            x1, x2, y1, y2, label = convert_coordinates(bbox)
            danger = danger_level(x1, x2, y1, y2, label)
            print(bbox, danger)
            rect = patches.Rectangle((x1*520, y1*240), (x2-x1)*520, (y2-y1)*240, facecolor=cmap(danger-0.001), linewidth=2, alpha=0.5)
            ax.add_patch(rect)
            plt.text((x1+x2)*260, (y1+y2)*120, f"{danger:.2f}", ha='center', va='center')
        plt.xlim(0, 520)
        plt.ylim(240, 0)
        plt.show()
