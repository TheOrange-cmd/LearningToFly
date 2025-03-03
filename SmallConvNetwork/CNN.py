import torchvision.io
import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, width, height):
        self.image_paths = sorted(glob.glob(image_dir + "/*.jpg"))
        self.label_paths = sorted(glob.glob(image_dir.replace("images" , "labels") + "/*.txt"))
        self.width = width
        self.height = height

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
        image = torchvision.io.read_image(self.image_paths[idx]).float().to("cuda") / 255
        bboxes = self.get_bboxes(self.label_paths[idx])
        cell_danger_levels = torch.tensor(cell_contains_bbox(bboxes, grid_lines)).float()
        return image, cell_danger_levels

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines, danger_levels):
        super(ObjectDetectionModel, self).__init__()
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.model = torch.nn.Sequential(
            # input shape: 3x240x520
            torch.nn.MaxPool2d(3),

            # input shape: 3x80x173
            torch.nn.Conv2d(3, 16, 3, padding=1, stride=2),
            torch.nn.ReLU(),

            # input shape: 16x40x87
            torch.nn.Conv2d(16, 64, 3, padding=1, stride=2),
            torch.nn.ReLU(),

            # input shape: 64x20x44
            torch.nn.Conv2d(64, 256, 3, padding=1),
            torch.nn.ReLU(),

            # input shape: 256x20x44
            torch.nn.Conv2d(256, 64, 1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # input shape: 64x10x22
            torch.nn.Flatten(),

            # input shape: 64x10x22
            torch.nn.LazyLinear(1024),
            torch.nn.ReLU(),

            # input shape: 1024
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),

            # input shape: 256
            torch.nn.LazyLinear(danger_levels * (len(grid_lines[0]) - 1) * (len(grid_lines[1]) - 1)),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (danger_levels, len(grid_lines[0]) - 1, len(grid_lines[1]) - 1))
        )
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, image_dir_train, image_dir_val, width, height, batch_size=8):
        super(ObjectDetectionDataModule, self).__init__()
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val
        self.width = width
        self.height = height
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomImageDataset(self.image_dir_train, self.width, self.height)
        self.val_dataset = CustomImageDataset(self.image_dir_val, self.width, self.height)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

def convert_coordinates(bbox):
    # Convert from x, y, w, h to x1, y1, x2, y2
    x1 = bbox["x"] - bbox["w"]/2 if bbox["x"] - bbox["w"]/2 > 0 else 0
    y1 = bbox["y"] - bbox["h"]/2 if bbox["y"] - bbox["h"]/2 > 0 else 0
    x2 = bbox["x"] + bbox["w"]/2 if bbox["x"] + bbox["w"]/2 < width else width
    y2 = bbox["y"] + bbox["h"]/2 if bbox["y"] + bbox["h"]/2 < height else height
    return x1, x2, y1, y2

def cell_contains_bbox(bboxes, grid_lines):
    box_inside_grid = [[[0 for _ in range(len(grid_lines[1])-1)] for _ in range(len(grid_lines[0])-1)] for _ in range(danger_levels)]
    for bbox in bboxes:
        x1, x2, y1, y2 = convert_coordinates(bbox)
        for i in range(len(grid_lines[0])-1):
            left = grid_lines[0][i]
            right = grid_lines[0][i+1]
            if ((left <= x1 <= right) or (left <= x2 <= right)) or (x1 <= left and x2 >= right):
                for j in range(len(grid_lines[1])-1):
                    bottom = grid_lines[1][j]
                    top = grid_lines[1][j+1]
                    if (bottom <= y1 <= top) or (bottom <= y2 <= top):
                        danger = danger_level(x1, x2, y1, y2, bbox["label"])
                        box_inside_grid[0][i][j] = 1 # Safe
                        if danger >= 0.6:
                            box_inside_grid[1][i][j] = 1 # Warning
                        else:
                            continue
                        if 0.7 <= danger <= 1:
                            box_inside_grid[2][i][j] = 1 # Danger
    return box_inside_grid

def danger_level(x1, x2, y1, y2, label):
    x1_rel = x1 / width
    y1_rel = y1 / height
    x2_rel = x2 / width
    y2_rel = y2 / height

    if x1_rel >= 0.8:
        side_distance = (x1_rel - 0.6) * 2
    elif x1_rel >= 0.6:
        side_distance = (x1_rel - 0.6) / 2
    elif x2_rel <= 0.6:
        side_distance = (0.6 - x2_rel) / 2
    elif x2_rel <= 0.2:
        side_distance = (0.6 - x2_rel) * 2
    else:
        side_distance = 0

    if label == 0:
        width_distance = (x2_rel - x1_rel) / 0.25
        height_distance = min((y2_rel - y1_rel) / 0.8, 0.7)
        if side_distance > 0.25:
            multiplier = 2
        else:
            multiplier = 1
        danger = min(max([width_distance, height_distance]) - side_distance * multiplier, 1)
    if label == 1:
        width_distance = (x2_rel - x1_rel) / 0.15
        height_distance = min((y2_rel - y1_rel) / 1.1, 0.7)
        if side_distance > 0.25:
            multiplier = 2
        else:
            multiplier = 1
        danger = min(max([width_distance, height_distance]) - side_distance * multiplier, 1)
    return danger


def plot_image(sample_image, sample_danger, truth, grid_lines, grid=False):
    # Display the sample image.
    plt.imshow(sample_image.squeeze(0).permute(1, 2, 0))
    # Draw grid lines if enabled.
    if grid:
        for grid_line_y in grid_lines[1]:
            plt.axhline(grid_line_y, color="blue", linestyle="--")
        for grid_line_x in grid_lines[0]:
            plt.axvline(grid_line_x, color="blue", linestyle="--")

    # Overlay predicted danger levels.
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            # Prediction overlays (filled cells)
            if sample_danger[2][i][j] > 0.8:
                plt.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                                 [grid_lines[1][j], grid_lines[1][j]],
                                 [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                                 color="red", alpha=0.5)
            elif sample_danger[1][i][j] > 0.8:
                plt.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                                 [grid_lines[1][j], grid_lines[1][j]],
                                 [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                                 color="orange", alpha=0.5)
            elif sample_danger[0][i][j] > 0.8:
                plt.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                                 [grid_lines[1][j], grid_lines[1][j]],
                                 [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                                 color="green", alpha=0.5)

    ax = plt.gca()
    if truth is not None:
        for i in range(len(grid_lines[0]) - 1):
            for j in range(len(grid_lines[1]) - 1):
                if (truth[2][i][j] == 1):
                    if not sample_danger[2][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="red", facecolor="none", hatch="///"))
                elif (truth[1][i][j] == 1):
                    if not sample_danger[1][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="orange", facecolor="none", hatch="///"))
                elif (truth[0][i][j] == 1):
                    if not sample_danger[0][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="green", facecolor="none", hatch="///"))
                elif (truth[0][i][j] == 0) and (sample_danger[0][i][j] > 0.8) or (sample_danger[1][i][j] > 0.8) or (sample_danger[2][i][j] > 0.8):
                    ax.add_patch(plt.Rectangle(
                        (grid_lines[0][i], grid_lines[1][j]),
                        grid_lines[0][i + 1] - grid_lines[0][i],
                        grid_lines[1][j + 1] - grid_lines[1][j],
                        linewidth=3, edgecolor="white", facecolor="none", hatch="///"))


    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.show()

image_dir_train = "./SmallConvNetwork/dataset_syn/images/train"
image_dir_val = "./SmallConvNetwork/dataset_syn/images/val"

width = 520
height = 240

grid_lines = [[0,0.2, 0.4,0.6, 0.8,1],[0,1]]

grid_lines[0] = [line*width for line in grid_lines[0]]
grid_lines[1] = [line*height for line in grid_lines[1]]

danger_levels = 3

train = False

data_module = ObjectDetectionDataModule(image_dir_train,image_dir_val, width, height)
model = ObjectDetectionModel(grid_lines, danger_levels)

if train:
    trainer = pl.Trainer(max_epochs=10, check_val_every_n_epoch=1, log_every_n_steps=10)
    trainer.fit(model, data_module)

#model.eval()
model.to("cpu")
torch.set_num_threads(1)
plot_set = CustomImageDataset(image_dir_val,width,height)

image_paths = sorted(glob.glob('./SmallConvNetwork/Test_video/*.jpg'))

fig, ax = plt.subplots()

def update(frame):
    ax.clear()  # clear the axes for the new frame
    image = torchvision.io.read_image(image_paths[frame]).float() / 255
    start_frame = time.time()
    sample_danger = model(image.unsqueeze(0)).squeeze(0)
    print(f"{1 / (time.time() - start_frame):.1f} fps")

    # Call the modified plot_image function that accepts an axis as parameter
    # Instead of calling plt.show() inside plot_image, pass 'ax' so that the figure updates.
    plot_image(sample_image=image, sample_danger=sample_danger, truth=None, grid_lines=grid_lines, grid=False)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return ax

# Create the figure and axis
fig, ax = plt.subplots()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(image_paths), interval=100)

plt.show()






