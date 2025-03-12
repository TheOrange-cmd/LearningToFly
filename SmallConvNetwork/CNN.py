import torchvision.io
import glob
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
                   "x": int(float(bbox.split(" ")[1])*self.width),
                   "y": int(float(bbox.split(" ")[2])*self.height),
                   "w": int(float(bbox.split(" ")[3])*self.width),
                   "h": int(float(bbox.split(" ")[4])*self.height)} for bbox in bboxes]
        return bboxes

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx]).float().cuda()
        if image.shape[-2:] != (self.width, self.height):
            raise ValueError(f"Image shape {image.shape} of {self.image_paths[idx]} is not equal to the expected shape ({self.width}, {self.height})")
        bboxes = self.get_bboxes(self.label_paths[idx])
        cell_danger_levels = torch.tensor(cell_contains_bbox(bboxes, self.grid_lines, self.width, self.height)).float()
        print(cell_danger_levels)
        return image, cell_danger_levels

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines):
        super(ObjectDetectionModel, self).__init__()
        self.grid_lines = grid_lines

        self.conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(4),

            # input shape: 3x130x60
            torch.nn.Conv2d(3, 16, 3, padding=1, stride=2),
            torch.nn.ReLU(),

            # input shape: 16x65x30
            torch.nn.Conv2d(16, 64, 3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            # input shape: 64x33x15
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            torch.nn.ReLU(),

            # input shape: 128x17x8
            torch.nn.Conv2d(128, 32, 1, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            #output shape: 32x8x4
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32*8*4, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128,len(grid_lines) - 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = x.view(x.size(0), len(self.grid_lines) - 1)
        return x

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
    def __init__(self, image_dir_train, image_dir_val, width, height, grid_lines, batch_size=16):
        super(ObjectDetectionDataModule, self).__init__()
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.grid_lines = grid_lines

    def setup(self, stage=None):
        self.train_dataset = CustomImageDataset(self.image_dir_train, self.width, self.height, self.grid_lines)
        self.val_dataset = CustomImageDataset(self.image_dir_val, self.width, self.height, self.grid_lines)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

def convert_coordinates(bbox, width, height):
    # Convert from x, y, w, h to x1, y1, x2, y2
    x1 = bbox["x"] - bbox["w"]/2 if bbox["x"] - bbox["w"]/2 > 0 else 0
    y1 = bbox["y"] - bbox["h"]/2 if bbox["y"] - bbox["h"]/2 > 0 else 0
    x2 = bbox["x"] + bbox["w"]/2 if bbox["x"] + bbox["w"]/2 < width else width
    y2 = bbox["y"] + bbox["h"]/2 if bbox["y"] + bbox["h"]/2 < height else height
    return x1, x2, y1, y2

def cell_contains_bbox(bboxes, grid_lines, width, height):
    danger_list = [0 for _ in range(len(grid_lines)-1)]
    for bbox in bboxes:
        x1, x2, y1, y2 = convert_coordinates(bbox, width, height)
        box_danger = danger_level(x1, x2, y1, y2, bbox["label"], width, height)
        for i in range(len(grid_lines)-1):
            left = grid_lines[i]
            right = grid_lines[i+1]
            if ((left <= x1 <= right) or (left <= x2 <= right)) or (x1 <= left and x2 >= right):
                danger_list[i] = max(danger_list[i], box_danger)
    return danger_list

def danger_level(x1, x2, y1, y2, label, width, height, min_danger=0.0):
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
    return max(min_danger, danger)

def plot_image(sample_image, sample_danger, grid_lines, grid=False):
    colors = [(0, "green"), (0.5, "orange"), (1, "red")]
    cmap = LinearSegmentedColormap.from_list("traffic_light", colors)

    sample_danger = sample_danger.detach().numpy()

    # Rotate image to display it correctly. Divide by 255 to scale pixel values to [0, 1].
    plt.imshow(torch.rot90(sample_image.squeeze(0), k=1, dims=[1,2]).permute(1, 2, 0)/255)

    # Overlay predicted danger levels.
    for i in range(len(grid_lines) - 1):
        plt.fill_between([grid_lines[i], grid_lines[i + 1]],
                         [0, 0],
                         [height, height],
                         color=cmap(sample_danger[i]), alpha=0.5)
        plt.text((grid_lines[i] +grid_lines[i + 1])/2 , height/2, f"{sample_danger[i]:.2f}", ha='center', va='center')


    plt.xlim(0, width)
    plt.ylim(height, 0)

if __name__ == "__main__":
    image_dir_train = "./SmallConvNetwork/dataset/images/train"
    image_dir_val = "./SmallConvNetwork/dataset/images/val"

    # these are the dimensions of the image when it is rotated by 90 degrees, so it is displayed correctly
    width = 520
    height = 240

    grid_lines = [0,0.2, 0.4,0.6, 0.8,1]
    grid_lines = [line*width for line in grid_lines]

    train = True
    save_model = False
    save_video = False

    data_module = ObjectDetectionDataModule(image_dir_train,image_dir_val, width, height,grid_lines)
    model = ObjectDetectionModel(grid_lines)

    if train:
        trainer = pl.Trainer(max_epochs=10, check_val_every_n_epoch=1, log_every_n_steps=10)
        trainer.fit(model, data_module)
    else:
        model = ObjectDetectionModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=9-step=160.ckpt", grid_lines=grid_lines)

    if save_model:
        model.to_onnx("./SmallConvNetwork/model_rgb.onnx", torch.randn(1, 3, width, height))

    # plot a video to test the predictions
    model.to("cpu")
    torch.set_num_threads(1)
    plot_set = CustomImageDataset(image_dir_val,width,height, grid_lines)

    image_paths = sorted(glob.glob('./SmallConvNetwork/Test_video/*.jpg'))

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()  # clear the axes for the new frame
        image = torchvision.io.read_image(image_paths[frame]).float()
        start_frame = time.time()
        sample_danger = model(image.unsqueeze(0)).squeeze(0)
        print(f"{1 / (time.time() - start_frame):.1f} fps")

        # Call the plot_image function to update the plot
        plot_image(sample_image=image, sample_danger=sample_danger, grid_lines=grid_lines, grid=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        return ax

    ani = animation.FuncAnimation(fig, update, frames=len(image_paths), interval=100)

    if save_video:
        # Save the animation to an MP4 file using ffmpeg writer
        ani.save('./SmallConvNetwork/output.mp4', writer='ffmpeg', fps=10)

    plt.show()






