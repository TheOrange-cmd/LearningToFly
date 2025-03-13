from Danger_Calculation import generate_danger_level_list, read_bboxes
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.io
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import time


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, width, height, grid_lines):
        self.image_paths = sorted(glob.glob(image_dir + "/*.jpg"))
        self.label_paths = sorted(glob.glob(image_dir.replace("images" , "labels") + "/*.txt"))
        self.width = width
        self.height = height
        self.grid_lines = grid_lines
        self.p = 0.2
        self.transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=self.p, brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1)),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx]).float().cuda()
        if image.shape[-2:] != (self.width, self.height):
            raise ValueError(f"Image shape {image.shape} of {self.image_paths[idx]} is not equal to the expected shape ({self.width}, {self.height})")
        image_rotated = torch.rot90(image, k=1, dims=[1,2]).cpu().permute(1, 2, 0).numpy()/255
        bboxes = read_bboxes(self.label_paths[idx])
        transformed = self.transform(image=image_rotated, bboxes=bboxes)
        transformed_image = torch.rot90(torch.from_numpy(transformed["image"]).permute(2, 0, 1) * 255.0, k=3, dims=[1,2]).float()
        transformed_labels = transformed["bboxes"]

        cell_danger_levels = torch.tensor(generate_danger_level_list(transformed_labels, self.grid_lines, self.width,
                                                                     self.height)).float()
        return transformed_image, cell_danger_levels

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

            torch.nn.Linear(128,len(grid_lines) - 1),
        )
        self.loss_fn = torch.nn.MSELoss()

    def custom_relu(self, x):
        x = torch.maximum(x, torch.tensor(0.0))
        x = torch.minimum(x, torch.tensor(1.0))
        return x

    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = x.view(x.size(0), len(self.grid_lines) - 1)
        x = self.custom_relu(x)
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
        return torch.optim.Adam(self.parameters(), lr=0.0005)

class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, image_dir_train, image_dir_val, width, height, grid_lines, batch_size=8):
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

def plot_image(sample_image, sample_danger, grid_lines, grid=False):
    colors = [(0, "green"), (0.5, "orange"), (1, "red")]
    cmap = LinearSegmentedColormap.from_list("traffic_light", colors)

    sample_danger = sample_danger.detach().numpy()

    # Rotate image to display it correctly. Divide by 255 to scale pixel values to [0, 1].
    plt.imshow(torch.rot90(sample_image.squeeze(0), k=1, dims=[1,2]).permute(1, 2, 0)/255)

    # Overlay predicted danger levels.
    for i in range(len(grid_lines) - 1):
        plt.fill_between([grid_lines[i]*width, grid_lines[i + 1]*width],
                         [0, 0],
                         [height, height],
                         color=cmap(sample_danger[i]), alpha=0.5)
        plt.text((grid_lines[i]*width +grid_lines[i + 1]*width)/2 , height/2, f"{sample_danger[i]:.2f}", ha='center', va='center')


    plt.xlim(0, width)
    plt.ylim(height, 0)

if __name__ == "__main__":
    image_dir_train = "./SmallConvNetwork/dataset/images/train"
    image_dir_val = "./SmallConvNetwork/dataset/images/val"

    # these are the dimensions of the image when it is rotated by 90 degrees, so it is displayed correctly
    width = 520
    height = 240

    # This defines how wide the columns are
    columns = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # Option for running the file
    train = True
    save_model = True
    save_video = False

    # fit the model
    if train:
        data_module = ObjectDetectionDataModule(image_dir_train, image_dir_val, width, height, columns)
        model = ObjectDetectionModel(columns)
        trainer = pl.Trainer(max_epochs=20, check_val_every_n_epoch=1, log_every_n_steps=10)
        trainer.fit(model, data_module)
    # load a trained model
    else:
        model = ObjectDetectionModel.load_from_checkpoint("lightning_logs/version_99/checkpoints/epoch=19-step=1120.ckpt", grid_lines=columns)

    # save the model to onnx
    if save_model:
        model.to_onnx("./SmallConvNetwork/model_rgb_smaller.onnx", torch.randn(1, 3, width, height))

    # plot a video to test the predictions
    model.to("cpu")
    torch.set_num_threads(1)

    # Use the images from this folder
    test_images = './SmallConvNetwork/Test_video/*.jpg'
    image_paths = sorted(glob.glob(test_images))

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()  # clear the axes for the new frame
        image = torchvision.io.read_image(image_paths[frame]).float()

        # Record frame processing time
        start_frame = time.time()
        frame_danger = model(image.unsqueeze(0)).squeeze(0)
        print(f"{1 / (time.time() - start_frame):.1f} fps")

        # Call the plot_image function to update the plot
        plot_image(sample_image=image, sample_danger=frame_danger, grid_lines=columns, grid=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        return ax

    # Make the animation
    animation_fps = 10
    ani = animation.FuncAnimation(fig, update, frames=len(image_paths), interval=1000/animation_fps)

    if save_video:
        # Save the animation to an MP4 file using ffmpeg writer
        ani.save('./SmallConvNetwork/output.mp4', writer='ffmpeg', fps=animation_fps)

    plt.show()






