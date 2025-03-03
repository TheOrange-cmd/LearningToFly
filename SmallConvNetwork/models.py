import pytorch_lightning as pl
import torch
import torch.nn as nn

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines, danger_levels, input_shape=(3, 240, 520), learning_rate=0.001):
        super().__init__()
        
        # Store grid dimensions and model parameters
        self.grid_x = len(grid_lines[0]) - 1
        self.grid_y = len(grid_lines[1]) - 1
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.output_size = self.danger_levels * self.grid_x * self.grid_y
        self.learning_rate = learning_rate
        
        # Calculate feature dimensions
        h, w = input_shape[1], input_shape[2]
        h, w = h // 3, w // 3  # MaxPool2d(3)
        h, w = (h + 2*1 - 3) // 2 + 1, (w + 2*1 - 3) // 2 + 1  # Conv2d stride 2
        h, w = (h + 2*1 - 3) // 2 + 1, (w + 2*1 - 3) // 2 + 1  # Conv2d stride 2
        h, w = h, w  # Conv2d stride 1
        h, w = h, w  # Conv2d stride 1
        h, w = h // 2, w // 2  # MaxPool2d(2)
        
        # Calculate flattened feature size
        self.flattened_features = 64 * h * w
        
        # Define the model architecture
        # Convolutional features
        self.conv_features = nn.Sequential(
            # input shape: 3x240x520
            nn.MaxPool2d(3),
            
            # input shape: 3x80x173
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            
            # input shape: 16x40x87
            nn.Conv2d(16, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            
            # input shape: 64x20x44
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            
            # input shape: 256x20x44
            nn.Conv2d(256, 64, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # input shape: 64x10x22
            nn.Flatten()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # Replace LazyLinear with standard Linear layers
            nn.Linear(self.flattened_features, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.ReLU(),
            
            nn.Linear(256, self.output_size),
            nn.Sigmoid()
        )
        
        # Define loss function
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x):
        # Convolutional part
        x = self.conv_features(x)
        
        # Fully connected part
        x = self.fc_layers(x)
        
        # Reshape instead of using Unflatten
        x = x.view(-1, self.danger_levels, self.grid_x, self.grid_y)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, image_dir_train, image_dir_val, width, height, grid_lines, danger_levels, batch_size=8, device="cpu"):
        super(ObjectDetectionDataModule, self).__init__()
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.device = device

    def setup(self, stage=None):
        from dataset import CustomImageDataset
        self.train_dataset = CustomImageDataset(self.image_dir_train, self.width, self.height, 
                                               self.grid_lines, self.danger_levels, self.device)
        self.val_dataset = CustomImageDataset(self.image_dir_val, self.width, self.height, 
                                             self.grid_lines, self.danger_levels, self.device)

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
import pytorch_lightning as pl
import torch
import torch.nn as nn

class LightweightObjectDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines, danger_levels, input_shape=(3, 240, 520), learning_rate=0.001):
        super().__init__()
        
        # Store grid dimensions and model parameters
        self.grid_x = len(grid_lines[0]) - 1
        self.grid_y = len(grid_lines[1]) - 1
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.output_size = self.danger_levels * self.grid_x * self.grid_y
        self.learning_rate = learning_rate
        
        # Define the model architecture with ONNX-compatible operations
        self.conv_features = nn.Sequential(
            # First aggressively downsample with large stride
            nn.Conv2d(3, 8, kernel_size=7, stride=4, padding=3),  # Reduce to 60x130
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce to 30x65
            
            # Second layer with moderate channels
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),  # Reduce to 15x33
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce to 7x16
            
            # Final extraction with limited parameters
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Maintain size at 7x16
            nn.ReLU(),
            
            # Replace AdaptiveAvgPool2d with standard pooling and reshape operations
            # First calculate the kernel sizes needed to get to 4x5 output 
            # Assuming input is approximately 7x16 after previous operations
            nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3)),  # This will give us approximately 4x5
        )
        
        # Calculate the flattened size based on the conv output
        # Use a dummy forward pass to calculate the exact size after conv_features
        dummy_input = torch.zeros(1, 3, input_shape[1], input_shape[2])
        with torch.no_grad():
            dummy_output = self.conv_features(dummy_input)
        
        # Calculate flattened feature size based on actual output
        self.flattened_features = dummy_output.numel() // dummy_output.shape[0]
        
        # Smaller fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size),
            nn.Sigmoid()
        )
        
        # Define loss function
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x):
        # Convolutional part
        x = self.conv_features(x)
        
        # Fully connected part
        x = self.fc_layers(x)
        
        # Reshape for output
        x = x.view(-1, self.danger_levels, self.grid_x, self.grid_y)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)