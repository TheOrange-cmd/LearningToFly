import torch
import torch.nn as nn
import torch.nn.functional as F

class BorderDetector(nn.Module):
    # def __init__(self, conv_channels=[4, 8, 8], dropout_rate=0.1):
    #     super(BorderDetector, self).__init__()
    #     # print("\nModel architecture:")
    #     # print(f"Input size: 240x240")
    #     # print(f"After initial pool: 120x120")
        
    #     # Initial pooling
    #     self.initial_pool = nn.MaxPool2d(2, 2)
        
    #     # First conv block (3 -> first channel size)
    #     self.conv1 = nn.Conv2d(3, conv_channels[0], kernel_size=3, stride=2, padding=1)
    #     # print(f"Layer 0: 120x120 -> 60x60, channels: {conv_channels[0]}")
        
    #     # Second conv block
    #     self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1)
    #     # print(f"Layer 1: 60x60 -> 30x30, channels: {conv_channels[1]}")
        
    #     # Third conv block
    #     self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1)
    #     # print(f"Layer 2: 30x30 -> 15x15, channels: {conv_channels[2]}")
        
    #     # ReLU and Dropout
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout2d(dropout_rate)
        
    #     # Calculate flattened size
    #     with torch.no_grad():
    #         x = torch.zeros(1, 3, 240, 240)
    #         x = self.initial_pool(x)
    #         x = self.conv1(x)
    #         x = self.conv2(x)
    #         x = self.conv3(x)
    #         self.flat_size = x.numel()
    #         # print(f"\nFlattened feature size: {self.flat_size}")
        
    #     # Classifier
    #     self.classifier = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(self.flat_size, 1),
    #         nn.Sigmoid()
    #     )
        
    #     # Initialize weights
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)

    # def forward(self, x, mask=None):
    #     # Basic forward pass without any NaN checking
    #     x = self.initial_pool(x)
        
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
        
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
        
    #     x = self.conv3(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
        
    #     x = self.classifier(x)
        
    #     return x
    
    def __init__(self, conv_channels=[4, 8, 8], dropout_rate=0.1):
        super(BorderDetector, self).__init__()
        
        # Initial pooling
        self.initial_pool = nn.MaxPool2d(2, 2)
        
        # First conv block with instance norm
        self.conv1 = nn.Conv2d(3, conv_channels[0], kernel_size=3, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(conv_channels[0], affine=True)
        
        # Second conv block with instance norm
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(conv_channels[1], affine=True)
        
        # Third conv block with instance norm
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(conv_channels[2], affine=True)
        
        # ReLU and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.zeros(1, 3, 240, 240)
            x = self.initial_pool(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flat_size = x.numel()
        
        # Classifier with layer normalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 32),
            nn.LayerNorm(32),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        x = self.initial_pool(x)
        
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.classifier(x)
        
        return x
    
    def check_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                weight_norm = param.data.norm()
                print(f"{name}:")
                print(f"  Weight norm: {weight_norm:.3f}")
                print(f"  Gradient norm: {grad_norm:.3f}")
                print(f"  Ratio: {grad_norm/weight_norm:.3f}")