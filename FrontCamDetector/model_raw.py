import torch
import torch.nn as nn
import torch.nn.functional as F

class ObstacleDetector(nn.Module):
    def __init__(self, config):
        super(ObstacleDetector, self).__init__()
        self.debug_mode = False
        
        # Extract config values
        channels = config['conv_channels']
        fc_size = config['fc_size']
        dropout_rate = config['dropout_rate']
        self.use_efficient = config.get('use_efficient', False)  # New flag for efficient convolutions
        
        # Proper input size calculation
        self.input_size = 240 // config['downscale_factor']
        channels = config['conv_channels']
        
        # Calculate final feature map size
        size_after_conv = self.input_size // (2 ** 3)  # After 3 pooling layers
        self.flat_size = channels[-1] * size_after_conv * size_after_conv
        
        if self.use_efficient:
            # Depthwise separable convolutions
            self.conv1 = nn.Sequential(
                # Depthwise conv
                nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
                # Pointwise conv
                nn.Conv2d(3, channels[0], kernel_size=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, groups=channels[0]),
                nn.Conv2d(channels[0], channels[1], kernel_size=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1, groups=channels[1]),
                nn.Conv2d(channels[1], channels[2], kernel_size=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
        else:
            # Original convolution layers
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            
            self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(2)
        
        # Use more efficient FC layers with intermediate squeeze
        if self.use_efficient:
            self.classifier = nn.Sequential(
                nn.Linear(self.flat_size, fc_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                # Additional squeeze layer
                nn.Linear(fc_size, fc_size // 2),
                nn.ReLU(),
                nn.Linear(fc_size // 2, 3),  # Output 3 values
            )
        else:
            # Original FC layers
            self.fc1 = nn.Linear(self.flat_size, fc_size)
            self.relu4 = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(fc_size, 3)  # Output 3 values

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if self.debug_mode:
            print(f"Input shape: {x.shape}")
            
        if self.use_efficient:
            x = self.conv1(x)
            if self.debug_mode: print(f"After conv1: {x.shape}")
            x = self.conv2(x)
            if self.debug_mode: print(f"After conv2: {x.shape}")
            x = self.conv3(x)
            if self.debug_mode: print(f"After conv3: {x.shape}")
        else:
            x = self.pool1(self.relu1(self.conv1(x)))
            if self.debug_mode: print(f"After conv1: {x.shape}")
            x = self.pool2(self.relu2(self.conv2(x)))
            if self.debug_mode: print(f"After conv2: {x.shape}")
            x = self.pool3(self.relu3(self.conv3(x)))
            if self.debug_mode: print(f"After conv3: {x.shape}")
        
        x = x.view(-1, self.flat_size)
        
        # Add the missing classification layers
        if self.use_efficient:
            x = self.classifier(x)
        else:
            x = self.relu4(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        
        if self.debug_mode:
            print(f"Final output shape: {x.shape}")
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)