import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomImageDataset, CustomImageDatasetUYVY
from utils import DangerLevelConfig

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

class SafetyLosses:
    # Configurations for different safety preferences
    CONFIGS = {
        'conservative': {
            'class_weights': [1.0, 2.0, 4.0],
            'penalty_matrix': [
                [0.0, 4.0, 8.0],  # Safe predictions
                [1.0, 0.0, 2.0],  # Warning predictions
                [2.0, 1.0, 0.0]   # Danger predictions
            ],
            'smoothness_weight': 0.1
        },
        'balanced': {
            'class_weights': [1.0, 1.5, 2.0],
            'penalty_matrix': [
                [0.0, 2.0, 4.0],  # Penalties for predicting safe when actually warning/danger
                [1.0, 0.0, 2.0],  # Penalties for predicting warning when actually safe/danger
                [2.0, 1.0, 0.0]   # Penalties for predicting danger when actually safe/warning
            ],
            'smoothness_weight': 0.1
        },
        'smooth': {
            'class_weights': [1.0, 1.5, 2.0],
            'penalty_matrix': [
                [0.0, 1.5, 3.0],
                [0.5, 0.0, 1.5],
                [1.5, 0.5, 0.0]
            ],
            'smoothness_weight': 0.2
        },
        'balanced_plus': {
            'class_weights': [1.0, 1.75, 2.25],  # Slightly increase warning/danger weights
            'penalty_matrix': [
                [0.0, 2.0, 4.5],    # Increase penalty for safe->danger
                [1.0, 0.0, 2.0],    # Keep warning transitions balanced
                [2.0, 1.0, 0.0]     # Keep danger->warning penalty moderate
            ],
            'smoothness_weight': 0.15  # Slightly increase smoothness
        },
        'balanced_safe': {
            'class_weights': [1.0, 1.5, 2.5],  # More emphasis on danger
            'penalty_matrix': [
                [0.0, 2.0, 5.0],    # Higher penalty for safe->danger
                [1.0, 0.0, 2.0],    # Keep warning transitions
                [1.5, 1.0, 0.0]     # Lower penalty for over-predicting danger
            ],
            'smoothness_weight': 0.1
        }

    }

    @staticmethod
    def hierarchical_loss(config_name='balanced'):
        class HierarchicalLoss(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.class_weights = torch.tensor(config['class_weights'])
                self.penalty_matrix = torch.tensor(config['penalty_matrix'])
                self.smoothness_weight = config['smoothness_weight']
                
            def forward(self, pred, target):
                # Apply sigmoid to get probabilities
                pred_probs = torch.sigmoid(pred)
                
                # Reshape to [N, C]
                batch_size, num_classes, height, width = pred.shape
                pred_flat = pred_probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
                target_flat = target.permute(0, 2, 3, 1).reshape(-1, num_classes)
                
                # Basic cross-entropy with class weights
                weights = self.class_weights.to(pred.device)
                weighted_bce = F.binary_cross_entropy(
                    pred_flat, target_flat, 
                    weight=weights,
                    reduction='none'
                )
                
                # Hierarchical penalty
                pred_classes = pred_flat.argmax(dim=1)  # Get predicted class
                target_classes = target_flat.argmax(dim=1)  # Get true class
                
                penalty_matrix = self.penalty_matrix.to(pred.device)
                hierarchy_penalty = penalty_matrix[pred_classes, target_classes]
                
                # Smoothness regularization (optional with mutually exclusive classes)
                if self.smoothness_weight > 0:
                    # Consider adjacent predictions in the grid
                    pred_grid = pred_probs.permute(0, 2, 3, 1)  # [batch, height, width, classes]
                    smoothness_loss = 0.0
                    
                    # Horizontal smoothness
                    if width > 1:
                        diff_h = torch.abs(pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :])
                        smoothness_loss += diff_h.mean()
                    
                    # Vertical smoothness (if we had multiple rows)
                    if height > 1:
                        diff_v = torch.abs(pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :])
                        smoothness_loss += diff_v.mean()
                    
                    smoothness_loss *= self.smoothness_weight
                else:
                    smoothness_loss = 0.0
                
                # Combine losses
                total_loss = weighted_bce.mean() + hierarchy_penalty.mean() + smoothness_loss
                
                return total_loss
                
        return HierarchicalLoss(SafetyLosses.CONFIGS[config_name])


    
class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, image_dir_train, image_dir_val, width, height, grid_lines, danger_levels, 
                 batch_size=8, device="cpu", fold_indices=None):
        super(ObjectDetectionDataModule, self).__init__()
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels
        self.device = device
        self.fold_indices = fold_indices  # (train_indices, val_indices) for cross-validation
        self.correct_labels = self.create_label_corrector()

    @staticmethod
    def create_label_corrector():
        ''' Correct the labels to no overlap and to merge background with safe '''
        def correct_labels(labels):
            corrected = torch.zeros_like(labels)
            
            # Debug print for first batch
            if torch.is_tensor(labels) and labels.shape[0] < 10:  # Small batch for readability
                print("\nLabel Correction Debug:")
                print("Original labels shape:", labels.shape)
                print("Sample original labels:")
                print(labels[0])  # Print first sample's labels
                
                # Count overlaps in first sample
                overlaps = (labels[0].sum(dim=0) > 1).sum()
                backgrounds = (labels[0].sum(dim=0) == 0).sum()
                print(f"Number of overlapping cells: {overlaps}")
                print(f"Number of background cells: {backgrounds}")
            
            # Apply corrections
            danger_mask = labels[:, 2:3] > 0.5
            warning_mask = (labels[:, 1:2] > 0.5) & ~danger_mask
            safe_mask = ((labels[:, 0:1] > 0.5) | (labels.sum(dim=1, keepdim=True) == 0)) & ~danger_mask & ~warning_mask
            
            corrected[:, 2:3] = danger_mask.float()
            corrected[:, 1:2] = warning_mask.float()
            corrected[:, 0:1] = safe_mask.float()
            
            # Debug print corrected labels
            if torch.is_tensor(labels) and labels.shape[0] < 10:
                print("\nCorrected labels:")
                print(corrected[0])  # Print first sample's corrected labels
                
                # Verify corrections
                new_overlaps = (corrected[0].sum(dim=0) > 1).sum()
                new_backgrounds = (corrected[0].sum(dim=0) == 0).sum()
                print(f"Number of overlapping cells after correction: {new_overlaps}")
                print(f"Number of background cells after correction: {new_backgrounds}")
                
                # Verify each cell has exactly one class
                single_class = (corrected[0].sum(dim=0) == 1).all()
                print(f"All cells have exactly one class: {single_class}")
                
            return corrected
        return correct_labels

    def _get_dataset_labels(self, dataset):
        labels = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            
            if i < 3:  # Print details for first 3 samples
                print(f"\nSample {i} details:")
                print("Target shape:", target.shape)
                print("Target values:")
                print(target)
                print("Class sums:", target.sum(dim=(1, 2)))
                
                # Print number of cells with each class
                for class_idx, class_name in enumerate(['safe', 'warning', 'dangerous']):
                    cells_with_class = (target[class_idx] > 0.5).sum().item()
                    print(f"Number of {class_name} cells:", cells_with_class)
            
            # Get the most severe class present (if any cell has dangerous > 0.5, that's the label)
            has_dangerous = (target[2] > 0.5).any()
            has_warning = (target[1] > 0.5).any()
            has_safe = (target[0] > 0.5).any()
            
            if has_dangerous:
                label = 2
            elif has_warning:
                label = 1
            else:
                label = 0
                
            labels.append(label)
        
        # Print overall distribution
        label_counts = torch.bincount(torch.tensor(labels))
        print("\nOverall label distribution:")
        print("Safe:", label_counts[0].item())
        print("Warning:", label_counts[1].item() if len(label_counts) > 1 else 0)
        print("Dangerous:", label_counts[2].item() if len(label_counts) > 2 else 0)
        
        return labels

    def _create_sampler(self, labels):
        from torch.utils.data import WeightedRandomSampler
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = [class_weights[label] for label in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )

    def setup(self, stage=None):
        from dataset import CustomImageDataset
        # Create full dataset
        full_dataset = CustomImageDataset(
            self.image_dir_train, 
            self.width, 
            self.height, 
            self.grid_lines, 
            self.danger_levels, 
            self.device
        )
        
        if self.fold_indices is not None:
            # Use fold indices for cross-validation
            train_indices, val_indices = self.fold_indices
            
            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)
        else:
            self.train_dataset = full_dataset
            self.val_dataset = CustomImageDataset(
                self.image_dir_val, 
                self.width, 
                self.height, 
                self.grid_lines, 
                self.danger_levels, 
                self.device
            )
        
        # Get labels for stratified sampling
        self.train_labels = self._get_dataset_labels(self.train_dataset)
        self.val_labels = self._get_dataset_labels(self.val_dataset)

        # Debug print class distribution
        train_class_counts = torch.bincount(torch.tensor(self.train_labels))
        val_class_counts = torch.bincount(torch.tensor(self.val_labels))
        
        print("\nClass distribution in datasets:")
        print("Train:", {i: count.item() for i, count in enumerate(train_class_counts)})
        print("Val:", {i: count.item() for i, count in enumerate(val_class_counts)})

    def train_dataloader(self):
        sampler = self._create_sampler(self.train_labels)
        loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            sampler=sampler
        )
        return CorrectedDataLoader(loader, self.correct_labels)
    
    def val_dataloader(self):
        sampler = self._create_sampler(self.val_labels)
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            sampler=sampler
        )
        return CorrectedDataLoader(loader, self.correct_labels)
    
class CorrectedDataLoader:
    def __init__(self, dataloader, correction_fn):
        self.dataloader = dataloader
        self.correction_fn = correction_fn
        
    def __iter__(self):
        for images, labels in self.dataloader:
            yield images, self.correction_fn(labels)
    
    def __len__(self):
        return len(self.dataloader)
    

class LightweightGridDetectionModel(pl.LightningModule):
    def __init__(self, grid_lines, danger_levels, input_shape=(3, 240, 520), learning_rate=0.001, hierarchical_config='balanced'):
        super().__init__()
        
        # Store grid dimensions and model parameters
        self.grid_x = len(grid_lines[0]) - 1  # 5 columns
        self.grid_y = len(grid_lines[1]) - 1  # 1 row (for now)
        self.grid_lines = grid_lines
        self.danger_levels = danger_levels  # 3 levels (safe, warning, danger)
        self.output_size = self.danger_levels * self.grid_x * self.grid_y
        self.learning_rate = learning_rate
        
        # Efficient small model architecture
        # Convolutional features with reduced complexity
        self.conv_features = nn.Sequential(
            # input shape: 3x240x520
            nn.MaxPool2d(2),  # Reduce size immediately to 3x120x260
            
            # First conv block - reduce channels while keeping spatial info
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2),  # 8x60x130
            nn.ReLU(),
            nn.BatchNorm2d(8),
            
            # Second conv block - extract features
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),  # 16x30x65
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # Third conv block - final feature extraction
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),  # 16x30x65
            nn.ReLU(),
            
            # Global pooling to dramatically reduce parameters
            nn.AdaptiveAvgPool2d((5, 1)),  # Force output to match our grid size: 16x5x1
            
            # No flattening - we'll reshape later
        )
        
        # 1x1 convolutions for final classification per grid cell
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),  # 8x5x1
            nn.ReLU(),
            nn.Conv2d(8, danger_levels, kernel_size=1),  # danger_levels x 5 x 1
            nn.Sigmoid()  # Output confidence values between 0-1
        )
        
        # Define loss function
        self.loss_fn = SafetyLosses.hierarchical_loss(hierarchical_config)
        self.loss_fn_name = f'hierarchical_{hierarchical_config}'

        # Dictionary to track metrics for each loss function
        self.validation_metrics = {}

    def calculate_safety_metrics(self, pred, target):
        # Reshape from [batch, classes, grid, 1] to [batch * grid, classes]
        batch_size, n_classes, grid_size, _ = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(-1, n_classes)
        target = target.permute(0, 2, 3, 1).reshape(-1, n_classes)
        
        pred_classes = (torch.sigmoid(pred) > 0.5).float()
        
        # Dangerous misclassifications (predicted safe when dangerous)
        dangerous_misclass = ((pred_classes[:, 0] == 1) & (target[:, 2] == 1)).float().mean()
        
        # Safe misclassifications (predicted dangerous when safe)
        safe_misclass = ((pred_classes[:, 2] == 1) & (target[:, 0] == 1)).float().mean()
        
        # "Safe" errors (predicted warning instead of dangerous)
        safe_errors = ((pred_classes[:, 1] == 1) & (target[:, 2] == 1)).float().mean()
        
        return {
            'dangerous_misclass': dangerous_misclass.item(),
            'safe_misclass': safe_misclass.item(),
            'safe_errors': safe_errors.item()
        }
        
    def forward(self, x):
        print(f"Model input shape: {x.shape}")
        
        # Normalize input
        x = x.float() / 255.0
        print(f"After normalization: {x.shape}")
        
        x = self.stage1(x)
        print(f"After stage1: {x.shape}")
        
        x = self.stage2(x)
        print(f"After stage2: {x.shape}")
        
        x = self.stage3(x)
        print(f"After stage3: {x.shape}")
        
        x = self.pool(x)
        print(f"After pool: {x.shape}")
        
        x = self.classifier(x)
        print(f"After classifier: {x.shape}")
        
        return x
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log(f'train_loss_{self.loss_fn_name}', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate regular loss
        val_loss = self.loss_fn(y_hat, y)
        
        # Reshape predictions and targets
        batch_size, n_classes, grid_size, _ = y_hat.shape
        y_hat_flat = y_hat.permute(0, 2, 3, 1).reshape(-1, n_classes)
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, n_classes)
        
        # Get probabilities
        pred_probs = torch.sigmoid(y_hat_flat)
        
        # Make predictions mutually exclusive
        pred_classes = torch.zeros_like(pred_probs)
        max_probs, max_indices = pred_probs.max(dim=1)
        pred_classes[torch.arange(pred_classes.shape[0]), max_indices] = 1
        
        if batch_idx < 3:  # Print first 3 batches
            print(f"\nDetailed Metric Calculation for {self.loss_fn_name} - Batch {batch_idx}:")
            print("\nProbabilities and Predictions:")
            for i in range(min(5, y_hat_flat.shape[0])):
                print(f"\nSample {i}:")
                print("Probabilities:", pred_probs[i])
                print("Prediction:", pred_classes[i])
                print("Target:", y_flat[i])
            
            print("\nBatch Statistics:")
            print("Number of safe targets:", (y_flat[:, 0] == 1).sum().item())
            print("Number of warning targets:", (y_flat[:, 1] == 1).sum().item())
            print("Number of dangerous targets:", (y_flat[:, 2] == 1).sum().item())
            
            print("\nPrediction Statistics:")
            print("Number of safe predictions:", (pred_classes[:, 0] == 1).sum().item())
            print("Number of warning predictions:", (pred_classes[:, 1] == 1).sum().item())
            print("Number of dangerous predictions:", (pred_classes[:, 2] == 1).sum().item())
        
        # Calculate metrics
        accuracies = {
            f'val_accuracy_{class_name}': ((pred_classes[:, i] == y_flat[:, i]).float().mean())
            for i, class_name in enumerate(['safe', 'warning', 'danger'])
        }
        
        dangerous_misclass = ((pred_classes[:, 0] == 1) & (y_flat[:, 2] == 1)).float().mean()
        
        # Log metrics
        self.log('val_loss', val_loss)
        for name, value in accuracies.items():
            self.log(name, value)
        self.log('val_dangerous_misclass', dangerous_misclass)
        
        return {
            'val_loss': val_loss,
            **accuracies,
            'val_dangerous_misclass': dangerous_misclass
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
# class LightweightGridDetectionModelUYVY(pl.LightningModule):
#     def __init__(self, grid_lines, input_shape=(1, 240, 520*2), learning_rate=0.001, hierarchical_config='balanced'):
#         super().__init__()
        
#         self.grid_x = len(grid_lines[0]) - 1
#         self.grid_y = len(grid_lines[1]) - 1
#         self.grid_lines = grid_lines
#         self.danger_config = DangerLevelConfig()
#         self.learning_rate = learning_rate

#         # Process features with fixed dimensions
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(3, 4), stride=(1, 4), padding=(1, 0)),
#             nn.ReLU()
#         )
        
#         self.stage2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
        
#         self.stage3 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
        
#         self.pool = nn.AdaptiveAvgPool2d((5, 1))
        
#         self.classifier = nn.Sequential(
#             nn.Conv2d(16, len(self.danger_config.LEVEL_DEFINITIONS), kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.loss_fn = SafetyLosses.hierarchical_loss(hierarchical_config)
#         self.loss_fn_name = f'hierarchical_{hierarchical_config}'

#     def _debug_shape(self, x, name):
#         # print(f"\nDebug {name}:")
#         # print(f"Shape: {x.shape}")
#         # print(f"Type: {x.dtype}")
#         # print(f"Device: {x.device}")
#         # print(f"Range: [{x.min().item():.3f}, {x.max().item():.3f}]")
#         if torch.isnan(x).any():
#             print("WARNING: Contains NaN values!")
#         if torch.isinf(x).any():
#             print("WARNING: Contains Inf values!")
#         return x

#     def forward(self, x):
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)
        
#         x = x.float() / 255.0
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.pool(x)
#         x = self.classifier(x)
        
#         return x

#     def training_step(self, batch, batch_idx):
#         # print(f"\n=== Starting training step {batch_idx} ===")
#         images, targets = batch
#         # print(f"Batch input shape: {images.shape}")
#         # print(f"Batch target shape: {targets.shape}")
        
#         outputs = self(images)
#         loss = self.loss_fn(outputs, targets)
        
#         self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         val_loss = self.loss_fn(y_hat, y)
        
#         # Reshape predictions and targets to 2D: (batch*height*width, n_classes)
#         y_hat_flat = y_hat.permute(0, 2, 3, 1).reshape(-1, y_hat.shape[1])
#         y_flat = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])
        
#         # Get predicted classes (after sigmoid)
#         pred_probs = torch.sigmoid(y_hat_flat)
#         pred_classes = (pred_probs > 0.5).float()
        
#         # Calculate per-class metrics
#         metrics = {}
#         for level in self.danger_config.LEVEL_DEFINITIONS:
#             level_name = level["name"].lower()
#             level_idx = level["level"]
            
#             # Get masks for true positives, false positives, etc.
#             true_positives = ((pred_classes[:, level_idx] == 1) & (y_flat[:, level_idx] == 1)).sum()
#             true_negatives = ((pred_classes[:, level_idx] == 0) & (y_flat[:, level_idx] == 0)).sum()
#             false_positives = ((pred_classes[:, level_idx] == 1) & (y_flat[:, level_idx] == 0)).sum()
#             false_negatives = ((pred_classes[:, level_idx] == 0) & (y_flat[:, level_idx] == 1)).sum()
            
#             # Calculate metrics
#             accuracy = (true_positives + true_negatives) / len(y_flat)
#             precision = true_positives / (true_positives + false_positives + 1e-8)
#             recall = true_positives / (true_positives + false_negatives + 1e-8)
#             f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
#             # Store metrics
#             metrics.update({
#                 f'val_accuracy_{level_name}': accuracy,
#                 f'val_precision_{level_name}': precision,
#                 f'val_recall_{level_name}': recall,
#                 f'val_f1_{level_name}': f1
#             })
        
#         # Calculate critical misclassification rate (dangerous regions classified as safe)
#         dangerous_as_safe = ((pred_classes[:, self.danger_config.LEVELS['SAFE']] == 1) & 
#                             (y_flat[:, self.danger_config.LEVELS['DANGER']] == 1)).sum()
#         total_dangerous = (y_flat[:, self.danger_config.LEVELS['DANGER']] == 1).sum()
#         critical_misclass_rate = dangerous_as_safe / (total_dangerous + 1e-8)
        
#         # Calculate class distribution in this batch
#         class_distribution = {
#             level["name"].lower(): (y_flat[:, level["level"]] == 1).sum() / len(y_flat)
#             for level in self.danger_config.LEVEL_DEFINITIONS
#         }
        
#         # Log everything
#         self.log('val_loss', val_loss, prog_bar=True)
#         self.log('val_critical_misclass', critical_misclass_rate, prog_bar=True)
        
#         for name, value in metrics.items():
#             self.log(name, value, prog_bar=True)
        
#         for class_name, dist in class_distribution.items():
#             self.log(f'val_dist_{class_name}', dist)
        
#         return {
#             'val_loss': val_loss,
#             'val_critical_misclass': critical_misclass_rate,
#             **metrics,
#             'class_distribution': class_distribution
#         }
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class LightweightGridDetectionModelUYVY(pl.LightningModule):
    def __init__(self, grid_lines, input_shape=(1, 240, 520*2), learning_rate=0.001):
        super().__init__()
        
        self.grid_x = len(grid_lines[0]) - 1
        self.grid_y = len(grid_lines[1]) - 1
        self.grid_lines = grid_lines
        self.danger_config = DangerLevelConfig()
        self.learning_rate = learning_rate
        
        # Network architecture
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 4), stride=(1, 4), padding=(1, 0)),
            nn.ReLU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((5, 1))
        
        self.classifier = nn.Sequential(
            nn.Conv2d(16, len(self.danger_config.LEVEL_DEFINITIONS), kernel_size=1)
        )
        
        # Initialize base weights
        self.register_buffer('base_weights', torch.tensor([1.0, 3.0, 2.0]))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = x.float() / 255.0
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.classifier(x)
        
        return x
    
    def calculate_f1_loss(self, y_pred, y_true):
        """
        Calculate F1 score as a differentiable loss function
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(y_pred)
        
        # Calculate F1 score components
        tp = torch.sum(probs * y_true, dim=(0, 2, 3))
        fp = torch.sum(probs * (1 - y_true), dim=(0, 2, 3))
        fn = torch.sum((1 - probs) * y_true, dim=(0, 2, 3))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Return negative mean F1 score (since we're minimizing)
        return -f1.mean()

    def _get_current_weights(self):
        epoch = self.current_epoch
        # Reduce the danger multiplier to allow more safe predictions
        danger_multiplier = min(3.0, 1.0 + epoch * 0.3)  # Reduced from 5.0
        weights = self.base_weights.clone()
        weights[1:] *= danger_multiplier
        return weights

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        # Calculate F1-based loss
        loss = self.calculate_f1_loss(outputs, targets)
        
        # Log just the overall loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate F1-based loss
        val_loss = self.calculate_f1_loss(y_hat, y)
        
        # Calculate critical misclassification rate
        with torch.no_grad():
            probs = torch.sigmoid(y_hat)
            pred_classes = (probs > 0.5).float()
            dangerous_as_safe = ((pred_classes[:, self.danger_config.LEVELS['SAFE']] == 1) & 
                            (y[:, self.danger_config.LEVELS['DANGER']] == 1)).float().mean()
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_critical_misclass', dangerous_as_safe, prog_bar=True)
        
        return {
            'val_loss': val_loss,
            'val_critical_misclass': dangerous_as_safe
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitoring F1-based loss
                "frequency": 1
            }
        }

    
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

class ObjectDetectionDataModuleUYVY(pl.LightningDataModule):
    def __init__(self, image_dir_train, image_dir_val, width, height, grid_lines, 
                 batch_size=8, device="cpu"):
        super(ObjectDetectionDataModuleUYVY, self).__init__()
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val
        self.width = width  # Original width
        self.height = height
        self.batch_size = batch_size
        self.grid_lines = grid_lines
        self.device = device

    def setup(self, stage=None):
        self.train_dataset = CustomImageDatasetUYVY(
            self.image_dir_train, 
            self.width,
            self.height, 
            self.grid_lines,
            self.device
        )
        
        self.val_dataset = CustomImageDatasetUYVY(
            self.image_dir_val, 
            self.width,
            self.height, 
            self.grid_lines,
            self.device
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

