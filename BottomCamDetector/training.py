import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from tqdm import tqdm
import gc
import psutil
import json
import gc
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data_handling import BorderDataset, prepare_dataset
from model import BorderDetector
from utils import EarlyStopping, TrainingMetrics, evaluate_model, save_checkpoint


def create_balanced_sampler(labels):
    # Calculate weights inversely proportional to class frequencies
    class_counts = np.bincount(labels.astype(int))
    weights = 1. / class_counts
    sample_weights = weights[labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler



def train_model(config=None):
    # Add memory tracking

    
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    print("Initial memory state:")
    print_memory_usage()
    
    if config is None:
        config = {
            'conv_channels': [4, 8, 8],
            'dropout_rate': 0.1,
            'batch_size': 2,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'num_epochs': 30,
            'num_rotations': 9,
            'augment_brightness_contrast': True,
            'patience': 5,
            'accumulation_steps': 4  # Add gradient accumulation
        }
    
    # Create output directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/{"test_" if config.get("test_mode", False) else ""}training_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use test directory for test runs
    data_dir = 'test_verified_images' if config.get('test_mode', False) else 'verified_images'
    
    # Prepare dataset
    images, masks, labels, sources = prepare_dataset(
        data_dir,
        num_rotations=config['num_rotations'],
        augment_brightness_contrast=config['augment_brightness_contrast']
    )
    # print(f"Dataset size: {len(images)} images")
    
    # Split dataset
    X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(
        images, masks, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Create data loaders
    train_dataset = BorderDataset(X_train, masks_train, y_train)
    val_dataset = BorderDataset(X_val, masks_val, y_val)
    # Create weighted sampler for training data
    train_sampler = create_balanced_sampler(train_dataset.labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,  # Use balanced sampler
        num_workers=0,
        pin_memory=False
    )
    
    val_sampler = create_balanced_sampler(val_dataset.labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=False
    )
        
    # Initialize model and training components
    model = BorderDetector(
        conv_channels=config['conv_channels'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Number of epochs for first restart
        T_mult=2,  # Multiply T_0 by this factor after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=config['patience'])
    metrics_tracker = TrainingMetrics()

    print("\nMemory before training:")
    print_memory_usage()
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Clear memory before each epoch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # print(f"\nMemory at start of epoch {epoch+1}:")
        # print_memory_usage()
        
        model.train()
        # print("\nChecking model parameters:")
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        accumulated_labels = []
        accumulated_outputs = [] 
        accumulated_images = [] 
        for batch_idx, (images, masks, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            try:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)

                # Normalize input more aggressively if needed
                if images.abs().max() > 1.0:
                    images = torch.tanh(images)  # Squash to [-1,1]
                outputs = model(images, masks)
                
                # Only monitor first accumulation step
                # if batch_idx == 0:
                #     print(f"\nFirst batch:")
                #     print(f"Images: [{images.min().item():.3f}, {images.max().item():.3f}]")
                #     print(f"Outputs: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                #     print("\nInitial gradients:")
                #     model.check_gradients()

                # Monitor for any issues with outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\nNaN/Inf detected in batch {batch_idx}")
                    print(f"Outputs range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                    print(f"Gradients at failure:")
                    model.check_gradients()
                    raise ValueError("NaN/Inf detected in outputs")
                
                # Calculate and scale loss
                scale_factor = len(model.conv1.weight.view(-1)) / 100
                scaled_loss = criterion(outputs, labels) / (config['accumulation_steps'] * scale_factor)
                scaled_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                # Calculate metrics
                train_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
            except Exception as e:
                if "out of memory" in str(e):
                    print(f"\nOUT OF MEMORY on batch {batch_idx}")
                    print_memory_usage()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise e
        
        # Handle any remaining gradients
        if (batch_idx + 1) % config['accumulation_steps'] != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_accuracy']
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        metrics_tracker.add_metrics(
            train_loss=train_loss, 
            val_loss=val_loss, 
            train_acc=train_acc, 
            val_acc=val_acc, 
            lr=current_lr,
            val_precision=val_metrics.get('val_precision'),
            val_recall=val_metrics.get('val_recall'),
            val_f1=val_metrics.get('val_f1'),
            val_auc=val_metrics.get('val_auc')
        )

        # Print progress with additional metrics
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print(f'Val Precision: {val_metrics["val_precision"]:.4f}')
        print(f'Val Recall: {val_metrics["val_recall"]:.4f}')
        print(f'Val F1: {val_metrics["val_f1"]:.4f}')
        print(f'Val AUC: {val_metrics["val_auc"]:.4f}')
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metrics_tracker,
            os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        )
        
        # Early stopping
        early_stopping(val_loss, model.state_dict())
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Save final metrics and plots
    metrics_tracker.plot_metrics(os.path.join(output_dir, 'training_curves.png'))
    metrics_tracker.save_metrics(os.path.join(output_dir, 'metrics.json'))
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    return model, metrics_tracker, output_dir