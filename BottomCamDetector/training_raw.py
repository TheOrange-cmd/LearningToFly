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
import torch.nn.functional as F
from data_handling_raw import BorderDataset, prepare_dataset
from model_raw import BorderDetector
from utils_raw import EarlyStopping, TrainingMetrics, evaluate_model, save_checkpoint, export_to_onnx
from torch.quantization import get_default_qconfig

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
        print("No configuration provided, stopping ...")
        return
    
    # Create output directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results_raw/{"test_" if config.get("test_mode", False) else ""}training_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use test directory for test runs
    data_dir = 'test_verified_images_raw' if config.get('test_mode', False) else 'verified_images_raw'
    
    # Prepare dataset
    images, labels, sources = prepare_dataset(
        data_dir,
        num_rotations=config['num_rotations'],
        augment_brightness_contrast=config['augment_brightness_contrast']
    )
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Create data loaders
    train_dataset = BorderDataset(X_train, y_train)
    val_dataset = BorderDataset(X_val, y_val)
    
    # Create weighted sampler for training data
    train_sampler = create_balanced_sampler(train_dataset.labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True 
    )
    
    val_sampler = create_balanced_sampler(val_dataset.labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True  
    )
    
    # Initialize QAT model
    model = BorderDetector(config)
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    class StableBCELoss(nn.Module):
        def __init__(self, epsilon=1e-7):
            super().__init__()
            self.epsilon = epsilon

        def forward(self, pred, target):
            pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
            return F.binary_cross_entropy(pred, target)

    # In training loop:
    criterion = StableBCELoss()
    early_stopping = EarlyStopping(patience=config['patience'])
    metrics_tracker = TrainingMetrics()
    
    print("\nMemory before training:")
    print_memory_usage()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        accumulated_loss = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            if torch.isnan(outputs).any():
                print(f"\nNaN detected at epoch {epoch}, batch {batch_idx}")
                model.debug_mode = True
                _ = model(images)
                return None, None, None

            loss = criterion(outputs, labels) / config['accumulation_steps']
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % config['accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                
                # Reset accumulated loss
                accumulated_loss = 0
            
            train_loss += loss.item() * config['accumulation_steps']
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
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
    
    # Convert to fully quantized model
    model.eval()
    
    # Save final model and metrics
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    metrics_tracker.plot_metrics(os.path.join(output_dir, 'training_curves.png'))
    metrics_tracker.save_metrics(os.path.join(output_dir, 'metrics.json'))
    
    # Export to ONNX
    export_to_onnx(model, os.path.join(output_dir, 'model.onnx'))

    # Print results
    print("\nFinal metrics:")
    metrics_tracker.print_metrics()

    
    return model, metrics_tracker, output_dir

