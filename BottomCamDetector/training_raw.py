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
from sklearn.model_selection import StratifiedKFold
import numpy as np
class StableBCELoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        return F.binary_cross_entropy(pred, target)

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

def select_best_model_checkpoint(metrics_tracker, window_size=3):
    """
    Select the best model checkpoint using a sophisticated selection strategy.
    
    Args:
        metrics_tracker: Object containing all training metrics
        window_size: Size of the moving window for smoothing (default: 3)
    
    Returns:
        best_epoch: The epoch number of the best checkpoint
        selection_metrics: Dictionary containing the metrics that led to this selection
    """
    # Convert metrics to numpy arrays for easier manipulation
    val_auc = np.array(metrics_tracker.val_auc_scores)
    val_loss = np.array(metrics_tracker.val_losses)
    
    # Calculate moving averages to reduce noise
    def moving_average(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window
    
    if len(val_auc) > window_size:
        smooth_auc = moving_average(val_auc, window_size)
        smooth_loss = moving_average(val_loss, window_size)
        # Adjust indices to account for the window
        offset = window_size - 1
    else:
        smooth_auc = val_auc
        smooth_loss = val_loss
        offset = 0
    
    # Find peaks in AUC (local maxima)
    peak_indices = []
    for i in range(1, len(smooth_auc) - 1):
        if smooth_auc[i] > smooth_auc[i-1] and smooth_auc[i] >= smooth_auc[i+1]:
            peak_indices.append(i)
    
    if not peak_indices:
        # If no peaks found, just use the maximum AUC
        best_idx = np.argmax(smooth_auc)
    else:
        # Among the peaks, prefer later ones if they're close in performance
        # This helps avoid selecting early lucky epochs
        tolerance = 0.001  # 0.1% tolerance
        best_peak_auc = smooth_auc[peak_indices[-1]]
        best_idx = peak_indices[-1]
        
        # Look at earlier peaks
        for idx in reversed(peak_indices[:-1]):
            if smooth_auc[idx] > best_peak_auc + tolerance:
                best_idx = idx
                best_peak_auc = smooth_auc[idx]
    
    # Convert back to original epoch index
    best_epoch = best_idx + offset
    
    return best_epoch, metrics_tracker.val_auc_scores[best_epoch]

def train_model(config, dataset_tuple):
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
    
    # Use provided dataset if available, otherwise prepare it
    if dataset_tuple is not None:
        images, labels, sources = dataset_tuple
    else:
        raise ValueError("No dataset provided")
    
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
    


    # In training loop:
    criterion = StableBCELoss()
    early_stopping = EarlyStopping(patience=config['patience'])
    metrics_tracker = TrainingMetrics()
    
    print("\nMemory before training:")
    print_memory_usage()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Training loop
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        accumulated_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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
            os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'),
            config
        )
        
        # Early stopping
        early_stopping(val_loss, model.state_dict())
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Find and load the best model checkpoint
    best_epoch, best_val_auc = select_best_model_checkpoint(metrics_tracker)
    if best_val_auc > 0.9:
        print(f"Best model found at epoch {best_epoch} with AUC: {best_val_auc:.3f}")
    
        # Path to the best checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{best_epoch}.pt')
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['model_config']
        model = BorderDetector(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Export the best model to ONNX
        best_model_path = os.path.join(output_dir, f'best_model_epoch_{best_epoch}.onnx')
        export_to_onnx(model, best_model_path)
        
        print(f"Best model from epoch {best_epoch} exported to {best_model_path}")
    else:
        print(f"No model with AUC > 0.9 found, best AUC: {best_val_auc}. Skipping export.")

    
    return model, metrics_tracker, output_dir

def train_with_cross_validation(config, dataset_tuple, n_folds=5):
    """
    Train the model using k-fold cross-validation.
    
    Args:
        config: Configuration dictionary
        dataset_tuple: Tuple containing (images, labels, sources)
        n_folds: Number of folds for cross-validation
    
    Returns:
        Dictionary containing cross-validation results
    """
    # Unpack dataset
    main_images, main_labels, sources = dataset_tuple
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results from each fold
    cv_results = {
        'val_auc_scores': [],
        'val_accuracies': [],
        'val_f1_scores': [],
        'models': [],
        'fold_metrics': []
    }
    
    # Create a base output directory for all folds
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f'results_raw/cv_{"test_" if config.get("test_mode", False) else ""}training_{timestamp}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save configuration for this cross-validation run
    cv_config = config.copy()
    cv_config['n_folds'] = n_folds
    with open(os.path.join(base_output_dir, 'cv_config.json'), 'w') as f:
        json.dump(cv_config, f, indent=4)
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(main_images, main_labels)):
        print(f"\n{'='*50}")
        print(f"Training fold {fold+1}/{n_folds}")
        print(f"{'='*50}")
        
        # print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        # print(f"image and label size: {main_images.shape}, {main_labels.shape}")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(base_output_dir, f'fold_{fold+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Split data for this fold
        X_train, X_val = main_images[train_idx], main_images[val_idx]
        y_train, y_val = main_labels[train_idx], main_labels[val_idx]
        
        # Create datasets for this fold
        train_dataset = BorderDataset(X_train, y_train)
        val_dataset = BorderDataset(X_val, y_val)
        
        # Create samplers
        train_sampler = create_balanced_sampler(train_dataset.labels)
        val_sampler = create_balanced_sampler(val_dataset.labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        # Initialize model, optimizer, scheduler, and criterion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BorderDetector(config).to(device)
        
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
        
        criterion = StableBCELoss()
        early_stopping = EarlyStopping(patience=config['patience'])
        metrics_tracker = TrainingMetrics()
        
        # Training loop
        for epoch in tqdm(range(config['num_epochs']), desc=f"Fold {fold+1} Epochs"):
            # Train
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            optimizer.zero_grad()
            accumulated_loss = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if torch.isnan(outputs).any():
                    print(f"\nNaN detected at fold {fold+1}, epoch {epoch}, batch {batch_idx}")
                    model.debug_mode = True
                    _ = model(images)
                    return None

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
                os.path.join(fold_output_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                config
            )
            
            # Early stopping
            early_stopping(val_loss, model.state_dict())
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Find and load the best model checkpoint for this fold
        best_epoch, best_val_auc = select_best_model_checkpoint(metrics_tracker)
        print(f"Fold {fold+1} - Best model at epoch {best_epoch+1} with AUC: {best_val_auc:.4f}")
        
        # Path to the best checkpoint
        checkpoint_path = os.path.join(fold_output_dir, f'checkpoint_epoch_{best_epoch+1}.pt')
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        fold_model = BorderDetector(config)
        fold_model.load_state_dict(checkpoint['model_state_dict'])
        fold_model.eval()
        
        # Export the best model to ONNX
        best_model_path = os.path.join(fold_output_dir, f'best_model_fold_{fold+1}.onnx')
        export_to_onnx(fold_model, best_model_path)
        
        # Store results for this fold
        cv_results['val_auc_scores'].append(best_val_auc)
        cv_results['val_accuracies'].append(metrics_tracker.val_accuracies[best_epoch])
        cv_results['val_f1_scores'].append(metrics_tracker.val_f1_scores[best_epoch])
        cv_results['models'].append(fold_model)
        cv_results['fold_metrics'].append({
            'best_epoch': best_epoch + 1,
            'train_loss': metrics_tracker.train_losses[best_epoch],
            'val_loss': metrics_tracker.val_losses[best_epoch],
            'train_acc': metrics_tracker.train_accuracies[best_epoch],
            'val_acc': metrics_tracker.val_accuracies[best_epoch],
            'val_auc': metrics_tracker.val_auc_scores[best_epoch],
            'val_f1': metrics_tracker.val_f1_scores[best_epoch]
        })
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate and print average metrics
    avg_auc = np.mean(cv_results['val_auc_scores'])
    avg_acc = np.mean(cv_results['val_accuracies'])
    avg_f1 = np.mean(cv_results['val_f1_scores'])
    
    print("\n" + "="*60)
    print(f"Cross-validation results ({n_folds} folds):")
    print(f"Average AUC: {avg_auc:.4f} ± {np.std(cv_results['val_auc_scores']):.4f}")
    print(f"Average Accuracy: {avg_acc:.4f} ± {np.std(cv_results['val_accuracies']):.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {np.std(cv_results['val_f1_scores']):.4f}")
    
    # Save overall results
    summary_path = os.path.join(base_output_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'avg_auc': float(avg_auc),
            'avg_acc': float(avg_acc),
            'avg_f1': float(avg_f1),
            'std_auc': float(np.std(cv_results['val_auc_scores'])),
            'std_acc': float(np.std(cv_results['val_accuracies'])),
            'std_f1': float(np.std(cv_results['val_f1_scores'])),
            'fold_results': cv_results['fold_metrics']
        }, f, indent=4)
    
    print(f"Summary saved to {summary_path}")
    
    return cv_results, base_output_dir

