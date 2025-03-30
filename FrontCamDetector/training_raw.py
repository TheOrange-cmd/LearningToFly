import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import gc
import json
import gc
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from data_handling_raw import DangerDataset
from model_raw import ObstacleDetector
from utils_raw import EarlyStopping, TrainingMetrics, evaluate_model, save_checkpoint, export_to_onnx
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from IPython.display import display, HTML
from torch.optim.lr_scheduler import OneCycleLR

# Apply dark styling to all progress bars
display(HTML("""
    <style>
    /* Dark theme for all progress bars */
    .jupyter-widgets-output-area .progress {
        background-color: #222;
        border-radius: 8px;
        height: 20px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);
    }
    .jupyter-widgets-output-area .progress-bar-success {
        background-color: #375a7f;
        background-image: linear-gradient(90deg, 
                        #375a7f 0%, 
                        #3498db 100%);
    }
    </style>
    """))


# class StableBCELoss(nn.Module):
#     def __init__(self, epsilon=1e-7):
#         super().__init__()
#         self.epsilon = epsilon

#     def forward(self, pred, target):
#         pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
#         return F.binary_cross_entropy(pred, target)



# class CombinedLoss(nn.Module):
#     def __init__(self, mse_weight=0.3, huber_weight=0.4, variance_weight=0.3):
#         super(CombinedLoss, self).__init__()
#         self.mse_weight = mse_weight
#         self.huber_weight = huber_weight
#         self.variance_weight = variance_weight
#         self.huber = HuberLoss(delta=0.1)
        
#     def forward(self, predictions, targets):
#         # Standard MSE component
#         mse_loss = F.mse_loss(predictions, targets)
        
#         # Huber loss component
#         huber_loss = self.huber(predictions, targets)
        
#         # Variance matching component (encourage similar std dev)
#         pred_std = torch.std(predictions, dim=0)
#         target_std = torch.std(targets, dim=0)
#         variance_loss = F.mse_loss(pred_std, target_std)
        
#         # Combined loss
#         return (self.mse_weight * mse_loss + 
#                 self.huber_weight * huber_loss + 
#                 self.variance_weight * variance_loss)

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

def train_with_cross_validation(config, dataset, n_folds=5):
    """
    Train the model using k-fold cross-validation.
    
    Args:
        config: Configuration dictionary
        dataset: DangerDataset instance
        n_folds: Number of folds for cross-validation
    
    Returns:
        Dictionary containing cross-validation results
    """
    # Get indices for cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results from each fold
    cv_results = {
        'val_mse_scores': [],
        'val_mae_scores': [],
        'val_huber_scores': [],
        'models': [],
        'fold_metrics': []
    }

    # Create base output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f'results_raw/cv_{"test_" if config.get("test_mode", False) else ""}training_{timestamp}'
    os.makedirs(base_output_dir, exist_ok=True)

    # print if device is available
    print(f"Cuda available: {torch.cuda.is_available()}")
    import sys
    # Perform cross-validation
    with tqdm_notebook(total=n_folds, desc="CV Folds", file=sys.stdout, colour='black') as fold_bar:
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
            # Create fold directory
            fold_output_dir = os.path.join(base_output_dir, f'fold_{fold+1}')
            os.makedirs(fold_output_dir, exist_ok=True)

            # Create samplers
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            # Create data loaders
            train_loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=0,
                pin_memory=False
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                sampler=val_sampler,
                num_workers=0,
                pin_memory=False
            )
            
            # Model setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ObstacleDetector(config).to(device)
            model.debug_mode = config.get('debug_mode', False)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                eps=1e-8
            )
            
            # scheduler = CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=15,
            #     T_mult=2,
            #     eta_min=1e-7
            # )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=config['num_epochs'] * len(train_loader),
                pct_start=0.3,  # Spend 30% of training warming up
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=1000,  # Final LR = max_lr/1000
                anneal_strategy='cos'
            )
            # use huber loss
            criterion = nn.HuberLoss(reduction='mean', delta=0.1)
            early_stopping = EarlyStopping(patience=config['patience'])
            metrics_tracker = TrainingMetrics()
            
            # Use tqdm_notebook for the inner loop too
            best_val_mse = float('inf')
            best_val_mae = float('inf')
            best_val_huber = float('inf')
            
            with tqdm_notebook(total=config['num_epochs'], desc=f"Fold {fold+1} Epochs" , colour='grey') as epoch_bar:
                for epoch in range(config['num_epochs']):
                    model.train()
                    train_loss = 0
                    train_mse = 0
                    train_mae = 0
                    
                    # Training loop
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(device), labels.to(device)
                        
                        # Forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                        optimizer.step()
                        
                        # Update metrics
                        train_loss += loss.item()
                        train_mse += F.mse_loss(outputs, labels).item()
                        train_mae += F.l1_loss(outputs, labels).item()
                        scheduler.step() # use for OneCycleLR
                    
                    # Validation
                    val_metrics = evaluate_model(model, val_loader, device)

                    # Update metrics for display
                    avg_train_loss = train_loss/len(train_loader)
                    
                    # Update epoch progress bar
                    epoch_bar.set_postfix({
                        'train': f"{avg_train_loss:.4f}",
                        'val': f"{val_metrics['val_loss']:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                    # Update metrics tracker with all regression metrics
                    metrics_tracker.add_metrics(
                        train_loss=avg_train_loss,
                        val_loss=val_metrics['val_loss'],
                        train_mse=train_mse/len(train_loader),
                        val_mse=val_metrics['val_mse'],
                        train_mae=train_mae/len(train_loader),
                        val_mae=val_metrics['val_mae'],
                        lr=optimizer.param_groups[0]['lr']
                    )
                    
                    # Save checkpoint
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        metrics_tracker,
                        os.path.join(fold_output_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                        config
                    )

                    best_val_mae = min(best_val_mae, val_metrics['val_mae'])
                    best_val_mse = min(best_val_mse, val_metrics['val_mse'])
                    best_val_huber = min(best_val_huber, val_metrics['val_huber'])
                    
                    # Early stopping - use print instead of tqdm.write in notebooks
                    early_stopping(val_metrics['val_huber'], model.state_dict())
                    if early_stopping.early_stop:
                        epoch_bar.set_postfix({
                            'train': f"{avg_train_loss:.4f}",
                            'val': f"{val_metrics['val_loss']:.4f}",
                            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                            "Early stopping": "True"
                        })
                        break
                    
                    # Update the epoch bar
                    epoch_bar.update(1)
                    # scheduler.step() use for cosine annealing
                    
            # Update fold bar after epoch loop is done
            fold_bar.set_postfix({
                'best_mse': f"{best_val_mse:.4f}",
                'best_mae': f"{best_val_mae:.4f}",
                'best_huber': f"{best_val_huber:.4f}"
            })
            fold_bar.update(1)
            
            # Find best model based on validation huber
            best_epoch = np.argmin(metrics_tracker.val_losses)
            best_epoch = int(best_epoch)
            best_val_mse = metrics_tracker.val_mse[best_epoch]
            best_val_mae = metrics_tracker.val_mae[best_epoch]
            
            # Load best checkpoint
            checkpoint_path = os.path.join(fold_output_dir, f'checkpoint_epoch_{best_epoch+1}.pt')
            checkpoint = torch.load(checkpoint_path)
            fold_model = ObstacleDetector(config)
            fold_model.load_state_dict(checkpoint['model_state_dict'])
            fold_model.eval()
            
            # Export to ONNX
            best_model_path = os.path.join(fold_output_dir, f'best_model_fold_{fold+1}.onnx')
            export_to_onnx(fold_model, best_model_path)
            
            # Store results
            cv_results['val_mse_scores'].append(best_val_mse)
            cv_results['val_mae_scores'].append(best_val_mae)
            cv_results['models'].append(fold_model)
            cv_results['val_huber_scores'].append(best_val_huber)
            cv_results['fold_metrics'].append({
                'best_epoch': int(best_epoch + 1), 
                'train_loss': float(metrics_tracker.train_losses[best_epoch]),
                'val_loss': float(metrics_tracker.val_losses[best_epoch]),
                'train_mse': float(metrics_tracker.train_mse[best_epoch]),
                'val_mse': float(best_val_mse),
                'train_mae': float(metrics_tracker.train_mae[best_epoch]),
                'val_mae': float(best_val_mae)
            })
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate final metrics
    avg_mse = np.mean(cv_results['val_mse_scores'])
    avg_mae = np.mean(cv_results['val_mae_scores'])
    avg_huber = np.mean(cv_results['val_huber_scores'])
    
    print("\n" + "="*60)
    print(f"Cross-validation results ({n_folds} folds):")
    print(f"Average MSE: {avg_mse:.4f} ± {np.std(cv_results['val_mse_scores']):.4f}")
    print(f"Average MAE: {avg_mae:.4f} ± {np.std(cv_results['val_mae_scores']):.4f}")
    print(f"Average Huber: {avg_huber:.4f} ± {np.std(cv_results['val_huber_scores']):.4f}")
    
    # Save summary
    summary_path = os.path.join(base_output_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'avg_mse': float(avg_mse),
            'avg_mae': float(avg_mae),
            'avg_huber': float(avg_huber),
            'std_mse': float(np.std(cv_results['val_mse_scores'])),
            'std_mae': float(np.std(cv_results['val_mae_scores'])),
            'std_huber': float(np.std(cv_results['val_huber_scores'])),
            'fold_results': cv_results['fold_metrics'],
            'config': config
        }, f, indent=4)
    
    return cv_results, base_output_dir

