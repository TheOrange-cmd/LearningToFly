import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from torch.quantization import get_default_qconfig
from model_raw import ObstacleDetector
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, val_loss, model_state):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model_state
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model_state
            self.counter = 0

class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.train_mae = []
        self.val_mae = []
        self.learning_rates = []
        
    def add_metrics(self, train_loss, val_loss, train_mse=None, val_mse=None, train_mae=None, val_mae=None, lr=None):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        if train_mse is not None:
            self.train_mse.append(float(train_mse))
        if val_mse is not None:
            self.val_mse.append(float(val_mse))
        if train_mae is not None:
            self.train_mae.append(float(train_mae))
        if val_mae is not None:
            self.val_mae.append(float(val_mae))
        if lr is not None:
            self.learning_rates.append(float(lr))
    
    def to_dict(self):
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'learning_rates': self.learning_rates,
        }
        return metrics_dict
    
    def print_metrics(self):
        print(f"Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}")
        print(f"Train MSE: {self.train_mse[-1]:.4f}, Val MSE: {self.val_mse[-1]:.4f}")
        print(f"Train MAE: {self.train_mae[-1]:.4f}, Val MAE: {self.val_mae[-1]:.4f}")
        print(f"Learning Rate: {self.learning_rates[-1]:.6f}")
    
    @classmethod
    def from_dict(cls, data):
        metrics = cls()
        metrics.train_losses = data['train_losses']
        metrics.val_losses = data['val_losses']
        metrics.train_accuracies = data['train_accuracies']
        metrics.val_accuracies = data['val_accuracies']
        metrics.learning_rates = data['learning_rates']
        return metrics
    
    def plot_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    
    def save_metrics(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_metrics(cls, load_path):
        with open(load_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate MSE and MAE
            mse = F.mse_loss(outputs, labels).item()
            mae = F.l1_loss(outputs, labels).item()
            
            total_mse += mse
            total_mae += mae
    
    # Average metrics over the dataset
    metrics = {
        'val_loss': total_loss / len(dataloader),
        'val_mse': total_mse / len(dataloader),
        'val_mae': total_mae / len(dataloader),
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': config,
        'metrics': metrics,
    }, path)

def export_to_onnx(model, output_path):
    """
    Export model to ONNX format optimized for onnx2c
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, model.input_size, model.input_size)
    
    # Export the model
    torch.onnx.export(model,
                     dummy_input,
                     output_path,
                     export_params=True,
                     opset_version=12,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'])
