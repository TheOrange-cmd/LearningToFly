import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from torch.quantization import get_default_qconfig
from model_raw import ObstacleDetector
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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

    def add_metrics(self, train_loss, val_loss, train_mse, val_mse, train_mae, val_mae, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.learning_rates.append(lr)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_huber = 0.0
    huber_criterion = nn.HuberLoss(reduction='mean', delta=0.1)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate regression metrics
            mse = F.mse_loss(outputs, labels)
            mae = F.l1_loss(outputs, labels)
            huber = huber_criterion(outputs, labels)
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_huber += huber.item()
    
    return {
        'val_loss': total_huber / len(dataloader),  # Use Huber as primary val_loss
        'val_mse': total_mse / len(dataloader),
        'val_mae': total_mae / len(dataloader),
        'val_huber': total_huber / len(dataloader)
    }

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
