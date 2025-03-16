import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from torch.quantization import get_default_qconfig
from model_raw import BorderDetector

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
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        # Add additional metric lists
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_auc_scores = []
        self.val_auc_scores = []
        
    def add_metrics(self, train_loss, val_loss, train_acc, val_acc, lr, 
                   train_precision=None, val_precision=None,
                   train_recall=None, val_recall=None,
                   train_f1=None, val_f1=None,
                   train_auc=None, val_auc=None):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_accuracies.append(float(train_acc))
        self.val_accuracies.append(float(val_acc))
        self.learning_rates.append(float(lr))
        # Add additional metrics
        if train_precision is not None:
            self.train_precisions.append(float(train_precision))
        if val_precision is not None:
            self.val_precisions.append(float(val_precision))
        if train_recall is not None:
            self.train_recalls.append(float(train_recall))
        if val_recall is not None:
            self.val_recalls.append(float(val_recall))
        if train_f1 is not None:
            self.train_f1_scores.append(float(train_f1))
        if val_f1 is not None:
            self.val_f1_scores.append(float(val_f1))
        if train_auc is not None:
            self.train_auc_scores.append(float(train_auc))
        if val_auc is not None:
            self.val_auc_scores.append(float(val_auc))
    
    def to_dict(self):
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
        }
        # Add additional metrics if they exist
        if self.train_precisions:
            metrics_dict['train_precisions'] = self.train_precisions
        if self.val_precisions:
            metrics_dict['val_precisions'] = self.val_precisions
        if self.train_recalls:
            metrics_dict['train_recalls'] = self.train_recalls
        if self.val_recalls:
            metrics_dict['val_recalls'] = self.val_recalls
        if self.train_f1_scores:
            metrics_dict['train_f1_scores'] = self.train_f1_scores
        if self.val_f1_scores:
            metrics_dict['val_f1_scores'] = self.val_f1_scores
        if self.train_auc_scores:
            metrics_dict['train_auc_scores'] = self.train_auc_scores
        if self.val_auc_scores:
            metrics_dict['val_auc_scores'] = self.val_auc_scores
        return metrics_dict
    
    def print_metrics(self):
        print(f"Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}")
        print(f"Train Accuracy: {self.train_accuracies[-1]:.2f}%, Val Accuracy: {self.val_accuracies[-1]:.2f}%")
        print(f"Learning Rate: {self.learning_rates[-1]:.6f}")
        # Print additional metrics
        if self.train_precisions:
            print(f"Train Precision: {self.train_precisions[-1]:.4f}, Val Precision: {self.val_precisions[-1]:.4f}")
        if self.train_recalls:
            print(f"Train Recall: {self.train_recalls[-1]:.4f}, Val Recall: {self.val_recalls[-1]:.4f}")
        if self.train_f1_scores:
            print(f"Train F1 Score: {self.train_f1_scores[-1]:.4f}, Val F1 Score: {self.val_f1_scores[-1]:.4f}")
        if self.train_auc_scores:
            print(f"Train AUC Score: {self.train_auc_scores[-1]:.4f}, Val AUC Score: {self.val_auc_scores[-1]:.4f}")
    
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
    all_labels = []
    all_predictions = []
    all_outputs = []  # For AUC calculation
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if torch.any(outputs < 0) or torch.any(outputs > 1):
                print(f"Warning: outputs outside [0,1] range: min={outputs.min().item()}, max={outputs.max().item()}")
                outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            
            # Store for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_outputs = np.array(all_outputs)
    
    # Calculate metrics
    metrics = {
        'val_loss': total_loss / len(dataloader),
        'val_accuracy': 100 * np.mean(all_predictions == all_labels),
        'val_precision': precision_score(all_labels, all_predictions, zero_division=0),
        'val_recall': recall_score(all_labels, all_predictions, zero_division=0),
        'val_f1': f1_score(all_labels, all_predictions, zero_division=0),
        'val_auc': roc_auc_score(all_labels, all_outputs)
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)

def export_to_onnx(model, output_path):
    """
    Export model to ONNX format optimized for onnx2c
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 120, 120)
    
    # Export the model
    torch.onnx.export(model,
                     dummy_input,
                     output_path,
                     export_params=True,
                     opset_version=12,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'])
