'''
This script loads an ONNX model and a dataset of images with danger values,
and compares the true danger values with the predictions made by the model. It visualizes the results in a dashboard format, allowing for easy comparison of true vs predicted values.
While the comparison is not strictly valid because we are comparing predictions with the same images used for training, it is a useful debugging tool to check if the model is working correctly. I find in early tests that when training with just MSE, the small model architecture would simply learn the mean of the training set, which is not useful. I solved this by using the huber loss instead. 

Author: Daniel Rugge (2025), assisted by Claude Sonnet 3.5
''' 

import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import os
from pathlib import Path
import onnxruntime as ort
from data_handling_raw import DangerDataset
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
import time

def load_onnx_model(onnx_path):
    """
    Load an ONNX model for inference
    
    Args:
        onnx_path: Path to the ONNX model file
        
    Returns:
        session: ONNX Runtime inference session
    """
    # Check if CUDA is available for ONNX Runtime
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get model input name
    input_name = session.get_inputs()[0].name
    
    # Get expected input shape
    input_shape = session.get_inputs()[0].shape
    input_size = input_shape[2] if len(input_shape) == 4 else 120  # Default to 120 if not specified
    
    print(f"Loaded ONNX model from {onnx_path}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    return session, input_name, input_size

def compare_true_vs_predicted_onnx(session, input_name, input_size, dataset, num_samples=5, indices=None):
    """
    Compare true danger values with predictions from the ONNX model
    
    Args:
        session: ONNX Runtime session
        input_name: Input tensor name for the model
        input_size: Expected input size (height/width)
        dataset: DangerDataset instance
        num_samples: Number of samples to visualize (ignored if indices is provided)
        indices: Specific indices to visualize (optional)
        
    Returns:
        None (displays plots)
    """
    # Randomly select indices if not provided
    if indices is None:
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # Create a custom colormap for danger levels
    danger_cmap = LinearSegmentedColormap.from_list('danger', 
                                                   [(0, 'green'), 
                                                    (0.5, 'yellow'), 
                                                    (0.75, 'orange'),
                                                    (1, 'red')])
    
    for idx in indices:
        # Get the image and true danger values
        image_tensor, true_values = dataset[idx]
        
        # Convert image tensor to numpy for visualization
        # The images are in YUV format
        img_yuv = image_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize
        img_yuv_display = img_yuv.copy()
        img_yuv_display[..., 0] = img_yuv_display[..., 0] * 255.0  # Y channel
        img_yuv_display[..., 1:] = (img_yuv_display[..., 1:] + 0.5) * 255.0  # U/V channels
        img_yuv_display = img_yuv_display.astype(np.uint8)
        
        # Convert YUV to BGR to RGB for display
        img_bgr = cv2.cvtColor(img_yuv_display, cv2.COLOR_YUV2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare input for ONNX model
        # Add batch dimension and ensure correct format
        input_data = np.expand_dims(image_tensor.numpy(), axis=0).astype(np.float32)
        
        # Get predictions from ONNX model
        predictions = session.run(None, {input_name: input_data})[0][0]
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
        
        # Image with row divisions (instead of column divisions)
        ax_img = plt.subplot(gs[0, 0])
        ax_img.imshow(img_rgb)
        ax_img.set_title('Image with Row Regions')
        
        # Add row dividers (instead of column dividers)
        h, w = img_rgb.shape[:2]
        num_rows = len(true_values)
        row_height = h // num_rows
        
        for i in range(1, num_rows):
            ax_img.axhline(y=i*row_height, color='white', linestyle='--', alpha=0.7)
        
        # Add row numbers (instead of column numbers)
        for i in range(num_rows):
            center_y = (i + 0.5) * row_height
            ax_img.text(20, center_y, f"{i+1}", color='white', 
                      fontsize=12, ha='left', va='center',
                      bbox=dict(facecolor='black', alpha=0.7, pad=3))
        
        ax_img.axis('off')
        
        # True danger values visualization
        ax_true = plt.subplot(gs[0, 1])
        for i, val in enumerate(true_values):
            color = danger_cmap(val)
            ax_true.barh(i, 1, color=color)
            ax_true.text(0.5, i, f"{val:.2f}", ha='center', va='center', color='black' if val < 0.5 else 'white')
        
        ax_true.set_yticks(range(len(true_values)))
        ax_true.set_yticklabels([f"Row {i+1}" for i in range(len(true_values))])
        ax_true.set_xlim(0, 1)
        ax_true.set_title('True Danger Values')
        
        # Predicted danger values visualization
        ax_pred = plt.subplot(gs[0, 2])
        for i, val in enumerate(predictions):
            # Clip predicted values to 0-1 range for visualization
            val_clipped = np.clip(val, 0, 1)
            color = danger_cmap(val_clipped)
            ax_pred.barh(i, 1, color=color)
            ax_pred.text(0.5, i, f"{val:.2f}", ha='center', va='center', color='black' if val_clipped < 0.5 else 'white')
        
        ax_pred.set_yticks(range(len(predictions)))
        ax_pred.set_yticklabels([f"Row {i+1}" for i in range(len(predictions))])
        ax_pred.set_xlim(0, 1)
        ax_pred.set_title('Predicted Danger Values')
        
        # Comparison chart
        ax_comp = plt.subplot(gs[1, :])
        bar_positions = np.arange(len(true_values))
        bar_width = 0.35
        
        # True values
        ax_comp.bar(bar_positions - bar_width/2, true_values, bar_width, label='True', alpha=0.7)
        
        # Predicted values
        ax_comp.bar(bar_positions + bar_width/2, predictions, bar_width, label='Predicted', alpha=0.7)
        
        # Calculate errors
        errors = np.abs(predictions - true_values.numpy())
        mse = np.mean(np.square(predictions - true_values.numpy()))
        mae = np.mean(errors)
        
        ax_comp.set_xticks(bar_positions)
        ax_comp.set_xticklabels([f"Row {i+1}" for i in range(len(true_values))])
        ax_comp.set_ylabel('Danger Value')
        ax_comp.set_title(f'Comparison (MSE: {mse:.4f}, MAE: {mae:.4f})')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def batch_predict_and_compare_onnx(session, input_name, input_size, dataset, output_dir, batch_size=16, num_batches=5):
    """
    Run predictions on multiple batches and save comparison images using ONNX model
    
    Args:
        session: ONNX Runtime session
        input_name: Input tensor name for the model
        input_size: Expected input size (height/width)
        dataset: DangerDataset instance
        output_dir: Directory to save output images
        batch_size: Batch size for inference
        num_batches: Number of batches to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create a custom colormap for danger levels
    danger_cmap = LinearSegmentedColormap.from_list('danger', 
                                                   [(0, 'green'), 
                                                    (0.5, 'yellow'), 
                                                    (0.75, 'orange'),
                                                    (1, 'red')])
    
    # Metrics accumulation
    all_mse = []
    all_mae = []
    
    for batch_idx, (images, true_values) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        # Convert to numpy for ONNX inference
        images_np = images.numpy().astype(np.float32)
        
        # Process each sample in batch
        for i in range(len(images)):
            # Get predictions for this sample
            input_data = np.expand_dims(images_np[i], axis=0)
            predictions = session.run(None, {input_name: input_data})[0][0]
            
            # Get the image
            img_yuv = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
            
            # Denormalize
            img_yuv_display = img_yuv.copy()
            img_yuv_display[..., 0] = img_yuv_display[..., 0] * 255.0  # Y channel
            img_yuv_display[..., 1:] = (img_yuv_display[..., 1:] + 0.5) * 255.0  # U/V channels
            img_yuv_display = img_yuv_display.astype(np.uint8)
            
            # Convert YUV to BGR to RGB for display
            img_bgr = cv2.cvtColor(img_yuv_display, cv2.COLOR_YUV2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Get true values for this sample
            sample_true = true_values[i].numpy()
            sample_pred = predictions
            
            # Calculate errors
            errors = np.abs(sample_pred - sample_true)
            mse = np.mean(np.square(sample_pred - sample_true))
            mae = np.mean(errors)
            
            all_mse.append(mse)
            all_mae.append(mae)
            
            # Create figure
            fig = plt.figure(figsize=(14, 7))
            gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
            
            # Image with row divisions (instead of column divisions)
            ax_img = plt.subplot(gs[0, 0])
            ax_img.imshow(img_rgb)
            ax_img.set_title('Image with Row Regions')
            
            # Add row dividers (instead of column dividers)
            h, w = img_rgb.shape[:2]
            num_rows = len(sample_true)
            row_height = h // num_rows
            
            for j in range(1, num_rows):
                ax_img.axhline(y=j*row_height, color='white', linestyle='--', alpha=0.7)
            
            # Add row numbers (instead of column numbers)
            for j in range(num_rows):
                center_y = (j + 0.5) * row_height
                ax_img.text(20, center_y, f"{j+1}", color='white', 
                          fontsize=12, ha='left', va='center',
                          bbox=dict(facecolor='black', alpha=0.7, pad=3))
            
            ax_img.axis('off')
            
            # True danger values visualization
            ax_true = plt.subplot(gs[0, 1])
            for j, val in enumerate(sample_true):
                color = danger_cmap(val)
                ax_true.barh(j, 1, color=color)
                ax_true.text(0.5, j, f"{val:.2f}", ha='center', va='center', color='black' if val < 0.5 else 'white')
            
            ax_true.set_yticks(range(len(sample_true)))
            ax_true.set_yticklabels([f"Row {j+1}" for j in range(len(sample_true))])
            ax_true.set_xlim(0, 1)
            ax_true.set_title('True Danger Values')
            
            # Predicted danger values visualization
            ax_pred = plt.subplot(gs[0, 2])
            for j, val in enumerate(sample_pred):
                # Clip predicted values to 0-1 range for visualization
                val_clipped = np.clip(val, 0, 1)
                color = danger_cmap(val_clipped)
                ax_pred.barh(j, 1, color=color)
                ax_pred.text(0.5, j, f"{val:.2f}", ha='center', va='center', color='black' if val_clipped < 0.5 else 'white')
            
            ax_pred.set_yticks(range(len(sample_pred)))
            ax_pred.set_yticklabels([f"Row {j+1}" for j in range(len(sample_pred))])
            ax_pred.set_xlim(0, 1)
            ax_pred.set_title('Predicted Danger Values')
            
            # Comparison chart
            ax_comp = plt.subplot(gs[1, :])
            bar_positions = np.arange(len(sample_true))
            bar_width = 0.35
            
            # True values
            ax_comp.bar(bar_positions - bar_width/2, sample_true, bar_width, label='True', alpha=0.7)
            
            # Predicted values
            ax_comp.bar(bar_positions + bar_width/2, sample_pred, bar_width, label='Predicted', alpha=0.7)
            
            ax_comp.set_xticks(bar_positions)
            ax_comp.set_xticklabels([f"Row {j+1}" for j in range(len(sample_true))])
            ax_comp.set_ylabel('Danger Value')
            ax_comp.set_title(f'Comparison (MSE: {mse:.4f}, MAE: {mae:.4f})')
            ax_comp.legend()
            ax_comp.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            filename = f"batch_{batch_idx}_sample_{i}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close(fig)
            
    # Print overall metrics
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    print(f"Average MSE across all samples: {avg_mse:.4f}")
    print(f"Average MAE across all samples: {avg_mae:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Total samples: {len(all_mse)}\n")

def create_inference_dashboard_onnx(session, input_name, input_size, dataset, output_path, num_samples=20):
    """
    Create a comprehensive dashboard with model predictions vs true values using ONNX model
    
    Args:
        session: ONNX Runtime session
        input_name: Input tensor name for the model
        input_size: Expected input size (height/width)
        dataset: DangerDataset instance
        output_path: Path to save the HTML dashboard
        num_samples: Number of random samples to include
    """
    # Select random samples
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # HTML header
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Obstacle Detection Model Predictions</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #333;
                color: white;
                padding: 15px;
                text-align: center;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .sample-container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 15px;
                overflow: hidden;
            }
            .image-container {
                width: 60%;
                float: left;
            }
            .values-container {
                width: 40%;
                float: left;
                padding: 10px;
            }
            img {
                width: 100%;
                border-radius: 5px;
            }
            .meter {
                height: 30px;
                position: relative;
                background: #f3f3f3;
                border-radius: 25px;
                padding: 5px;
                margin-bottom: 10px;
            }
            .meter > span {
                display: block;
                height: 100%;
                border-radius: 20px;
                position: relative;
                overflow: hidden;
            }
            .label {
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                text-align: center;
                font-weight: bold;
                line-height: 30px;
                color: black;
                mix-blend-mode: difference;
            }
            .true-bar {
                background-color: #4CAF50;
            }
            .pred-bar {
                background-color: #2196F3;
            }
            .error-metrics {
                margin-top: 10px;
                font-weight: bold;
                text-align: center;
            }
            .clearfix::after {
                content: "";
                clear: both;
                display: table;
            }
            .row-label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .values-title {
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Obstacle Detection Model Predictions</h1>
            <p>Comparing true danger values with model predictions</p>
        </div>
    """
    
    # Process each sample
    all_mse = []
    all_mae = []
    
    for idx in indices:
        # Get the image and true danger values
        image_tensor, true_values = dataset[idx]
        
        # Convert to numpy for ONNX inference
        input_data = np.expand_dims(image_tensor.numpy(), axis=0).astype(np.float32)
        
        # Get predictions from ONNX model
        predictions = session.run(None, {input_name: input_data})[0][0]
        
        # Convert image tensor to numpy for visualization
        img_yuv = image_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize
        img_yuv_display = img_yuv.copy()
        img_yuv_display[..., 0] = img_yuv_display[..., 0] * 255.0  # Y channel
        img_yuv_display[..., 1:] = (img_yuv_display[..., 1:] + 0.5) * 255.0  # U/V channels
        img_yuv_display = img_yuv_display.astype(np.uint8)
        
        # Convert YUV to BGR to RGB for display
        img_bgr = cv2.cvtColor(img_yuv_display, cv2.COLOR_YUV2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Calculate metrics
        true_np = true_values.numpy()
        mse = np.mean(np.square(predictions - true_np))
        mae = np.mean(np.abs(predictions - true_np))
        all_mse.append(mse)
        all_mae.append(mae)
        
        # Save annotated image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_rgb)
        
        # Add row dividers (instead of column dividers)
        h, w = img_rgb.shape[:2]
        num_rows = len(true_np)
        row_height = h // num_rows
        
        # Add row dividers
        for i in range(1, num_rows):
            ax.axhline(y=i*row_height, color='white', linestyle='--', alpha=0.7)
        
        # Add row numbers
        for i in range(num_rows):
            center_y = (i + 0.5) * row_height
            ax.text(20, center_y, f"{i+1}", color='white', 
                  fontsize=12, ha='left', va='center',
                  bbox=dict(facecolor='black', alpha=0.7, pad=3))
        
        ax.axis('off')
        
        # Save annotated image
        annotated_img_filename = f"sample_{idx}_annotated.jpg"
        annotated_img_path = os.path.join(os.path.dirname(output_path), annotated_img_filename)
        plt.savefig(annotated_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Add this sample to HTML content
        html_content += f"""
        <div class="sample-container clearfix">
            <div class="image-container">
                <img src="{annotated_img_filename}" alt="Sample {idx}">
            </div>
            <div class="values-container">
                <div class="values-title">Sample {idx} - MSE: {mse:.4f}, MAE: {mae:.4f}</div>
        """
        
        # Add danger value meters for each row
        for i, (true_val, pred_val) in enumerate(zip(true_np, predictions)):
            # Calculate color based on value (green to red)
            true_color = get_danger_color_css(true_val)
            pred_val_clipped = np.clip(pred_val, 0, 1)  # Clip for color calculation
            pred_color = get_danger_color_css(pred_val_clipped)
            
            html_content += f"""
                <div class="row-label">Row {i+1}</div>
                <div class="meter">
                    <span class="true-bar" style="width: {true_val*100}%; background-color: {true_color};">
                        <span class="label">True: {true_val:.2f}</span>
                    </span>
                </div>
                <div class="meter">
                    <span class="pred-bar" style="width: {min(pred_val*100, 100)}%; background-color: {pred_color};">
                        <span class="label">Pred: {pred_val:.2f}</span>
                    </span>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    # Add overall metrics
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    
    html_content += f"""
        <div class="header">
            <h2>Overall Metrics</h2>
            <p>Average MSE: {avg_mse:.4f} | Average MAE: {avg_mae:.4f} | Number of Samples: {len(all_mse)}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard created at {output_path}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

def get_danger_color_css(value):
    """Generate a CSS color for a danger value from green to red"""
    if value < 0.33:
        # Green to yellow
        r = int(255 * (value * 3))
        g = 255
        b = 0
    elif value < 0.66:
        # Yellow to orange
        r = 255
        g = int(255 * (2 - value * 3))
        b = 0
    else:
        # Orange to red
        r = 255
        g = int(127 * (3 - value * 3))
        b = 0
    
    return f"rgb({r}, {g}, {b})"




def main():
    h5_path = "preprocessed_labels\\danger_values.h5"  # H5 file with preprocessed images and danger values
    onnx_path = "promising_models_3\downscale_4\model_5.onnx"  # Best model ONNX file
    output_dir = "prediction_visualizations"  # Directory to save output
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ONNX model
    session, input_name, input_size = load_onnx_model(onnx_path)
    
    # Load dataset
    # Determine downscale factor based on input size (assuming original images are 240x240)
    downscale_factor = 240 // input_size
    dataset = DangerDataset(h5_path, downscale_factor=downscale_factor, training=False)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Compare a few samples interactively
    compare_true_vs_predicted_onnx(session, input_name, input_size, dataset, num_samples=5)

if __name__ == "__main__":
    main()