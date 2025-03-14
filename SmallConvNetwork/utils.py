import matplotlib.pyplot as plt
import torch

def convert_coordinates(bbox, width, height):
    x1 = bbox["x"] - bbox["w"]/2 if bbox["x"] - bbox["w"]/2 > 0 else 0
    y1 = bbox["y"] - bbox["h"]/2 if bbox["y"] - bbox["h"]/2 > 0 else 0
    x2 = bbox["x"] + bbox["w"]/2 if bbox["x"] + bbox["w"]/2 < width else width
    y2 = bbox["y"] + bbox["h"]/2 if bbox["y"] + bbox["h"]/2 < height else height
    return x1, x2, y1, y2

def danger_level(x1, x2, y1, y2, label, width, height):
    x1_rel = x1 / width
    x2_rel = x2 / width
    y1_rel = y1 / height
    y2_rel = y2 / height
    
    # Calculate side distance
    if x1_rel >= 0.8:
        side_distance = (x1_rel - 0.8)
    elif x1_rel >= 0.6:
        side_distance = (x1_rel - 0.6) / 4
    elif x2_rel <= 0.6:
        side_distance = (0.6 - x2_rel) / 2
    elif x2_rel <= 0.2:
        side_distance = (0.6 - x2_rel) * 2
    else:
        side_distance = 0

    if label == 0:
        width_distance = (x2_rel - x1_rel) / 0.25
        height_distance = min((y2_rel - y1_rel) / 0.8, 0.7)
        multiplier = 1.5 if side_distance > 0.25 else 1
        danger = min(max([width_distance, height_distance]) - side_distance * multiplier, 1)
    if label == 1:
        width_distance = (x2_rel - x1_rel) / 0.15
        height_distance = min((y2_rel - y1_rel) / 1.1, 0.7)
        multiplier = 1.5 if side_distance > 0.25 else 1
        danger = min(max([width_distance, height_distance]) - side_distance * multiplier, 1)
    
    return max(danger, 0)

class DangerLevelConfig:
    def __init__(self):
        # Levels must be in descending order of danger
        self.LEVEL_DEFINITIONS = [
            {'name': 'DANGER', 'level': 2, 'threshold': 0.7},
            {'name': 'WARNING', 'level': 1, 'threshold': 0.6},
            {'name': 'SAFE', 'level': 0, 'threshold': 0.0}
        ]
        
        # Create convenience dictionaries
        self.THRESHOLDS = {level['name']: level['threshold'] for level in self.LEVEL_DEFINITIONS}
        self.LEVELS = {level['name']: level['level'] for level in self.LEVEL_DEFINITIONS}
        
    def get_level_for_value(self, value):
        """Returns appropriate danger level for a given danger value."""
        for level in self.LEVEL_DEFINITIONS:
            if value >= level['threshold']:
                return level['level']
        return self.LEVELS['SAFE']  # Default to safe

def cell_contains_bbox(bboxes, grid_lines, width, height):
    """
    Assigns danger levels to grid cells based on bounding boxes.
    Empty cells are considered safe.
    """
    config = DangerLevelConfig()
    num_columns = len(grid_lines[0]) - 1
    num_rows = len(grid_lines[1]) - 1
    num_levels = len(config.LEVELS)
    
    # Track maximum danger value for each cell
    cell_max_danger = [[0 for _ in range(num_rows)] 
                        for _ in range(num_columns)]
    
    def get_danger_level(danger_value):
        if danger_value >= config.THRESHOLDS['DANGER']:
            return config.LEVELS['DANGER']
        elif danger_value >= config.THRESHOLDS['WARNING']:
            return config.LEVELS['WARNING']
        else:
            return config.LEVELS['SAFE']
    
    def box_intersects_cell(x1, x2, left, right):
        return ((left <= x1 <= right) or 
                (left <= x2 <= right) or 
                (x1 <= left and x2 >= right))
    
    def box_intersects_cell_y(y1, y2, bottom, top):
        return ((bottom <= y1 <= top) or 
                (bottom <= y2 <= top) or
                (y1 <= bottom and y2 >= top))
    
    # First pass: calculate maximum danger value for each cell
    for bbox in bboxes:
        x1, x2, y1, y2 = convert_coordinates(bbox, width, height)
        
        for i in range(num_columns):
            left, right = grid_lines[0][i], grid_lines[0][i + 1]
            
            if box_intersects_cell(x1, x2, left, right):
                for j in range(num_rows):
                    bottom, top = grid_lines[1][j], grid_lines[1][j + 1]
                    
                    if box_intersects_cell_y(y1, y2, bottom, top):
                        danger = danger_level(x1, x2, y1, y2, bbox["label"], width, height)
                        cell_max_danger[i][j] = max(cell_max_danger[i][j], danger)
    
    # Initialize grid with all cells as safe
    box_inside_grid = [[[0 for _ in range(num_rows)] 
                         for _ in range(num_columns)] 
                         for _ in range(num_levels)]
    
    # Set all cells in the SAFE level (index 0) to 1 by default
    for i in range(num_columns):
        for j in range(num_rows):
            box_inside_grid[config.LEVELS['SAFE']][i][j] = 1
    
    # Second pass: override safe labels with higher danger levels where applicable
    for i in range(num_columns):
        for j in range(num_rows):
            if cell_max_danger[i][j] > 0:
                level = get_danger_level(cell_max_danger[i][j])
                if level != config.LEVELS['SAFE']:
                    # Clear the safe label and set the higher danger level
                    box_inside_grid[config.LEVELS['SAFE']][i][j] = 0
                    box_inside_grid[level][i][j] = 1

    return box_inside_grid

def plot_image(sample_image, sample_danger, truth, grid_lines, grid=False, ax=None):

    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Display the sample image
    ax.imshow(sample_image.squeeze(0).permute(1, 2, 0))
    
    # Draw grid lines if enabled
    if grid:
        for grid_line_y in grid_lines[1]:
            ax.axhline(grid_line_y, color="blue", linestyle="--")
        for grid_line_x in grid_lines[0]:
            ax.axvline(grid_line_x, color="blue", linestyle="--")

    width = sample_image.shape[2]
    height = sample_image.shape[1]
    
    # Overlay predicted danger levels
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            # Prediction overlays (filled cells)
            if sample_danger[2][i][j] > 0.8:
                ax.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                               [grid_lines[1][j], grid_lines[1][j]],
                               [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                               color="red", alpha=0.5)
            elif sample_danger[1][i][j] > 0.8:
                ax.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                               [grid_lines[1][j], grid_lines[1][j]],
                               [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                               color="orange", alpha=0.5)
            elif sample_danger[0][i][j] > 0.8:
                ax.fill_between([grid_lines[0][i], grid_lines[0][i + 1]],
                               [grid_lines[1][j], grid_lines[1][j]],
                               [grid_lines[1][j + 1], grid_lines[1][j + 1]],
                               color="green", alpha=0.5)

    if truth is not None:
        for i in range(len(grid_lines[0]) - 1):
            for j in range(len(grid_lines[1]) - 1):
                if (truth[2][i][j] == 1):
                    if not sample_danger[2][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="red", facecolor="none", hatch="///"))
                elif (truth[1][i][j] == 1):
                    if not sample_danger[1][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="orange", facecolor="none", hatch="///"))
                elif (truth[0][i][j] == 1):
                    if not sample_danger[0][i][j] > 0.8:
                        ax.add_patch(plt.Rectangle(
                            (grid_lines[0][i], grid_lines[1][j]),
                            grid_lines[0][i + 1] - grid_lines[0][i],
                            grid_lines[1][j + 1] - grid_lines[1][j],
                            linewidth=3, edgecolor="green", facecolor="none", hatch="///"))
                elif (truth[0][i][j] == 0) and (sample_danger[0][i][j] > 0.8) or (sample_danger[1][i][j] > 0.8) or (sample_danger[2][i][j] > 0.8):
                    ax.add_patch(plt.Rectangle(
                        (grid_lines[0][i], grid_lines[1][j]),
                        grid_lines[0][i + 1] - grid_lines[0][i],
                        grid_lines[1][j + 1] - grid_lines[1][j],
                        linewidth=3, edgecolor="white", facecolor="none", hatch="///"))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    if ax is None:
        plt.show()
    
    return ax

def plot_side_by_side(image, ground_truth, prediction, grid_lines, example_num=None):
    """
    Plot ground truth and prediction side by side for easy comparison
    
    Args:
        image: Tensor of shape (C, H, W) or numpy array of shape (H, W, C)
        ground_truth: Tensor of shape (3, grid_x, grid_y)
        prediction: Tensor of shape (3, grid_x, grid_y)
        grid_lines: List of grid lines [x_lines, y_lines]
        example_num: Optional example number for the title
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert image tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
            image = image.permute(1, 2, 0).cpu().numpy()
            if image.shape[2] == 1:  # Grayscale
                image = image[:, :, 0]
    
    # Convert tensors to numpy arrays
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Plot ground truth (left)
    ax1.imshow(image)
    ax1.set_title("Ground Truth" if example_num is None else f"Example {example_num} - Ground Truth")
    
    # Plot prediction (right)
    ax2.imshow(image)
    ax2.set_title("Prediction" if example_num is None else f"Example {example_num} - Prediction")
    
    # Add grid lines to both plots
    for x in grid_lines[0]:
        ax1.axvline(x, color='blue', linestyle='--', alpha=0.3)
        ax2.axvline(x, color='blue', linestyle='--', alpha=0.3)
    for y in grid_lines[1]:
        ax1.axhline(y, color='blue', linestyle='--', alpha=0.3)
        ax2.axhline(y, color='blue', linestyle='--', alpha=0.3)
    
    # Define threshold for visualization
    threshold = 0.5
    
    # Draw cells on ground truth (left)
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            x1, x2 = grid_lines[0][i], grid_lines[0][i + 1]
            y1, y2 = grid_lines[1][j], grid_lines[1][j + 1]
            
            # Apply colors based on ground truth
            if ground_truth[2][i][j] > threshold:  # Danger (red)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='red', alpha=0.4)
            elif ground_truth[1][i][j] > threshold:  # Warning (orange)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='orange', alpha=0.4)
            elif ground_truth[0][i][j] > threshold:  # Safe (green)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='green', alpha=0.4)
    
    # Draw cells on prediction (right)
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            x1, x2 = grid_lines[0][i], grid_lines[0][i + 1]
            y1, y2 = grid_lines[1][j], grid_lines[1][j + 1]
            
            # Apply colors based on prediction
            if prediction[2][i][j] > threshold:  # Danger (red)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='red', alpha=0.4)
            elif prediction[1][i][j] > threshold:  # Warning (orange)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='orange', alpha=0.4)
            elif prediction[0][i][j] > threshold:  # Safe (green)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='green', alpha=0.4)
            
            # Highlight differences with outlines
            if ((ground_truth[2][i][j] > threshold) != (prediction[2][i][j] > threshold) or
                (ground_truth[1][i][j] > threshold) != (prediction[1][i][j] > threshold) or
                (ground_truth[0][i][j] > threshold) != (prediction[0][i][j] > threshold)):
                ax2.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'w-', linewidth=2)
    
    # Add legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='green', alpha=0.4, label='Safe'),
        mpatches.Patch(color='orange', alpha=0.4, label='Warning'),
        mpatches.Patch(color='red', alpha=0.4, label='Danger')
    ]
    handles.append(mpatches.Patch(edgecolor='white', facecolor='none', label='Difference'))
    
    # Add legend to both plots
    ax1.legend(handles=handles[:-1], loc='upper right', fontsize='small')
    ax2.legend(handles=handles, loc='upper right', fontsize='small')
    
    # Set the same limits for both plots
    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)  # Inverted y-axis for image coordinates
    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def compare_pytorch_onnx_side_by_side(image, ground_truth, pytorch_pred, onnx_pred, grid_lines, example_num=None):
    """
    Plot ground truth, PyTorch prediction, and ONNX prediction side by side
    
    Args:
        image: Tensor of shape (C, H, W) or numpy array of shape (H, W, C)
        ground_truth: Tensor of shape (3, grid_x, grid_y)
        pytorch_pred: Tensor of shape (3, grid_x, grid_y) from PyTorch model
        onnx_pred: Tensor of shape (3, grid_x, grid_y) from ONNX model
        grid_lines: List of grid lines [x_lines, y_lines]
        example_num: Optional example number for the title
    """
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Convert image tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
            image = image.permute(1, 2, 0).cpu().numpy()
            if image.shape[2] == 1:  # Grayscale
                image = image[:, :, 0]
    
    # Convert tensors to numpy arrays
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(pytorch_pred, torch.Tensor):
        pytorch_pred = pytorch_pred.cpu().numpy()
    if isinstance(onnx_pred, torch.Tensor):
        onnx_pred = onnx_pred.cpu().numpy()
    
    # Plot ground truth (left)
    ax1.imshow(image)
    ax1.set_title("Ground Truth" if example_num is None else f"Example {example_num} - Ground Truth")
    
    # Plot PyTorch prediction (middle)
    ax2.imshow(image)
    ax2.set_title("PyTorch Prediction" if example_num is None else f"Example {example_num} - PyTorch")
    
    # Plot ONNX prediction (right)
    ax3.imshow(image)
    ax3.set_title("ONNX Prediction" if example_num is None else f"Example {example_num} - ONNX")
    
    # Add grid lines to all plots
    for x in grid_lines[0]:
        ax1.axvline(x, color='blue', linestyle='--', alpha=0.3)
        ax2.axvline(x, color='blue', linestyle='--', alpha=0.3)
        ax3.axvline(x, color='blue', linestyle='--', alpha=0.3)
    for y in grid_lines[1]:
        ax1.axhline(y, color='blue', linestyle='--', alpha=0.3)
        ax2.axhline(y, color='blue', linestyle='--', alpha=0.3)
        ax3.axhline(y, color='blue', linestyle='--', alpha=0.3)
    
    # Define threshold for visualization
    threshold = 0.5
    
    # Draw cells on ground truth (left)
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            x1, x2 = grid_lines[0][i], grid_lines[0][i + 1]
            y1, y2 = grid_lines[1][j], grid_lines[1][j + 1]
            
            # Apply colors based on ground truth
            if ground_truth[2][i][j] > threshold:  # Danger (red)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='red', alpha=0.4)
            elif ground_truth[1][i][j] > threshold:  # Warning (orange)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='orange', alpha=0.4)
            elif ground_truth[0][i][j] > threshold:  # Safe (green)
                ax1.fill_between([x1, x2], [y1, y1], [y2, y2], color='green', alpha=0.4)
    
    # Draw cells on PyTorch prediction (middle)
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            x1, x2 = grid_lines[0][i], grid_lines[0][i + 1]
            y1, y2 = grid_lines[1][j], grid_lines[1][j + 1]
            
            # Apply colors based on prediction
            if pytorch_pred[2][i][j] > threshold:  # Danger (red)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='red', alpha=0.4)
            elif pytorch_pred[1][i][j] > threshold:  # Warning (orange)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='orange', alpha=0.4)
            elif pytorch_pred[0][i][j] > threshold:  # Safe (green)
                ax2.fill_between([x1, x2], [y1, y1], [y2, y2], color='green', alpha=0.4)
            
            # Highlight differences with ground truth
            if ((ground_truth[2][i][j] > threshold) != (pytorch_pred[2][i][j] > threshold) or
                (ground_truth[1][i][j] > threshold) != (pytorch_pred[1][i][j] > threshold) or
                (ground_truth[0][i][j] > threshold) != (pytorch_pred[0][i][j] > threshold)):
                ax2.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'w-', linewidth=2)
    
    # Draw cells on ONNX prediction (right)
    for i in range(len(grid_lines[0]) - 1):
        for j in range(len(grid_lines[1]) - 1):
            x1, x2 = grid_lines[0][i], grid_lines[0][i + 1]
            y1, y2 = grid_lines[1][j], grid_lines[1][j + 1]
            
            # Apply colors based on prediction
            if onnx_pred[2][i][j] > threshold:  # Danger (red)
                ax3.fill_between([x1, x2], [y1, y1], [y2, y2], color='red', alpha=0.4)
            elif onnx_pred[1][i][j] > threshold:  # Warning (orange)
                ax3.fill_between([x1, x2], [y1, y1], [y2, y2], color='orange', alpha=0.4)
            elif onnx_pred[0][i][j] > threshold:  # Safe (green)
                ax3.fill_between([x1, x2], [y1, y1], [y2, y2], color='green', alpha=0.4)
            
            # Highlight differences with ground truth
            if ((ground_truth[2][i][j] > threshold) != (onnx_pred[2][i][j] > threshold) or
                (ground_truth[1][i][j] > threshold) != (onnx_pred[1][i][j] > threshold) or
                (ground_truth[0][i][j] > threshold) != (onnx_pred[0][i][j] > threshold)):
                ax3.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'w-', linewidth=2)
            
            # Highlight differences between PyTorch and ONNX with dashed yellow outline
            if ((pytorch_pred[2][i][j] > threshold) != (onnx_pred[2][i][j] > threshold) or
                (pytorch_pred[1][i][j] > threshold) != (onnx_pred[1][i][j] > threshold) or
                (pytorch_pred[0][i][j] > threshold) != (onnx_pred[0][i][j] > threshold)):
                ax3.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'y--', linewidth=2)
    
    # Add legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    handles = [
        mpatches.Patch(color='green', alpha=0.4, label='Safe'),
        mpatches.Patch(color='orange', alpha=0.4, label='Warning'),
        mpatches.Patch(color='red', alpha=0.4, label='Danger'),
        mlines.Line2D([], [], color='white', linewidth=2, label='Diff with Ground Truth'),
        mlines.Line2D([], [], color='yellow', linestyle='--', linewidth=2, label='Diff PyTorch vs ONNX')
    ]
    
    # Add appropriate legends to each plot
    ax1.legend(handles=handles[:3], loc='upper right', fontsize='small')
    ax2.legend(handles=handles[:4], loc='upper right', fontsize='small')
    ax3.legend(handles=handles, loc='upper right', fontsize='small')
    
    # Set the same limits for all plots
    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)  # Inverted y-axis for image coordinates
    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)
    ax3.set_xlim(0, image.shape[1])
    ax3.set_ylim(image.shape[0], 0)
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)