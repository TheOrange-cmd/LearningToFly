import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
import random

class FloorImageSeparator:
    def __init__(self):
        """Initialize the floor/non-floor image separator"""
        # Parameters for floor detection - expanded to include lighter greens
        self.hue_range_green = (35, 90)  # Expanded green hue range
        self.green_sat_threshold = 30  # Lowered saturation threshold for green
        
    def analyze_image(self, img_path, visualization=False):
        """
        Analyze an image to determine if it contains only floor or other elements
        
        Args:
            img_path: Path to the image
            visualization: Whether to show visualization
            
        Returns:
            Dictionary with analysis results
        """
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Could not load image"}
        
        # Convert to RGB (for visualization) and HSV (for analysis)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        hue_mean = np.mean(img_hsv[:, :, 0])
        hue_std = np.std(img_hsv[:, :, 0])
        sat_mean = np.mean(img_hsv[:, :, 1])
        val_mean = np.mean(img_hsv[:, :, 2])
        
        # Calculate hue histogram
        hist_hue = cv2.calcHist([img_hsv], [0], None, [36], [0, 180])
        hist_hue = hist_hue / np.sum(hist_hue)  # Normalize
        
        # Calculate dominant hue bin and value
        dominant_hue_bin = np.argmax(hist_hue)
        dominant_hue = (dominant_hue_bin * 5) + 2.5  # Center of the bin (180/36 = 5 degrees per bin)
        
        # Calculate percentage of green pixels
        # Green mask based on hue
        green_mask = cv2.inRange(img_hsv, 
                                np.array([self.hue_range_green[0], self.green_sat_threshold, 50]), 
                                np.array([self.hue_range_green[1], 255, 255]))
        green_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        # Calculate texture metrics
        # Use grayscale image for texture analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate gradient magnitude using Sobel
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient magnitude for visualization
        gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate texture metrics
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        
        # Calculate color consistency
        # Divide image into a 4x4 grid and calculate hue stats for each cell
        h, w = img.shape[:2]
        cell_h, cell_w = h // 4, w // 4
        
        hue_cell_means = []
        green_cell_percentages = []
        
        for i in range(4):
            for j in range(4):
                # Extract cell
                cell_hsv = img_hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                
                # Calculate stats
                hue_cell_means.append(np.mean(cell_hsv[:, :, 0]))
                
                # Green percentage in cell
                cell_green_mask = cv2.inRange(cell_hsv, 
                                           np.array([self.hue_range_green[0], self.green_sat_threshold, 50]), 
                                           np.array([self.hue_range_green[1], 255, 255]))
                green_cell_percentages.append(np.sum(cell_green_mask > 0) / (cell_green_mask.shape[0] * cell_green_mask.shape[1]))
        
        # Calculate consistency metrics
        hue_consistency = np.std(hue_cell_means)
        green_consistency = np.std(green_cell_percentages)
        
        # Determine if the image is pure floor
        is_floor = self._classify_floor(green_percentage, texture_mean, 
                                      hue_consistency, green_consistency,
                                      hue_mean, hue_std)
        
        # Calculate confidence score (higher value means more confident it's floor)
        confidence = self._calculate_confidence(green_percentage, texture_mean, 
                                             hue_consistency, green_consistency)
        
        # Prepare results
        result = {
            "is_floor": is_floor,
            "confidence": confidence,
            "green_percentage": green_percentage,
            "texture_mean": texture_mean,
            "texture_std": texture_std, 
            "hue_mean": hue_mean,
            "hue_std": hue_std,
            "dominant_hue": dominant_hue,
            "hue_consistency": hue_consistency,
            "green_consistency": green_consistency
        }
        
        # Visualization
        if visualization:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(green_mask, cmap='gray')
            plt.title(f"Green Mask ({green_percentage:.2%})")
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(gradient_norm, cmap='viridis')
            plt.title(f"Texture (mean: {texture_mean:.2f})")
            plt.axis('off')
            
            # Plot hue histogram
            plt.subplot(2, 3, 4)
            bin_edges = np.linspace(0, 180, 37)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.bar(bin_centers, hist_hue.flatten(), width=5)
            plt.axvline(x=dominant_hue, color='r', linestyle='--', label=f'Dominant: {dominant_hue:.1f}')
            plt.xlim(0, 180)
            plt.title(f"Hue Histogram (μ: {hue_mean:.1f}, σ: {hue_std:.1f})")
            plt.legend()
            
            # Plot grid analysis
            plt.subplot(2, 3, 5)
            grid = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    grid[i, j] = green_cell_percentages[i * 4 + j]
            
            plt.imshow(grid, cmap='YlGn', vmin=0, vmax=1)
            plt.title(f"Green % by Region (σ: {green_consistency:.3f})")
            for i in range(4):
                for j in range(4):
                    plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="black" if grid[i, j] > 0.5 else "white")
            plt.colorbar(shrink=0.8)
            
            # Display result
            plt.subplot(2, 3, 6)
            result_color = "green" if is_floor else "red"
            result_text = f"FLOOR (Confidence: {confidence:.2f})" if is_floor else f"NOT FLOOR (Confidence: {1-confidence:.2f})"
            plt.text(0.5, 0.5, result_text, 
                   ha='center', va='center', fontsize=20, color=result_color,
                   bbox=dict(facecolor='white', alpha=0.8))
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return result
    
    def _classify_floor(self, green_percentage, texture_mean, 
                      hue_consistency, green_consistency,
                      hue_mean, hue_std):
        """
        Classify an image as floor or not based on metrics
        
        This uses a rule-based approach with thresholds chosen based on floor characteristics
        """
        # Thresholds - adjusted based on feedback
        green_threshold = 0.45  # Lowered from 0.7 to 0.45 (45% green)
        texture_max = 30.0  # Increased from 25.0
        texture_min = 2.0   # Minimum texture (to avoid solid color images)
        hue_consistency_max = 15.0  # Maximum standard deviation of region hues
        green_consistency_max = 0.15  # Maximum standard deviation of region green percentages
        
        # Pure floor should have a green hue - expanded range
        hue_in_range = 40 <= hue_mean <= 85
        
        # Classification logic
        is_floor = (
            green_percentage >= green_threshold and
            texture_mean <= texture_max and
            texture_mean >= texture_min and
            hue_consistency <= hue_consistency_max and
            green_consistency <= green_consistency_max and
            hue_in_range
        )
        
        return is_floor
    
    def _calculate_confidence(self, green_percentage, texture_mean, 
                            hue_consistency, green_consistency):
        """
        Calculate confidence score for floor classification
        
        Returns a value between 0 and 1, where higher values indicate
        higher confidence that the image is floor
        """
        # Normalize each metric to a 0-1 scale where 1 is more floor-like
        # Adjusted to match new thresholds
        norm_green = min(1.0, green_percentage / 0.45)
        
        # For texture, we want a bell curve with highest confidence in the middle range
        # since floor has some texture but not too much
        texture_factor = max(0, 1.0 - abs(texture_mean - 12) / 18)
        
        # For consistency, lower values are better (more consistent)
        norm_hue_consistency = max(0, 1.0 - (hue_consistency / 15.0))
        norm_green_consistency = max(0, 1.0 - (green_consistency / 0.15))
        
        # Combine metrics with weights
        confidence = (
            norm_green * 0.4 +
            texture_factor * 0.3 +
            norm_hue_consistency * 0.15 +
            norm_green_consistency * 0.15
        )
        
        return confidence
    
    def process_folder(self, folder_path, max_images=None, visualization_samples=0):
        """
        Process all images in a folder and separate into floor and non-floor
        
        Args:
            folder_path: Path to the folder containing images
            max_images: Maximum number of images to process (None for all)
            visualization_samples: Number of samples to visualize from each group
            
        Returns:
            Dictionary with sorted image paths and statistics
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f)) and 
                      os.path.splitext(f.lower())[1] in image_extensions]
        
        # Random sample if needed
        if max_images and len(image_files) > max_images:
            random.seed(42)
            image_files = random.sample(image_files, max_images)
        
        # Process each image
        floor_images = []
        non_floor_images = []
        floor_confidences = []
        all_metrics = []
        error_count = 0
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Analyze image
                result = self.analyze_image(img_path)
                
                if "error" in result:
                    print(f"Error loading {img_file}: {result['error']}")
                    error_count += 1
                    continue
                
                # Store metrics for clustering
                metrics = [
                    result["green_percentage"],
                    result["texture_mean"],
                    result["hue_consistency"],
                    result["green_consistency"]
                ]
                all_metrics.append(metrics)
                
                # Sort into groups
                if result["is_floor"]:
                    floor_images.append(img_path)
                    floor_confidences.append(result["confidence"])
                else:
                    non_floor_images.append(img_path)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                error_count += 1
        
        if floor_images:
            # Sort floor images by confidence (descending)
            floor_with_conf = sorted(zip(floor_images, floor_confidences), 
                                   key=lambda x: x[1], reverse=True)
            floor_images = [img for img, conf in floor_with_conf]
            floor_confidences = [conf for img, conf in floor_with_conf]
        
        # Print summary
        print("\nClassification Results:")
        print(f"Total images processed: {len(floor_images) + len(non_floor_images)}")
        print(f"Floor images: {len(floor_images)}")
        print(f"Non-floor images: {len(non_floor_images)}")
        print(f"Images with errors: {error_count}")
        
        # Visualize samples
        if visualization_samples > 0:
            self._visualize_samples(floor_images, "Floor", 
                                  min(visualization_samples, len(floor_images)))
            self._visualize_samples(non_floor_images, "Non-Floor", 
                                  min(visualization_samples, len(non_floor_images)))
            
            # Also visualize metrics distribution
            self._visualize_metrics(all_metrics, floor_images, non_floor_images)
        
        return {
            "floor_images": floor_images,
            "non_floor_images": non_floor_images,
            "floor_confidences": floor_confidences,
            "error_count": error_count
        }
    
    def _visualize_samples(self, image_paths, group_name, n_samples):
        """Visualize sample images from a group"""
        if not image_paths:
            print(f"No {group_name} images to visualize")
            return
            
        # Select samples evenly from the list
        if len(image_paths) <= n_samples:
            samples = image_paths
        else:
            indices = np.linspace(0, len(image_paths)-1, n_samples, dtype=int)
            samples = [image_paths[i] for i in indices]
        
        # Calculate grid dimensions
        cols = min(5, n_samples)
        rows = (n_samples + cols - 1) // cols
        
        plt.figure(figsize=(15, 3 * rows))
        
        for i, img_path in enumerate(samples):
            plt.subplot(rows, cols, i + 1)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.imshow(img)
            plt.title(f"{group_name} sample {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Also display detailed analysis for some samples
        for i, img_path in enumerate(samples[:min(3, len(samples))]):
            print(f"\nDetailed analysis for {group_name} sample {i+1}:")
            self.analyze_image(img_path, visualization=True)
    
    def _visualize_metrics(self, all_metrics, floor_images, non_floor_images):
        """Visualize the distribution of metrics and clustering"""
        if not all_metrics:
            print("No metrics to visualize")
            return
            
        all_metrics = np.array(all_metrics)
        
        # Create labels for visualization
        if len(floor_images) + len(non_floor_images) == 0:
            print("No images to visualize metrics for")
            return
            
        labels = np.zeros(len(all_metrics))
        
        # Get the paths of all images that were processed successfully
        all_processed_images = floor_images + non_floor_images
        
        # Find which indices in all_metrics correspond to floor images
        floor_indices = []
        for i, (metrics, img_path) in enumerate(zip(all_metrics, all_processed_images)):
            if img_path in floor_images:
                floor_indices.append(i)
        
        labels[floor_indices] = 1
        
        # 2D scatter plots of metric pairs
        plt.figure(figsize=(15, 10))
        
        # Green percentage vs Texture
        plt.subplot(2, 2, 1)
        plt.scatter(all_metrics[:, 0], all_metrics[:, 1], 
                  c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('Green Percentage')
        plt.ylabel('Texture Mean')
        plt.title('Green Percentage vs Texture')
        plt.colorbar(label='Floor (1) / Non-Floor (0)')
        
        # Green percentage vs Hue Consistency
        plt.subplot(2, 2, 2)
        plt.scatter(all_metrics[:, 0], all_metrics[:, 2], 
                  c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('Green Percentage')
        plt.ylabel('Hue Consistency')
        plt.title('Green Percentage vs Hue Consistency')
        plt.colorbar(label='Floor (1) / Non-Floor (0)')
        
        # Texture vs Green Consistency
        plt.subplot(2, 2, 3)
        plt.scatter(all_metrics[:, 1], all_metrics[:, 3], 
                  c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('Texture Mean')
        plt.ylabel('Green Consistency')
        plt.title('Texture vs Green Consistency')
        plt.colorbar(label='Floor (1) / Non-Floor (0)')
        
        # Hue Consistency vs Green Consistency
        plt.subplot(2, 2, 4)
        plt.scatter(all_metrics[:, 2], all_metrics[:, 3], 
                  c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('Hue Consistency')
        plt.ylabel('Green Consistency')
        plt.title('Hue Consistency vs Green Consistency')
        plt.colorbar(label='Floor (1) / Non-Floor (0)')
        
        plt.tight_layout()
        plt.show()
        
        # Try K-means clustering to see if it matches our rule-based approach
        if len(all_metrics) > 10:  # Only do clustering if we have enough data
            # Normalize the metrics
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_metrics = scaler.fit_transform(all_metrics)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_metrics)
            
            # Visualize the clustering results
            plt.figure(figsize=(15, 5))
            
            # Compare clustering with rule-based approach
            plt.subplot(1, 2, 1)
            plt.scatter(all_metrics[:, 0], all_metrics[:, 1], 
                      c=cluster_labels, cmap='coolwarm', alpha=0.7)
            plt.xlabel('Green Percentage')
            plt.ylabel('Texture Mean')
            plt.title('K-means Clustering')
            plt.colorbar(label='Cluster')
            
            plt.subplot(1, 2, 2)
            plt.scatter(all_metrics[:, 0], all_metrics[:, 1], 
                      c=labels, cmap='coolwarm', alpha=0.7)
            plt.xlabel('Green Percentage')
            plt.ylabel('Texture Mean')
            plt.title('Rule-based Classification')
            plt.colorbar(label='Floor (1) / Non-Floor (0)')
            
            plt.tight_layout()
            plt.show()
            
            # Calculate agreement between clustering and rule-based approach
            agreement = np.sum((cluster_labels == 0) == (labels == 0)) / len(labels)
            agreement = max(agreement, 1 - agreement)  # Handle cluster label flipping
            print(f"Agreement between K-means and rule-based approach: {agreement:.2%}")


if __name__ == "__main__":
    # Replace with your image folder
    image_folder = "./merged"

    # Create separator
    separator = FloorImageSeparator()

    # Process all images in folder
    results = separator.process_folder(
        image_folder,
        max_images=2000,  # Process up to 100 images
        visualization_samples=5  # Show 5 samples from each group
    )

    # Access the results
    floor_images = results["floor_images"]
    non_floor_images = results["non_floor_images"]

    print(f"\nFound {len(floor_images)} pure floor images and {len(non_floor_images)} non-floor images")