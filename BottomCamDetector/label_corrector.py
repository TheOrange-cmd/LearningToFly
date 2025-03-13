import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class GridImageVerifier:
    def __init__(self, image_paths, confidences, grid_size=7):
        self.root = tk.Tk()
        self.root.title("Floor Image Verifier - Grid View")
        
        # Store image data
        self.image_paths = image_paths
        self.confidences = confidences
        self.grid_size = grid_size
        self.images_per_page = grid_size * grid_size
        self.current_page = 0
        self.total_pages = (len(image_paths) + self.images_per_page - 1) // self.images_per_page
        self.classifications = {path: True for path in image_paths}  # True = floor
        
        # Calculate thumbnail size - using smaller default size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.thumbnail_size = min(100, screen_width // (self.grid_size + 2), 
                                (screen_height - 100) // (self.grid_size + 1))
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Grid frame for images
        self.grid_frame = ttk.Frame(main_frame)
        self.grid_frame.pack(expand=True, fill='both')
        
        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(pady=5)
        
        ttk.Button(controls_frame, text="Previous Page", command=self.prev_page).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Next Page", command=self.next_page).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Save & Exit", command=self.save_and_exit).pack(side='left', padx=5)
        
        # Status
        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var).pack(pady=5)
        
        # Load first page
        self.update_page()
        
    def update_page(self):
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.image_paths))
        
        self.photo_refs = []  # Keep references to prevent garbage collection
        self.frames = {}  # Keep track of frames
        
        for i in range(start_idx, end_idx):
            row = (i - start_idx) // self.grid_size
            col = (i - start_idx) % self.grid_size
            
            img_path = self.image_paths[i]
            conf = self.confidences[i]
            is_floor = self.classifications[img_path]
            
            # Create frame for each image
            frame = tk.Frame(self.grid_frame, borderwidth=2, relief='solid')
            frame.grid(row=row, column=col, padx=1, pady=1)
            self.frames[img_path] = frame
            
            # Load and resize image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            scale = min(self.thumbnail_size/w, self.thumbnail_size/h)
            new_size = (int(w*scale), int(h*scale))
            img = cv2.resize(img, new_size)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.photo_refs.append(photo)
            
            # Create label with image
            label = tk.Label(frame, image=photo)
            label.pack()
            
            # Add confidence label (smaller font)
            tk.Label(frame, text=f"{conf:.2f}", font=('Arial', 8)).pack()
            
            # Set initial color
            frame.configure(bg='green' if is_floor else 'red')
            
            # Bind click event to both label and frame
            label.bind('<Button-1>', lambda e, path=img_path: self.toggle_classification(path))
            frame.bind('<Button-1>', lambda e, path=img_path: self.toggle_classification(path))
            
        # Update status
        self.status_var.set(f"Page {self.current_page + 1} of {self.total_pages} | "
                           f"Images {start_idx + 1}-{end_idx} of {len(self.image_paths)}")
    
    def toggle_classification(self, img_path):
        self.classifications[img_path] = not self.classifications[img_path]
        # Update the frame color immediately
        frame = self.frames[img_path]
        frame.configure(bg='green' if self.classifications[img_path] else 'red')
        self.root.update()
    
    def next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page()
    
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page()
    
    def save_and_exit(self):
        floor_images = [path for path, is_floor in self.classifications.items() if is_floor]
        not_floor_images = [path for path, is_floor in self.classifications.items() if not is_floor]
        
        print("\nResults:")
        print(f"Floor images: {len(floor_images)}")
        print(f"Not floor images: {len(not_floor_images)}")
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()
        return self.classifications

def verify_floor_images(floor_images, confidences):
    verifier = GridImageVerifier(floor_images, confidences)
    return verifier.run()


def save_and_exit(self):
    import shutil
    import os

    # Create output directories
    output_base = "verified_images"
    confirmed_floor_dir = os.path.join(output_base, "confirmed_floor")
    unlabeled_dir = os.path.join(output_base, "unlabeled")
    
    os.makedirs(confirmed_floor_dir, exist_ok=True)
    os.makedirs(unlabeled_dir, exist_ok=True)

    # Separate images
    floor_images = [path for path, is_floor in self.classifications.items() if is_floor]
    not_floor_images = [path for path, is_floor in self.classifications.items() if not is_floor]
    
    # Copy files to appropriate directories
    for img_path in floor_images:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(confirmed_floor_dir, filename))
    
    for img_path in not_floor_images:
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(unlabeled_dir, filename))
    
    print("\nResults:")
    print(f"Confirmed floor images copied to: {confirmed_floor_dir}")
    print(f"Unlabeled images copied to: {unlabeled_dir}")
    print(f"Floor images: {len(floor_images)}")
    print(f"Unlabeled images: {len(not_floor_images)}")
    
    self.root.quit()
    self.root.destroy()

# Usage
if __name__ == "__main__":
    from rough_split import FloorImageSeparator
    import os
    import shutil

    # Process images with separator
    separator = FloorImageSeparator()
    results = separator.process_folder(
        "./BottomCamDetector/merged",
        max_images=2000,
        visualization_samples=0
    )

    # Connect to verifier
    floor_images = results["floor_images"]
    floor_confidences = results["floor_confidences"]

    # Run manual verification
    classifications = verify_floor_images(floor_images, results["floor_confidences"])
    
    # Create output directories
    output_base = "verified_images"
    confirmed_floor_dir = os.path.join(output_base, "confirmed_floor")
    unlabeled_dir = os.path.join(output_base, "unlabeled")
    
    os.makedirs(confirmed_floor_dir, exist_ok=True)
    os.makedirs(unlabeled_dir, exist_ok=True)

    # Copy files to appropriate directories
    for img_path, is_floor in classifications.items():
        filename = os.path.basename(img_path)
        if is_floor:
            shutil.copy2(img_path, os.path.join(confirmed_floor_dir, filename))
        else:
            shutil.copy2(img_path, os.path.join(unlabeled_dir, filename))

    # Print results
    print(f"\nFinal Results:")
    print(f"Images copied to: {output_base}/")
    print(f"Floor images: {len([x for x in classifications.values() if x])} (in confirmed_floor/)")
    print(f"Unlabeled images: {len([x for x in classifications.values() if not x])} (in unlabeled/)")