import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class GridImageVerifier:
    def __init__(self, source_folder, destination_folder, grid_size=7):
        self.root = tk.Tk()
        self.root.title("Image Selector - Grid View")
        
        # Get image paths from source folder
        self.image_paths = [os.path.join(source_folder, f) for f in os.listdir(source_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create dummy confidences (we don't need them but keeping the structure)
        self.confidences = [1.0] * len(self.image_paths)
        
        # Store folders
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        os.makedirs(destination_folder, exist_ok=True)
        
        self.grid_size = grid_size
        self.images_per_page = grid_size * grid_size
        self.current_page = 0
        self.total_pages = (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page
        self.classifications = {path: False for path in self.image_paths}  # False = don't move
        
        # Calculate thumbnail size
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
        ttk.Button(controls_frame, text="Move Selected & Exit", command=self.save_and_exit).pack(side='left', padx=5)
        
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
            is_selected = self.classifications[img_path]
            
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
            frame.configure(bg='green' if is_selected else 'red')
            
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
        import shutil
        
        # Move selected images
        selected_images = [path for path, is_selected in self.classifications.items() if is_selected]
        
        # Move files to destination directory
        for img_path in selected_images:
            filename = os.path.basename(img_path)
            shutil.move(img_path, os.path.join(self.destination_folder, filename))
        
        print("\nResults:")
        print(f"Moved {len(selected_images)} images to: {self.destination_folder}")
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()



if __name__ == "__main__":
    verifier = GridImageVerifier("./BottomCamDetector/verified_images/unlabeled", "./BottomCamDetector/verified_images/boundaries")
    verifier.run()