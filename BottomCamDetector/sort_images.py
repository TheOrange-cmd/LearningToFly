''' 
    Script to sort raw images into different folders using a GUI with downscaling.
'''

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path
from PIL import Image, ImageTk
import shutil
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

class ImageSorter:
    def __init__(self, raw_dir, width, height, downsample=False):
        self.raw_dir = Path(raw_dir).resolve()
        self.original_width = width
        self.original_height = height
        self.downsample = downsample
        
        # Set working dimensions
        self.width = width // 2 if downsample else width
        self.height = height // 2 if downsample else height
        
        # Create output directories
        self.output_dirs = {
            'folder1': self.raw_dir.parent / 'split/boundaries',
            'folder2': self.raw_dir.parent / 'split/confirmed_floor',
            'folder3': self.raw_dir.parent / 'split/unlabeled'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Get list of raw files
        print(f"Reading raw files from {self.raw_dir}")
        self.raw_files = list(self.raw_dir.glob('*.raw'))
        self.current_index = 0
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Raw Image Sorter")

        # Add image canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add keyboard bindings
        self.root.bind('1', lambda e: self.move_to_folder('folder1'))
        self.root.bind('2', lambda e: self.move_to_folder('folder2'))
        self.root.bind('3', lambda e: self.move_to_folder('folder3'))
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        
        # Create buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(btn_frame, text="boundaries", command=lambda: self.move_to_folder('folder1')).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="confirmed_floor", command=lambda: self.move_to_folder('folder2')).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="unlabeled", command=lambda: self.move_to_folder('folder3')).pack(side=tk.LEFT, expand=True)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(nav_frame, text="Previous (←)", command=self.prev_image).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next (→)", command=self.next_image).pack(side=tk.LEFT, expand=True)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(side=tk.BOTTOM)
        
        # Resolution info label
        resolution_text = f"Resolution: {self.width}x{self.height} (Downsampled)" if downsample else f"Resolution: {self.width}x{self.height}"
        self.resolution_label = ttk.Label(self.root, text=resolution_text)
        self.resolution_label.pack(side=tk.BOTTOM)


        # Store the PhotoImage reference
        self.photo = None
        
        self.update_display()
    
    def read_raw_image(self, filepath):
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape((self.original_height, self.original_width * 2))
            
            if self.downsample:
                # Downsample UYVY data
                downsampled = np.zeros((self.height, self.width * 2), dtype=np.uint8)
                
                # Downsample Y values
                y_vals = data[:, 1::2]
                y_downsampled = y_vals[::2, ::2]
                downsampled[:, 1::2] = y_downsampled
                
                # Downsample U and V values
                u_vals = data[:, 0::4]
                v_vals = data[:, 2::4]
                u_downsampled = u_vals[::2, ::2]
                v_downsampled = v_vals[::2, ::2]
                
                downsampled[:, 0::4] = u_downsampled
                downsampled[:, 2::4] = v_downsampled
                
                data = downsampled
            
            # Convert UYVY to BGR
            yuv = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Extract Y values
            yuv[:, :, 0] = data[:, 1::2]
            
            # Extract U and V values and upsample
            u = data[:, 0::4]
            v = data[:, 2::4]
            
            # Duplicate U and V values for adjacent pixels
            yuv[:, ::2, 1] = u
            yuv[:, 1::2, 1] = u
            yuv[:, ::2, 2] = v
            yuv[:, 1::2, 2] = v
            
            # Convert to BGR
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return bgr, data
    
    def update_display(self):
        if not self.raw_files:
            self.status_label.config(text="No more images to sort!")
            return
        
        current_file = self.raw_files[self.current_index]
        self.status_label.config(text=f"Image {self.current_index + 1} of {len(self.raw_files)}: {current_file.name}")
        
        # Read and display image
        img, _ = self.read_raw_image(current_file)
        
        # Convert BGR to RGB for PIL
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_img = Image.fromarray(rgb_img)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def move_to_folder(self, folder_name):
        if not self.raw_files:
            return
            
        current_file = self.raw_files[self.current_index]
        dest = self.output_dirs[folder_name] / current_file.name
        
        # If downsampling, save the downsampled raw data
        if self.downsample:
            _, raw_data = self.read_raw_image(current_file)
            raw_data.tofile(str(dest))
        else:
            shutil.move(str(current_file), str(dest))
        
        # Remove from list and update display
        self.raw_files.pop(self.current_index)
        if self.raw_files:
            self.current_index = min(self.current_index, len(self.raw_files) - 1)
            self.update_display()
        else:
            self.canvas.delete("all")
            self.status_label.config(text="No more images to sort!")
    
    def next_image(self):
        if self.raw_files and self.current_index < len(self.raw_files) - 1:
            self.current_index += 1
            self.update_display()
    
    def prev_image(self):
        if self.raw_files and self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def run(self):
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    raw_dir = "./drone_pictures"  
    # original dimensions
    width = 240  
    height = 240
    downsample = True  # Set to True to enable downsampling by factor of 2
    
    sorter = ImageSorter(raw_dir, width, height, downsample)
    sorter.run()