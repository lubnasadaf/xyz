import pydicom
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def display_dicom_slices(dicom_path):
    """
    Display all slices from a multi-slice DICOM file with interactive navigation.
    
    Args:
        dicom_path: Path to the DICOM file
    """
    # Read the DICOM file
    ds = pydicom.dcmread(dicom_path)
    
    # Get pixel array (handles multi-frame DICOM)
    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
        pixel_array = ds.pixel_array
        num_slices = ds.NumberOfFrames
    else:
        pixel_array = ds.pixel_array
        # If single slice, add dimension
        if len(pixel_array.shape) == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        num_slices = pixel_array.shape[0]
    
    # Create interactive viewer
    fig, ax = plt.subplots(figsize=(10, 10))
    
    class SliceViewer:
        def __init__(self, data):
            self.data = data
            self.slices = data.shape[0]
            self.idx = 0
            self.im = ax.imshow(self.data[self.idx], cmap='gray')
            self.update()
            
        def on_scroll(self, event):
            if event.button == 'up':
                self.idx = (self.idx + 1) % self.slices
            else:
                self.idx = (self.idx - 1) % self.slices
            self.update()
            
        def on_key(self, event):
            if event.key == 'up' or event.key == 'right':
                self.idx = (self.idx + 1) % self.slices
            elif event.key == 'down' or event.key == 'left':
                self.idx = (self.idx - 1) % self.slices
            self.update()
            
        def update(self):
            self.im.set_data(self.data[self.idx])
            ax.set_title(f'Slice {self.idx + 1}/{self.slices}')
            fig.canvas.draw_idle()
    
    viewer = SliceViewer(pixel_array)
    fig.canvas.mpl_connect('scroll_event', viewer.on_scroll)
    fig.canvas.mpl_connect('key_press_event', viewer.on_key)
    
    plt.title(f'Slice 1/{num_slices}\nUse mouse scroll or arrow keys to navigate')
    plt.tight_layout()
    plt.show()


def display_dicom_grid(dicom_path, slices_per_row=4):
    """
    Display all slices in a grid layout.
    
    Args:
        dicom_path: Path to the DICOM file
        slices_per_row: Number of slices to display per row
    """
    ds = pydicom.dcmread(dicom_path)
    
    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
        pixel_array = ds.pixel_array
        num_slices = ds.NumberOfFrames
    else:
        pixel_array = ds.pixel_array
        if len(pixel_array.shape) == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        num_slices = pixel_array.shape[0]
    
    # Calculate grid dimensions
    num_rows = int(np.ceil(num_slices / slices_per_row))
    
    fig, axes = plt.subplots(num_rows, slices_per_row, figsize=(15, 3*num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_slices):
        row = i // slices_per_row
        col = i % slices_per_row
        axes[row, col].imshow(pixel_array[i], cmap='gray')
        axes[row, col].set_title(f'Slice {i+1}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, num_rows * slices_per_row):
        row = i // slices_per_row
        col = i % slices_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Replace with your DICOM file path
    dicom_file = "face_example.dcm"
    
    # Interactive viewer (scroll through slices)
    display_dicom_slices(dicom_file)
    
    # Or display all slices in a grid
    # display_dicom_grid(dicom_file, slices_per_row=5)