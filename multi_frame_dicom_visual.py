import pydicom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def view_multiframe_dicom(dicom_file):
    """View multi-frame DICOM file"""
    
    ds = pydicom.dcmread(dicom_file)
    
    print(f"Number of Frames: {ds.NumberOfFrames}")
    print(f"Dimensions: {ds.Rows} x {ds.Columns}")
    
    # Extract pixel data
    num_frames = ds.NumberOfFrames
    rows = ds.Rows
    cols = ds.Columns
    
    # Reshape pixel data to 3D array
    pixel_data = ds.pixel_array
    
    # Check if already 3D or needs reshaping
    if len(pixel_data.shape) == 2:
        # Need to reshape from flat to 3D
        volume = pixel_data.reshape((num_frames, rows, cols))
    else:
        volume = pixel_data
    
    print(f"Volume shape: {volume.shape}")
    
    # Create interactive viewer
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)
    
    im = ax.imshow(volume[0], cmap='gray')
    ax.set_title(f'Frame 1/{num_frames}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)
    
    def update(val):
        frame = int(slider.val)
        im.set_data(volume[frame])
        ax.set_title(f'Frame {frame+1}/{num_frames}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

# Usage
view_multiframe_dicom('output_3d_multiframe.dcm')