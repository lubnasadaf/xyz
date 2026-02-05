import pydicom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def view_3d_dicom_slices(dicom_file):
    """Interactive viewer to browse through 3D DICOM slices"""
    dcm = pydicom.dcmread(dicom_file)
    
    # Get the 3D volume
    if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
        # Multi-frame DICOM
        num_frames = dcm.NumberOfFrames
        rows = dcm.Rows
        cols = dcm.Columns
        
        # Reshape pixel array to 3D
        volume = np.frombuffer(dcm.PixelData, dtype=dcm.pixel_array.dtype)
        volume = volume.reshape((num_frames, rows, cols))
    else:
        # Single slice
        volume = dcm.pixel_array[np.newaxis, :, :]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Display first slice
    im = ax.imshow(volume[0], cmap='gray')
    ax.set_title(f'Slice 1/{volume.shape[0]}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0]-1, 
                    valinit=0, valstep=1)
    
    def update(val):
        slice_num = int(slider.val)
        im.set_data(volume[slice_num])
        ax.set_title(f'Slice {slice_num+1}/{volume.shape[0]}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

# Usage
view_3d_dicom_slices('output_3d_multiframe.dcm')