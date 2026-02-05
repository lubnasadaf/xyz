import pydicom
import matplotlib.pyplot as plt
import numpy as np

def view_mpr(dicom_file):
    """View 3D DICOM in three orthogonal planes"""
    dcm = pydicom.dcmread(dicom_file)
    
    # Get the 3D volume
    if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
        num_frames = dcm.NumberOfFrames
        rows = dcm.Rows
        cols = dcm.Columns
        volume = np.frombuffer(dcm.PixelData, dtype=dcm.pixel_array.dtype)
        volume = volume.reshape((num_frames, rows, cols))
    else:
        volume = dcm.pixel_array[np.newaxis, :, :]
    
    # Get middle slices
    mid_axial = volume.shape[0] // 2
    mid_sagittal = volume.shape[2] // 2
    mid_coronal = volume.shape[1] // 2
    
    # Create figure with three views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view (top-down)
    axes[0].imshow(volume[mid_axial, :, :], cmap='gray')
    axes[0].set_title(f'Axial (Slice {mid_axial})')
    axes[0].axis('off')
    
    # Sagittal view (side)
    axes[1].imshow(volume[:, :, mid_sagittal], cmap='gray', aspect='auto')
    axes[1].set_title(f'Sagittal (Slice {mid_sagittal})')
    axes[1].axis('off')
    
    # Coronal view (front)
    axes[2].imshow(volume[:, mid_coronal, :], cmap='gray', aspect='auto')
    axes[2].set_title(f'Coronal (Slice {mid_coronal})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
view_mpr('output_3d_multiframe.dcm')