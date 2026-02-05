import pydicom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import os

def mpr_viewer_interactive_fixed(folder_path):
    """
    Interactive Multi-Planar Reconstruction viewer with correct orientations
    """
    
    print("Loading DICOM files...")
    
    # Load all DICOM files
    slices = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            ds = pydicom.dcmread(filepath, force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except:
            continue
    
    if len(slices) == 0:
        print("No valid DICOM files found!")
        return
    
    # Sort slices
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass
    
    print(f"Loaded {len(slices)} slices")
    
    # Create 3D volume
    volume = np.stack([s.pixel_array for s in slices])
    print(f"Volume shape: {volume.shape} (slices, rows, cols)")
    
    # Normalize for better visualization
    volume = volume.astype(np.float32)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Initial slice positions (middle of each dimension)
    axial_idx = volume.shape[0] // 2
    sagittal_idx = volume.shape[2] // 2
    coronal_idx = volume.shape[1] // 2
    
    # Axial view (top-down) - no rotation needed
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(volume[axial_idx, :, :], cmap='gray', aspect='auto')
    ax1.set_title(f'Axial View (Slice {axial_idx}/{volume.shape[0]-1})')
    ax1.set_xlabel('Left-Right')
    ax1.set_ylabel('Anterior-Posterior')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Sagittal view (side view) - FIXED: rotate 90 degrees LEFT (k=3 or k=-1)
    ax2 = plt.subplot(1, 3, 2)
    sagittal_slice = np.rot90(volume[:, :, sagittal_idx], k=2)  # Changed from k=1 to k=3
    im2 = ax2.imshow(sagittal_slice, cmap='gray', aspect='auto')
    ax2.set_title(f'Sagittal View (Slice {sagittal_idx}/{volume.shape[2]-1})')
    ax2.set_xlabel('Anterior-Posterior')
    ax2.set_ylabel('Superior-Inferior')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Coronal view (front view) - FIXED: rotate 90 degrees LEFT (k=3 or k=-1)
    ax3 = plt.subplot(1, 3, 3)
    coronal_slice = np.rot90(volume[:, coronal_idx, :], k=2)  # Changed from k=1 to k=3
    im3 = ax3.imshow(coronal_slice, cmap='gray', aspect='auto')
    ax3.set_title(f'Coronal View (Slice {coronal_idx}/{volume.shape[1]-1})')
    ax3.set_xlabel('Left-Right')
    ax3.set_ylabel('Superior-Inferior')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Create sliders
    ax_axial = plt.axes([0.15, 0.15, 0.2, 0.03])
    ax_sagittal = plt.axes([0.15, 0.10, 0.2, 0.03])
    ax_coronal = plt.axes([0.15, 0.05, 0.2, 0.03])
    
    slider_axial = Slider(ax_axial, 'Axial', 0, volume.shape[0]-1, 
                          valinit=axial_idx, valstep=1)
    slider_sagittal = Slider(ax_sagittal, 'Sagittal', 0, volume.shape[2]-1, 
                            valinit=sagittal_idx, valstep=1)
    slider_coronal = Slider(ax_coronal, 'Coronal', 0, volume.shape[1]-1, 
                           valinit=coronal_idx, valstep=1)
    
    def update_axial(val):
        idx = int(slider_axial.val)
        im1.set_data(volume[idx, :, :])
        ax1.set_title(f'Axial View (Slice {idx}/{volume.shape[0]-1})')
        fig.canvas.draw_idle()
    
    def update_sagittal(val):
        idx = int(slider_sagittal.val)
        sagittal_slice = np.rot90(volume[:, :, idx], k=2)  # Changed from k=1 to k=3
        im2.set_data(sagittal_slice)
        ax2.set_title(f'Sagittal View (Slice {idx}/{volume.shape[2]-1})')
        fig.canvas.draw_idle()
    
    def update_coronal(val):
        idx = int(slider_coronal.val)
        coronal_slice = np.rot90(volume[:, idx, :], k=2)  # Changed from k=1 to k=3
        im3.set_data(coronal_slice)
        ax3.set_title(f'Coronal View (Slice {idx}/{volume.shape[1]-1})')
        fig.canvas.draw_idle()
    
    slider_axial.on_changed(update_axial)
    slider_sagittal.on_changed(update_sagittal)
    slider_coronal.on_changed(update_coronal)
    
    print("\nUse the sliders to navigate through different slices")
    plt.show()

# Usage
mpr_viewer_interactive_fixed("D:/xyz/dicom_files")