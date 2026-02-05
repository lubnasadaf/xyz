import pydicom
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def combined_mpr_3d_view_fixed(folder_path):
    """
    Combined view showing MPR and 3D representation with proper visualization
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
    print(f"Volume shape: {volume.shape}")
    
    # Normalize volume
    volume = volume.astype(np.float32)
    volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Get middle slices
    mid_axial = volume.shape[0] // 2
    mid_sagittal = volume.shape[2] // 2
    mid_coronal = volume.shape[1] // 2
    
    # Axial view
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(volume[mid_axial, :, :], cmap='gray')
    ax1.set_title(f'Axial View (Slice {mid_axial})')
    ax1.axis('off')
    
    # Sagittal view - FIXED rotation
    ax2 = plt.subplot(2, 2, 2)
    sagittal_slice = np.rot90(volume[:, :, mid_sagittal], k=1)
    ax2.imshow(sagittal_slice, cmap='gray')
    ax2.set_title(f'Sagittal View (Slice {mid_sagittal})')
    ax2.axis('off')
    
    # Coronal view - FIXED rotation
    ax3 = plt.subplot(2, 2, 3)
    coronal_slice = np.rot90(volume[:, mid_coronal, :], k=1)
    ax3.imshow(coronal_slice, cmap='gray')
    ax3.set_title(f'Coronal View (Slice {mid_coronal})')
    ax3.axis('off')
    
    # 3D visualization - FIXED
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    
    # Downsample significantly for 3D display
    downsample = 10
    volume_small = volume_normalized[::downsample, ::downsample, ::downsample]
    
    print(f"3D visualization using downsampled volume: {volume_small.shape}")
    
    # Use a lower threshold to see more structure
    threshold = np.percentile(volume_small, 40)  # CHANGED from 70 to 40
    print(f"3D threshold: {threshold:.3f}")
    
    voxels = volume_small > threshold
    
    # Count voxels to display
    num_voxels = np.sum(voxels)
    print(f"Number of voxels to display: {num_voxels}")
    
    if num_voxels > 0:
        # Create color array based on intensity
        colors = np.zeros(volume_small.shape + (3,))
        colors[voxels] = [0.8, 0.8, 0.8]  # Light gray
        
        # Plot voxels with better visibility
        ax4.voxels(voxels, facecolors=colors, edgecolors=None, alpha=0.5)
        ax4.set_title(f'3D Volume ({num_voxels} voxels)')
    else:
        ax4.text(0.5, 0.5, 0.5, 'Adjust threshold\nto see structure', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('3D Volume (no voxels above threshold)')
    
    ax4.set_xlabel('Axial')
    ax4.set_ylabel('Rows')
    ax4.set_zlabel('Cols')
    
    # Set background color to white
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.show()

# Usage
combined_mpr_3d_view_fixed("D:/xyz/dicom_files")