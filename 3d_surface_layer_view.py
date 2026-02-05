import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

def simple_layer_viewer(folder_path, downsample=4):
    """
    Simple fast viewer showing volume slices in 3D space
    Much faster than surface rendering
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
    
    # Downsample for speed
    volume_ds = volume[::downsample, ::downsample, ::downsample]
    print(f"Downsampled shape: {volume_ds.shape}")
    
    # Normalize
    volume_ds = volume_ds.astype(np.float32)
    volume_ds = (volume_ds - volume_ds.min()) / (volume_ds.max() - volume_ds.min())
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    
    # Left: 3D view with slice planes
    ax_3d = fig.add_subplot(121, projection='3d')
    
    # Right: Current slice
    ax_2d = fig.add_subplot(122)
    
    # Initial parameters
    current_z = volume_ds.shape[0] // 2
    num_slices_to_show = 5
    
    def draw_3d_slices(z_center, num_slices):
        ax_3d.clear()
        
        # Calculate slice range
        z_start = max(0, z_center - num_slices // 2)
        z_end = min(volume_ds.shape[0], z_center + num_slices // 2 + 1)
        
        # Create meshgrid for each slice
        Y, X = np.mgrid[0:volume_ds.shape[1], 0:volume_ds.shape[2]]
        
        # Plot multiple slices
        for z in range(z_start, z_end):
            ax_3d.contourf(X, Y, volume_ds[z, :, :], 
                          zdir='z', offset=z, 
                          levels=20, cmap='gray', alpha=0.7)
        
        ax_3d.set_xlabel('X (Columns)')
        ax_3d.set_ylabel('Y (Rows)')
        ax_3d.set_zlabel('Z (Slices)')
        ax_3d.set_title(f'3D Layer View (Z: {z_start} to {z_end})')
        ax_3d.set_xlim([0, volume_ds.shape[2]])
        ax_3d.set_ylim([0, volume_ds.shape[1]])
        ax_3d.set_zlim([0, volume_ds.shape[0]])
        ax_3d.view_init(elev=20, azim=45)
    
    # Initial 3D view
    draw_3d_slices(current_z, num_slices_to_show)
    
    # Initial 2D slice (use original resolution)
    actual_z = current_z * downsample
    im = ax_2d.imshow(volume[actual_z, :, :], cmap='gray')
    ax_2d.set_title(f'Full Resolution Slice at Z={actual_z}')
    ax_2d.axis('off')
    plt.colorbar(im, ax=ax_2d, fraction=0.046)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    # Create sliders
    ax_z_pos = plt.axes([0.15, 0.10, 0.3, 0.03])
    ax_num_slices = plt.axes([0.15, 0.05, 0.3, 0.03])
    
    slider_z = Slider(ax_z_pos, 'Z Position', 0, volume_ds.shape[0]-1, 
                      valinit=current_z, valstep=1)
    slider_num = Slider(ax_num_slices, 'Num Layers', 1, 20, 
                        valinit=num_slices_to_show, valstep=1)
    
    def update_z(val):
        z_pos = int(slider_z.val)
        num_slices = int(slider_num.val)
        
        draw_3d_slices(z_pos, num_slices)
        
        # Update 2D slice
        actual_z = min(z_pos * downsample, volume.shape[0] - 1)
        im.set_data(volume[actual_z, :, :])
        ax_2d.set_title(f'Full Resolution Slice at Z={actual_z}')
        
        fig.canvas.draw_idle()
    
    slider_z.on_changed(update_z)
    slider_num.on_changed(update_z)
    
    print("\nViewer ready! (Much faster)")
    print("Controls:")
    print("- Z Position: Center position of layer stack")
    print("- Num Layers: How many layers to show in 3D")
    
    plt.show()

# Usage
simple_layer_viewer("D:/xyz/dicom_files", downsample=4)