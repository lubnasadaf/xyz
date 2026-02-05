import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os

def test_rotations(folder_path):
    """
    Test different rotation options to find the correct orientation
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
    
    # Get middle slices
    mid_axial = volume.shape[0] // 2
    mid_sagittal = volume.shape[2] // 2
    mid_coronal = volume.shape[1] // 2
    
    # Test different rotations for Sagittal view
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sagittal View - Different Rotations (find the correct one)', fontsize=16)
    
    # Original
    axes[0, 0].imshow(volume[:, :, mid_sagittal], cmap='gray', aspect='auto')
    axes[0, 0].set_title('Original (no rotation)')
    axes[0, 0].axis('off')
    
    # Transpose
    axes[0, 1].imshow(volume[:, :, mid_sagittal].T, cmap='gray', aspect='auto')
    axes[0, 1].set_title('Transposed')
    axes[0, 1].axis('off')
    
    # k=1 (90° CCW)
    axes[0, 2].imshow(np.rot90(volume[:, :, mid_sagittal], k=1), cmap='gray', aspect='auto')
    axes[0, 2].set_title('rot90 k=1 (90° CCW)')
    axes[0, 2].axis('off')
    
    # k=2 (180°)
    axes[1, 0].imshow(np.rot90(volume[:, :, mid_sagittal], k=2), cmap='gray', aspect='auto')
    axes[1, 0].set_title('rot90 k=2 (180°)')
    axes[1, 0].axis('off')
    
    # k=3 (270° CCW = 90° CW)
    axes[1, 1].imshow(np.rot90(volume[:, :, mid_sagittal], k=3), cmap='gray', aspect='auto')
    axes[1, 1].set_title('rot90 k=3 (90° CW)')
    axes[1, 1].axis('off')
    
    # Transpose then rotate
    axes[1, 2].imshow(np.rot90(volume[:, :, mid_sagittal].T, k=1), cmap='gray', aspect='auto')
    axes[1, 2].set_title('Transpose + rot90 k=1')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test different rotations for Coronal view
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('Coronal View - Different Rotations (find the correct one)', fontsize=16)
    
    # Original
    axes2[0, 0].imshow(volume[:, mid_coronal, :], cmap='gray', aspect='auto')
    axes2[0, 0].set_title('Original (no rotation)')
    axes2[0, 0].axis('off')
    
    # Transpose
    axes2[0, 1].imshow(volume[:, mid_coronal, :].T, cmap='gray', aspect='auto')
    axes2[0, 1].set_title('Transposed')
    axes2[0, 1].axis('off')
    
    # k=1 (90° CCW)
    axes2[0, 2].imshow(np.rot90(volume[:, mid_coronal, :], k=1), cmap='gray', aspect='auto')
    axes2[0, 2].set_title('rot90 k=1 (90° CCW)')
    axes2[0, 2].axis('off')
    
    # k=2 (180°)
    axes2[1, 0].imshow(np.rot90(volume[:, mid_coronal, :], k=2), cmap='gray', aspect='auto')
    axes2[1, 0].set_title('rot90 k=2 (180°)')
    axes2[1, 0].axis('off')
    
    # k=3 (270° CCW = 90° CW)
    axes2[1, 1].imshow(np.rot90(volume[:, mid_coronal, :], k=3), cmap='gray', aspect='auto')
    axes2[1, 1].set_title('rot90 k=3 (90° CW)')
    axes2[1, 1].axis('off')
    
    # Transpose then rotate
    axes2[1, 2].imshow(np.rot90(volume[:, mid_coronal, :].T, k=1), cmap='gray', aspect='auto')
    axes2[1, 2].set_title('Transpose + rot90 k=1')
    axes2[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
test_rotations("D:/xyz/dicom_files")