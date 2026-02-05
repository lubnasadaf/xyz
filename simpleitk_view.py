import SimpleITK as sitk
import matplotlib.pyplot as plt

def view_with_sitk(dicom_file):
    """View DICOM using SimpleITK"""
    # Read DICOM
    image = sitk.ReadImage(dicom_file)
    
    # Convert to numpy array
    volume = sitk.GetArrayFromImage(image)
    
    # Display middle slice
    mid_slice = volume.shape[0] // 2
    
    plt.figure(figsize=(10, 10))
    plt.imshow(volume[mid_slice], cmap='gray')
    plt.title(f'Middle Slice ({mid_slice}/{volume.shape[0]})')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    print(f"Image size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")

# Usage (requires: pip install SimpleITK)
view_with_sitk('output_3d_multiframe.dcm')