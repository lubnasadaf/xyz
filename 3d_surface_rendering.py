import pydicom
import numpy as np
import pyvista as pv
import os

def render_3d_surface_pyvista(folder_path, threshold_percentile=50):
    """
    Create high-quality 3D surface rendering using PyVista
    More advanced than Plotly with better rendering quality
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
    
    # Get spacing information
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = slices[0].SliceThickness
        spacing = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
    except:
        spacing = [1.0, 1.0, 1.0]
    
    print(f"Voxel spacing: {spacing}")
    
    # Create PyVista grid
    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.spacing = spacing
    grid.point_data["values"] = volume.flatten(order="F")
    
    # Calculate threshold
    threshold_value = np.percentile(volume, threshold_percentile)
    print(f"Threshold value: {threshold_value}")
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add volume with opacity mapping
    plotter.add_volume(grid, cmap="gray", opacity="sigmoid")
    
    # Alternatively, create surface using marching cubes
    # surface = grid.contour([threshold_value])
    # plotter.add_mesh(surface, color='white', opacity=0.8)
    
    # Add axes
    plotter.show_axes()
    
    # Set camera position
    plotter.camera_position = 'xz'
    
    # Show the plot
    plotter.show()

# Usage - Install pyvista first: pip install pyvista
render_3d_surface_pyvista("D:/xyz/dicom_files", threshold_percentile=50)