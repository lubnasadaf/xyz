import pydicom
import numpy as np
import plotly.graph_objects as go
import os

def render_3d_volume_fixed(folder_path, output_html="3d_volume.html", 
                           downsample_factor=2, threshold_percentile=30):
    """
    Create interactive 3D volume rendering and save as HTML file
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
    print(f"Original volume shape: {volume.shape}")
    
    # Downsample for performance
    if downsample_factor > 1:
        volume = volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled volume shape: {volume.shape}")
    
    # Normalize to 0-1 range
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Apply threshold
    threshold = np.percentile(volume, threshold_percentile)
    print(f"Applying threshold at {threshold_percentile}th percentile: {threshold:.3f}")
    
    # Create 3D coordinates
    X, Y, Z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    # Create the 3D volume plot
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.1,
        surface_count=25,
        colorscale='Gray',
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    
    fig.update_layout(
        title='3D Volume Rendering - Rotate with mouse',
        scene=dict(
            xaxis_title='Axial (Head-Feet)',
            yaxis_title='Rows',
            zaxis_title='Columns',
            aspectmode='data',
        ),
        width=1200,
        height=900,
    )
    
    # Save to HTML file
    print(f"Saving 3D visualization to {output_html}...")
    fig.write_html(output_html)
    print(f"Done! Open {output_html} in your web browser to view the 3D volume.")
    
    # Try to open in browser
    import webbrowser
    try:
        webbrowser.open(output_html)
    except:
        print(f"Could not auto-open browser. Please manually open: {output_html}")

# Usage
render_3d_volume_fixed("D:/xyz/dicom_files", 
                       output_html="3d_volume.html",
                       downsample_factor=2, 
                       threshold_percentile=30)