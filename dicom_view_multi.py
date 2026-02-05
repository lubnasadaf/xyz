import pydicom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import os

def view_dicom_series_robust(folder_path):
    """
    Robust viewer for a series of DICOM files from a folder.
    Handles various common issues.
    """
    
    print(f"Reading DICOM files from: {folder_path}")
    
    # Load all DICOM files
    slices = []
    file_list = []
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(filepath):
            continue
            
        try:
            ds = pydicom.dcmread(filepath, force=True)
            
            # Check if it has pixel data
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
                file_list.append(filename)
            else:
                print(f"Skipping {filename} - no pixel data")
                
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue
    
    if len(slices) == 0:
        print("No valid DICOM files with pixel data found!")
        return
    
    print(f"Loaded {len(slices)} valid DICOM slices")
    
    # Try to sort by Instance Number
    try:
        slices_with_files = list(zip(slices, file_list))
        slices_with_files.sort(key=lambda x: int(x[0].InstanceNumber))
        slices = [s[0] for s in slices_with_files]
        file_list = [s[1] for s in slices_with_files]
        print("Sorted by Instance Number")
    except:
        # Try to sort by Slice Location
        try:
            slices_with_files = list(zip(slices, file_list))
            slices_with_files.sort(key=lambda x: float(x[0].SliceLocation))
            slices = [s[0] for s in slices_with_files]
            file_list = [s[1] for s in slices_with_files]
            print("Sorted by Slice Location")
        except:
            # Try to sort by Image Position Patient (Z coordinate)
            try:
                slices_with_files = list(zip(slices, file_list))
                slices_with_files.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
                slices = [s[0] for s in slices_with_files]
                file_list = [s[1] for s in slices_with_files]
                print("Sorted by Image Position Patient (Z)")
            except:
                print("Warning: Could not sort slices - using file order")
    
    # Print some info about first slice
    print(f"\nFirst slice info:")
    print(f"  Rows: {slices[0].Rows}")
    print(f"  Columns: {slices[0].Columns}")
    print(f"  Modality: {getattr(slices[0], 'Modality', 'Unknown')}")
    
    # Create volume - handle different data types
    try:
        # Try to stack all pixel arrays
        volume = np.stack([s.pixel_array for s in slices])
        print(f"\nVolume shape: {volume.shape}")
        print(f"Volume dtype: {volume.dtype}")
        print(f"Value range: [{volume.min()}, {volume.max()}]")
    except Exception as e:
        print(f"Error creating volume: {e}")
        return
    
    # Create interactive viewer
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
    # Display first slice
    im = ax.imshow(volume[0], cmap='gray', interpolation='nearest')
    ax.set_title(f'Slice 1/{len(slices)}\nFile: {file_list[0]}')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(slices)-1, valinit=0, valstep=1)
    
    def update(val):
        slice_num = int(slider.val)
        im.set_data(volume[slice_num])
        im.set_clim(vmin=volume[slice_num].min(), vmax=volume[slice_num].max())
        ax.set_title(f'Slice {slice_num+1}/{len(slices)}\nFile: {file_list[slice_num]}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add keyboard controls
    def on_key(event):
        if event.key == 'right' or event.key == 'up':
            new_val = min(slider.val + 1, len(slices) - 1)
            slider.set_val(new_val)
        elif event.key == 'left' or event.key == 'down':
            new_val = max(slider.val - 1, 0)
            slider.set_val(new_val)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nViewer controls:")
    print("  - Use slider or arrow keys to navigate")
    print("  - Close window to exit")
    
    plt.show()

# Usage
folder_path = "D:/xyz/dicom_files"
view_dicom_series_robust(folder_path)