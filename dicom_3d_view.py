import pydicom
import numpy as np
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from datetime import datetime

def stack_dicom_slices_corrected(input_folder, output_file):
    """
    Stack multiple 2D DICOM slices into a single multi-frame DICOM file.
    """
    
    # Get all DICOM files
    dicom_files = []
    for file in sorted(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, file)
        try:
            dcm = pydicom.dcmread(filepath)
            dicom_files.append((filepath, dcm))
        except:
            continue
    
    # Sort by Instance Number or Slice Location
    try:
        dicom_files.sort(key=lambda x: int(x[1].InstanceNumber))
    except:
        try:
            dicom_files.sort(key=lambda x: float(x[1].SliceLocation))
        except:
            print("Warning: Could not sort by Instance Number or Slice Location")
    
    print(f"Found {len(dicom_files)} DICOM slices")
    
    if len(dicom_files) == 0:
        print("No DICOM files found!")
        return
    
    # Read reference slice for metadata
    reference_dcm = dicom_files[0][1]
    
    # Get dimensions
    rows = reference_dcm.Rows
    cols = reference_dcm.Columns
    num_slices = len(dicom_files)
    
    print(f"Creating 3D volume: {num_slices} x {rows} x {cols}")
    
    # Create 3D numpy array
    volume_3d = np.zeros((num_slices, rows, cols), dtype=reference_dcm.pixel_array.dtype)
    
    # Stack all slices
    for i, (filepath, dcm) in enumerate(dicom_files):
        volume_3d[i, :, :] = dcm.pixel_array
        if (i + 1) % 50 == 0:
            print(f"Loaded {i+1}/{num_slices} slices")
    
    print(f"All {num_slices} slices loaded")
    
    # Create a new DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    
    # Create the FileDataset instance
    ds = FileDataset(output_file, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Copy important metadata from reference
    ds.PatientName = getattr(reference_dcm, 'PatientName', 'Anonymous')
    ds.PatientID = getattr(reference_dcm, 'PatientID', '12345')
    
    # Set creation date/time
    dt = datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')
    ds.StudyDate = getattr(reference_dcm, 'StudyDate', dt.strftime('%Y%m%d'))
    ds.StudyTime = getattr(reference_dcm, 'StudyTime', dt.strftime('%H%M%S'))
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.SeriesTime = dt.strftime('%H%M%S')
    
    # Set UIDs
    ds.StudyInstanceUID = getattr(reference_dcm, 'StudyInstanceUID', generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    
    # Set modality
    ds.Modality = getattr(reference_dcm, 'Modality', 'CT')
    
    # Set image dimensions
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    
    # Multi-frame specific attributes
    ds.NumberOfFrames = num_slices
    ds.InstanceNumber = 1
    
    # Spacing information
    ds.SliceThickness = 1.0
    ds.SpacingBetweenSlices = 1.0
    if hasattr(reference_dcm, 'PixelSpacing'):
        ds.PixelSpacing = reference_dcm.PixelSpacing
    else:
        ds.PixelSpacing = [1.0, 1.0]
    
    # Set pixel data
    ds.PixelData = volume_3d.tobytes()
    
    # Save the file
    ds.save_as(output_file, write_like_original=False)
    print(f"\nMulti-frame DICOM saved to: {output_file}")
    print(f"Dimensions: {num_slices} frames x {rows} rows x {cols} columns")
    
    return ds

# Usage
input_folder = "D:/xyz/dicom_files"
output_file = "output_3d_multiframe.dcm"

stack_dicom_slices_corrected(input_folder, output_file)