#!/usr/bin/env python3
"""
Interactive 3D Facial Landmark Picker from DICOM Files
Allows manual selection of facial markers (eyes, nose, chin, ears) on a 3D mesh.
"""

import os
import numpy as np
import pydicom
import pyvista as pv
from pathlib import Path
from skimage import measure
import json


class FacialLandmarkPicker:
    def __init__(self, dicom_folder):
        """
        Initialize the facial landmark picker.
        
        Parameters:
        -----------
        dicom_folder : str
            Path to folder containing DICOM files
        """
        self.dicom_folder = Path(dicom_folder)
        self.volume = None
        self.mesh = None
        self.landmarks = {}
        self.plotter = None
        
    def load_dicom_series(self):
        """Load DICOM series and create a 3D volume."""
        print("Loading DICOM files...")
        
        # Get all DICOM files
        dicom_files = sorted([f for f in self.dicom_folder.glob('*.dcm')])
        
        if not dicom_files:
            # Try without extension
            dicom_files = sorted([f for f in self.dicom_folder.iterdir() if f.is_file()])
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {self.dicom_folder}")
        
        print(f"Found {len(dicom_files)} DICOM files")
        
        # Read first file to get dimensions
        first_slice = pydicom.dcmread(str(dicom_files[0]))
        img_shape = first_slice.pixel_array.shape
        
        # Create volume array
        volume = np.zeros((len(dicom_files), img_shape[0], img_shape[1]))
        
        # Load all slices
        slices = []
        for i, filepath in enumerate(dicom_files):
            ds = pydicom.dcmread(str(filepath))
            slices.append(ds)
            volume[i, :, :] = ds.pixel_array
        
        # Sort slices by ImagePositionPatient if available
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            volume = np.array([s.pixel_array for s in slices])
        except:
            print("Warning: Could not sort by ImagePositionPatient, using file order")
        
        # Get voxel spacing
        try:
            pixel_spacing = slices[0].PixelSpacing
            slice_thickness = slices[0].SliceThickness
            self.spacing = [float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1])]
        except:
            print("Warning: Could not get spacing info, using default [1, 1, 1]")
            self.spacing = [1.0, 1.0, 1.0]
        
        self.volume = volume
        print(f"Volume shape: {volume.shape}")
        print(f"Voxel spacing: {self.spacing}")
        
        return volume
    
    def create_mesh(self, threshold=None, smoothing_iterations=50):
        """
        Create 3D mesh from volume using marching cubes.
        
        Parameters:
        -----------
        threshold : float, optional
            Intensity threshold for surface extraction. If None, auto-computed.
        smoothing_iterations : int
            Number of smoothing iterations
        """
        print("Creating 3D mesh...")
        
        if self.volume is None:
            raise ValueError("Volume not loaded. Call load_dicom_series() first.")
        
        # Auto-compute threshold if not provided
        if threshold is None:
            # Use a percentile-based threshold for soft tissue
            threshold = np.percentile(self.volume, 70)
            print(f"Auto-computed threshold: {threshold}")
        
        # Apply marching cubes algorithm
        verts, faces, normals, values = measure.marching_cubes(
            self.volume, 
            level=threshold,
            spacing=self.spacing
        )
        
        # Create PyVista mesh
        faces_pv = np.hstack([[3] + list(face) for face in faces])
        self.mesh = pv.PolyData(verts, faces_pv)
        
        # Apply smoothing
        if smoothing_iterations > 0:
            print(f"Applying {smoothing_iterations} smoothing iterations...")
            self.mesh = self.mesh.smooth(n_iter=smoothing_iterations)
        
        print(f"Mesh created with {self.mesh.n_points} points and {self.mesh.n_cells} cells")
        
        return self.mesh
    
    def pick_landmarks_interactive(self):
        """
        Interactive landmark picking interface.
        Left-click to pick points, right-click to remove last point.
        """
        print("\n" + "="*60)
        print("INTERACTIVE LANDMARK PICKING")
        print("="*60)
        print("\nInstructions:")
        print("  1. Rotate the model to get a good view")
        print("  2. Press a key to select landmark type:")
        print("     'r' - Right Eye")
        print("     'l' - Left Eye")
        print("     'n' - Nose Tip")
        print("     'c' - Chin")
        print("     'e' - Right Ear")
        print("     'a' - Left Ear")
        print("     'o' - Other (custom)")
        print("  3. Click on the mesh to pick the point")
        print("  4. Press 'u' to undo last point")
        print("  5. Press 's' to save landmarks")
        print("  6. Press 'q' to quit")
        print("="*60 + "\n")
        
        self.plotter = pv.Plotter()
        self.plotter.add_mesh(self.mesh, color='tan', opacity=0.9)
        
        # Store picked points
        self.landmark_actors = []
        self.current_landmark_type = None
        
        landmark_types = {
            'r': 'right_eye',
            'l': 'left_eye',
            'n': 'nose_tip',
            'c': 'chin',
            'e': 'right_ear',
            'a': 'left_ear',
            'o': 'other'
        }
        
        landmark_colors = {
            'right_eye': 'red',
            'left_eye': 'blue',
            'nose_tip': 'green',
            'chin': 'yellow',
            'right_ear': 'purple',
            'left_ear': 'orange',
            'other': 'white'
        }
        
        # Create callback functions for each key
        def make_landmark_callback(lm_type):
            def callback():
                self.current_landmark_type = lm_type
                print(f"\nSelected landmark type: {self.current_landmark_type}")
                print("Click on the mesh to pick the point...")
            return callback
        
        def undo_callback():
            # Undo last point
            if self.landmark_actors:
                actor = self.landmark_actors.pop()
                self.plotter.remove_actor(actor)
                # Remove from landmarks dict
                for lm_type, points in list(self.landmarks.items()):
                    if points:
                        removed = points.pop()
                        print(f"\nRemoved {lm_type}: {removed}")
                        if not points:
                            del self.landmarks[lm_type]
                        break
        
        def save_callback():
            self.save_landmarks()
        
        def quit_callback():
            print("\nClosing...")
            self.plotter.close()
        
        def picking_callback(point):
            if self.current_landmark_type is None:
                print("\nPlease select a landmark type first (press r/l/n/c/e/a/o)")
                return
            
            # Add point to landmarks
            if self.current_landmark_type not in self.landmarks:
                self.landmarks[self.current_landmark_type] = []
            
            self.landmarks[self.current_landmark_type].append(point.tolist())
            
            # Display point
            color = landmark_colors[self.current_landmark_type]
            sphere = pv.Sphere(radius=2.0, center=point)
            actor = self.plotter.add_mesh(sphere, color=color, opacity=1.0)
            self.landmark_actors.append(actor)
            
            print(f"Picked {self.current_landmark_type}: {point}")
            print(f"Total landmarks: {sum(len(v) for v in self.landmarks.values())}")
        
        # Enable point picking
        self.plotter.enable_surface_point_picking(
            callback=picking_callback,
            show_point=False,
            color='red',
            point_size=10
        )
        
        # Add key press callbacks
        self.plotter.add_key_event('r', make_landmark_callback('right_eye'))
        self.plotter.add_key_event('l', make_landmark_callback('left_eye'))
        self.plotter.add_key_event('n', make_landmark_callback('nose_tip'))
        self.plotter.add_key_event('c', make_landmark_callback('chin'))
        self.plotter.add_key_event('e', make_landmark_callback('right_ear'))
        self.plotter.add_key_event('a', make_landmark_callback('left_ear'))
        self.plotter.add_key_event('o', make_landmark_callback('other'))
        self.plotter.add_key_event('u', undo_callback)
        self.plotter.add_key_event('s', save_callback)
        self.plotter.add_key_event('q', quit_callback)
        
        # Show the plot
        self.plotter.show()
        
        return self.landmarks
    
    def save_landmarks(self, filename='facial_landmarks.json'):
        """Save landmarks to a JSON file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        landmarks_serializable = {
            k: [list(p) for p in v] for k, v in self.landmarks.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(landmarks_serializable, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Landmarks saved to: {output_path.absolute()}")
        print(f"{'='*60}")
        print("\nSummary:")
        for lm_type, points in self.landmarks.items():
            print(f"  {lm_type}: {len(points)} point(s)")
        print(f"\nTotal: {sum(len(v) for v in self.landmarks.values())} landmarks")
        
    def visualize_landmarks(self, landmark_file='facial_landmarks.json'):
        """Visualize saved landmarks on the mesh."""
        # Load landmarks
        with open(landmark_file, 'r') as f:
            landmarks = json.load(f)
        
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, color='tan', opacity=0.9)
        
        landmark_colors = {
            'right_eye': 'red',
            'left_eye': 'blue',
            'nose_tip': 'green',
            'chin': 'yellow',
            'right_ear': 'purple',
            'left_ear': 'orange',
            'other': 'white'
        }
        
        # Add landmarks
        for lm_type, points in landmarks.items():
            color = landmark_colors.get(lm_type, 'white')
            for point in points:
                sphere = pv.Sphere(radius=2.0, center=point)
                plotter.add_mesh(sphere, color=color, opacity=1.0)
        
        plotter.show()


def main():
    """Main function to run the facial landmark picker."""
    import sys
    
    # Get DICOM folder from command line or use default
    if len(sys.argv) > 1:
        dicom_folder = sys.argv[1]
    else:
        dicom_folder = input("Enter path to DICOM folder: ").strip()
    
    # Create picker instance
    picker = FacialLandmarkPicker(dicom_folder)
    
    # Load DICOM series
    try:
        picker.load_dicom_series()
    except Exception as e:
        print(f"Error loading DICOM files: {e}")
        return
    
    # Create mesh
    threshold = input("\nEnter threshold value (or press Enter for auto): ").strip()
    threshold = float(threshold) if threshold else None
    
    try:
        picker.create_mesh(threshold=threshold, smoothing_iterations=50)
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return
    
    # Pick landmarks interactively
    landmarks = picker.pick_landmarks_interactive()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL LANDMARKS")
    print("="*60)
    for lm_type, points in landmarks.items():
        print(f"\n{lm_type}:")
        for i, point in enumerate(points):
            print(f"  Point {i+1}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")


if __name__ == "__main__":
    main()