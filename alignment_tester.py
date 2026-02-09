"""
Expanded DICOM Face Overlay with Complex Transformations

Includes combinations of flips AND rotations
Use keys 0-9, then a-z for extended presets
"""

import cv2
import numpy as np
import pyvista as pv
import pydicom
import glob
import urllib.request
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# EXPANDED PRESETS - combinations of flips and rotations
PRESETS = {
    # Basic flips (0-7)
    '0': {'flip_x': False, 'flip_y': False, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Original'},
    '1': {'flip_x': False, 'flip_y': True, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y'},
    '2': {'flip_x': False, 'flip_y': False, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Z'},
    '3': {'flip_x': False, 'flip_y': True, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y+Z'},
    '4': {'flip_x': True, 'flip_y': False, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X'},
    '5': {'flip_x': True, 'flip_y': True, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Y'},
    '6': {'flip_x': True, 'flip_y': False, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Z'},
    '7': {'flip_x': True, 'flip_y': True, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Y+Z'},
    
    # Rotations only (8-9)
    '8': {'flip_x': False, 'flip_y': False, 'flip_z': False, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Rot X 90°'},
    '9': {'flip_x': False, 'flip_y': False, 'flip_z': False, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Rot X -90°'},
    
    # Flip Y + Rotations (a-d)
    'a': {'flip_x': False, 'flip_y': True, 'flip_z': False, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y + Rot X 90°'},
    'b': {'flip_x': False, 'flip_y': True, 'flip_z': False, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y + Rot X -90°'},
    'c': {'flip_x': False, 'flip_y': True, 'flip_z': False, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y + Rot X 180°'},
    'd': {'flip_x': False, 'flip_y': True, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip Y + Rot Y 180°'},
    
    # Flip Z + Rotations (e-h)
    'e': {'flip_x': False, 'flip_y': False, 'flip_z': True, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Z + Rot X 90°'},
    'f': {'flip_x': False, 'flip_y': False, 'flip_z': True, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Z + Rot X -90°'},
    'g': {'flip_x': False, 'flip_y': False, 'flip_z': True, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Z + Rot X 180°'},
    'h': {'flip_x': False, 'flip_y': False, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip Z + Rot Y 180°'},
    
    # Flip Y+Z + Rotations (i-l)
    'i': {'flip_x': False, 'flip_y': True, 'flip_z': True, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y+Z + Rot X 90°'},
    'j': {'flip_x': False, 'flip_y': True, 'flip_z': True, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y+Z + Rot X -90°'},
    'k': {'flip_x': False, 'flip_y': True, 'flip_z': True, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip Y+Z + Rot X 180°'},
    'l': {'flip_x': False, 'flip_y': True, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip Y+Z + Rot Y 180°'},
    
    # Flip X + Rotations (m-p)
    'm': {'flip_x': True, 'flip_y': False, 'flip_z': False, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X + Rot X 90°'},
    'n': {'flip_x': True, 'flip_y': False, 'flip_z': False, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X + Rot X -90°'},
    'o': {'flip_x': True, 'flip_y': False, 'flip_z': False, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X + Rot X 180°'},
    'p': {'flip_x': True, 'flip_y': False, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip X + Rot Y 180°'},
    
    # Flip X+Z + Rotations (q-t)
    'q': {'flip_x': True, 'flip_y': False, 'flip_z': True, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Z + Rot X 90°'},
    'r': {'flip_x': True, 'flip_y': False, 'flip_z': True, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Z + Rot X -90°'},
    's': {'flip_x': True, 'flip_y': False, 'flip_z': True, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Z + Rot X 180°'},
    't': {'flip_x': True, 'flip_y': False, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip X+Z + Rot Y 180°'},
    
    # Flip X+Y+Z + Rotations (u-x)
    'u': {'flip_x': True, 'flip_y': True, 'flip_z': True, 'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Y+Z + Rot X 90°'},
    'v': {'flip_x': True, 'flip_y': True, 'flip_z': True, 'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Y+Z + Rot X -90°'},
    'w': {'flip_x': True, 'flip_y': True, 'flip_z': True, 'rotate_x': 180, 'rotate_y': 0, 'rotate_z': 0, 'name': 'Flip X+Y+Z + Rot X 180°'},
    'x': {'flip_x': True, 'flip_y': True, 'flip_z': True, 'rotate_x': 0, 'rotate_y': 180, 'rotate_z': 0, 'name': 'Flip X+Y+Z + Rot Y 180°'},
    
    # Y-axis rotations (y-z)
    'y': {'flip_x': False, 'flip_y': False, 'flip_z': False, 'rotate_x': 0, 'rotate_y': 90, 'rotate_z': 0, 'name': 'Rot Y 90°'},
    'z': {'flip_x': False, 'flip_y': False, 'flip_z': False, 'rotate_x': 0, 'rotate_y': -90, 'rotate_z': 0, 'name': 'Rot Y -90°'},
}

class ExpandedDICOMOverlay:
    def __init__(self, dicom_folder, landmarks_3d):
        self.dicom_folder = dicom_folder
        self.landmarks_3d_original = landmarks_3d
        self.current_preset = '0'
        self.tested_presets = set()
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(model_url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.mp_landmarks = {
            'right_eye': 33, 'left_eye': 263, 'nose_tip': 1,
            'chin': 152, 'right_ear': 127, 'left_ear': 356
        }
        
        # Load DICOM
        print("Loading DICOM...")
        self.volume, self.spacing = self.load_dicom_volume()
        self.mesh_3d_original = self.create_optimized_mesh()
        
        # Calculate mesh center
        self.mesh_center = self.mesh_3d_original.points.mean(axis=0)
        
        # ORIGINAL landmarks for PnP
        self.model_points_for_pnp = np.array([
            self.landmarks_3d_original['right_eye'][0],
            self.landmarks_3d_original['left_eye'][0],
            self.landmarks_3d_original['nose_tip'][0],
            self.landmarks_3d_original['chin'][0],
            self.landmarks_3d_original['right_ear'][1],
            self.landmarks_3d_original['left_ear'][1],
        ], dtype=np.float32)
        
        print("\n" + "="*70)
        print("EXPANDED TRANSFORMATIONS - 26 presets!")
        print("="*70)
        print("Press 0-9 for basic, a-z for advanced combinations")
        print("\nRECOMMENDED ORDER TO TRY:")
        print("  1. Try 'i' (Flip Y+Z + Rot X 90°) - Most common for CT scans")
        print("  2. Try 'e' (Flip Z + Rot X 90°)")
        print("  3. Try 'a' (Flip Y + Rot X 90°)")
        print("  4. Try 'q' (Flip X+Z + Rot X 90°)")
        print("  5. Then try others if needed")
        print("="*70 + "\n")
        
        # Apply initial preset
        self.apply_preset('0')
    
    def load_dicom_volume(self):
        dicom_files = sorted(glob.glob(f"{self.dicom_folder}/*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {self.dicom_folder}")
        
        first_slice = pydicom.dcmread(dicom_files[0])
        img_shape = first_slice.pixel_array.shape
        
        volume = np.zeros((len(dicom_files), img_shape[0], img_shape[1]), dtype=np.float32)
        for i, filepath in enumerate(dicom_files):
            volume[i, :, :] = pydicom.dcmread(filepath).pixel_array.astype(np.float32)
        
        spacing = [
            float(first_slice.SliceThickness) if hasattr(first_slice, 'SliceThickness') else 1.0,
            float(first_slice.PixelSpacing[0]),
            float(first_slice.PixelSpacing[1])
        ]
        
        print(f"Loaded {len(dicom_files)} slices")
        return volume, spacing
    
    def create_optimized_mesh(self):
        print("Creating mesh...")
        
        z_dim, y_dim, x_dim = self.volume.shape
        grid = pv.ImageData()
        grid.dimensions = (x_dim, y_dim, z_dim)
        grid.spacing = (self.spacing[2], self.spacing[1], self.spacing[0])
        
        volume_xyz = np.transpose(self.volume, (2, 1, 0))
        grid.point_data["values"] = volume_xyz.flatten(order="F")
        
        threshold = np.percentile(self.volume, 70)
        mesh = grid.contour([threshold], scalars="values")
        
        if mesh.n_points == 0:
            threshold = np.percentile(self.volume, 60)
            mesh = grid.contour([threshold], scalars="values")
        
        mesh = mesh.decimate(0.97)
        print(f"Mesh: {mesh.n_points} points")
        
        return mesh
    
    def transform_points_centered(self, points_3d, config, center):
        """Transform points around center"""
        # Center
        centered = points_3d - center
        
        # Flips
        transformed = centered.copy()
        if config['flip_x']:
            transformed[:, 0] = -transformed[:, 0]
        if config['flip_y']:
            transformed[:, 1] = -transformed[:, 1]
        if config['flip_z']:
            transformed[:, 2] = -transformed[:, 2]
        
        # Rotation X
        if config['rotate_x'] != 0:
            angle = np.radians(config['rotate_x'])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            transformed = (Rx @ transformed.T).T
        
        # Rotation Y
        if config['rotate_y'] != 0:
            angle = np.radians(config['rotate_y'])
            Ry = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            transformed = (Ry @ transformed.T).T
        
        # Rotation Z
        if config['rotate_z'] != 0:
            angle = np.radians(config['rotate_z'])
            Rz = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            transformed = (Rz @ transformed.T).T
        
        # Translate back
        return transformed + center
    
    def apply_preset(self, preset_key):
        """Apply preset"""
        self.current_preset = preset_key
        self.current_config = PRESETS[preset_key]
        self.tested_presets.add(preset_key)
        
        print(f"\n[{preset_key.upper()}] {self.current_config['name']}")
        print(f"    Tested {len(self.tested_presets)}/{len(PRESETS)} presets")
    
    def get_2d_landmarks(self, face_landmarks, w, h):
        image_points = []
        for key in ['right_eye', 'left_eye', 'nose_tip', 'chin', 'right_ear', 'left_ear']:
            idx = self.mp_landmarks[key]
            landmark = face_landmarks[idx]
            x = landmark.x * w
            y = landmark.y * h
            image_points.append([x, y])
        return np.array(image_points, dtype=np.float32)
    
    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.face_landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            image_points = self.get_2d_landmarks(face_landmarks, w, h)
            
            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
            
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points_for_pnp,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Transform mesh
                    original_mesh_points = self.mesh_3d_original.points
                    transformed_mesh_points = self.transform_points_centered(
                        original_mesh_points, 
                        self.current_config,
                        self.mesh_center
                    )
                    
                    # Project
                    points_2d, _ = cv2.projectPoints(
                        transformed_mesh_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    points_2d = points_2d.reshape(-1, 2)
                    
                    # Draw
                    for i in range(0, len(points_2d), 6):
                        x, y = int(points_2d[i][0]), int(points_2d[i][1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                    
                    # Landmarks
                    for point in image_points:
                        cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 0), -1)
                    
            except Exception as e:
                pass
        
        # Display
        cv2.putText(frame, f"[{self.current_preset.upper()}] {self.current_config['name']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tested: {len(self.tested_presets)}/26 | Try: i, e, a, q first", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam...")
        print("Try 'i' first (most common for medical scans)!\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.process_frame(rgb_frame)
            bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('Expanded Presets - Press 0-9, a-z, q to quit', bgr_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            else:
                key_char = chr(key)
                if key_char in PRESETS:
                    self.apply_preset(key_char)
        
        print("\n" + "="*70)
        print(f"Final: [{self.current_preset.upper()}] {PRESETS[self.current_preset]['name']}")
        print(f"Tested {len(self.tested_presets)} presets")
        print("\nConfiguration:")
        print(PRESETS[self.current_preset])
        print("="*70)
        
        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()


def main():
    landmarks_3d = {
        'right_eye': [np.array([376.91, 97.84, 146.96])],
        'left_eye': [np.array([376.84, 97.25, 213.39])],
        'nose_tip': [np.array([330.25, 52.24, 170.25])],
        'chin': [np.array([226.89, 71.91, 178.41])],
        'right_ear': [
            np.array([374.47, 201.74, 94.72]),
            np.array([331.24, 217.29, 96.60]),
            np.array([278.03, 200.02, 94.20])
        ],
        'left_ear': [
            np.array([367.78, 199.41, 268.72]),
            np.array([318.08, 216.06, 266.35]),
            np.array([278.09, 196.05, 265.34])
        ],
    }
    
    dicom_folder = r"dicom_files"
    
    try:
        overlay = ExpandedDICOMOverlay(dicom_folder, landmarks_3d)
        overlay.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()