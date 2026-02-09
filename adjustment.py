"""
ADJUSTABLE DICOM Face Overlay
Fine-tune position and size with keyboard controls
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

class AdjustableDICOMOverlay:
    def __init__(self, dicom_folder, landmarks_3d):
        self.dicom_folder = dicom_folder
        self.landmarks_3d_original = landmarks_3d
        
        # Adjustment parameters - YOU CAN TUNE THESE
        self.offset_x = 0.0  # Left/right adjustment
        self.offset_y = 0.0  # Up/down adjustment
        self.offset_z = 0.0  # Forward/back adjustment
        self.scale = 1.0     # Size adjustment
        
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
        
        # Store ORIGINAL landmarks for PnP
        self.model_points_for_pnp = np.array([
            self.landmarks_3d_original['right_eye'][0],
            self.landmarks_3d_original['left_eye'][0],
            self.landmarks_3d_original['nose_tip'][0],
            self.landmarks_3d_original['chin'][0],
            self.landmarks_3d_original['right_ear'][1],
            self.landmarks_3d_original['left_ear'][1],
        ], dtype=np.float32)
        
        self.print_controls()
    
    def print_controls(self):
        print("\n" + "="*70)
        print("ADJUSTABLE DICOM FACE OVERLAY")
        print("="*70)
        print("KEYBOARD CONTROLS:")
        print("  LEFT/RIGHT ARROWS  - Move overlay left/right")
        print("  UP/DOWN ARROWS     - Move overlay up/down")
        print("  W/S                - Move overlay forward/back")
        print("  +/=                - Make overlay bigger")
        print("  -/_                - Make overlay smaller")
        print("  R                  - Reset all adjustments")
        print("  Q                  - Quit")
        print("="*70)
        print("Current adjustments:")
        print(f"  X offset: {self.offset_x:+.1f}")
        print(f"  Y offset: {self.offset_y:+.1f}")
        print(f"  Z offset: {self.offset_z:+.1f}")
        print(f"  Scale: {self.scale:.2f}")
        print("="*70 + "\n")
    
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
        
        mesh = mesh.decimate(0.95)
        print(f"Mesh: {mesh.n_points} points")
        
        return mesh
    
    def transform_mesh_points(self, points_3d):
        """
        Apply transformation with adjustments:
        1. Rotate Y -90°
        2. Apply offset
        3. Apply scale
        """
        # Center the points
        centered = points_3d - self.mesh_center
        
        # Apply rotation around Y axis (-90 degrees)
        angle = np.radians(-90)
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        rotated = (Ry @ centered.T).T
        
        # Apply scale
        scaled = rotated * self.scale
        
        # Apply offset
        offset_vector = np.array([self.offset_x, self.offset_y, self.offset_z])
        adjusted = scaled + offset_vector
        
        # Translate back
        final_points = adjusted + self.mesh_center
        
        return final_points
    
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
                    # Transform mesh with adjustments
                    original_mesh_points = self.mesh_3d_original.points
                    transformed_mesh_points = self.transform_mesh_points(original_mesh_points)
                    
                    # Project to 2D
                    points_2d, _ = cv2.projectPoints(
                        transformed_mesh_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    points_2d = points_2d.reshape(-1, 2)
                    
                    # Draw mesh
                    for i in range(0, len(points_2d), 3):
                        x, y = int(points_2d[i][0]), int(points_2d[i][1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                    
                    # Draw face landmarks for reference
                    for point in image_points:
                        cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 0), -1)
                    
            except Exception as e:
                pass
        
        # Display controls and current values
        cv2.putText(frame, "DICOM Face Overlay - Adjustable", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"X:{self.offset_x:+.0f} Y:{self.offset_y:+.0f} Z:{self.offset_z:+.0f} Scale:{self.scale:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Arrows: Move | W/S: Depth | +/-: Size | R: Reset", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam...")
        print("Use arrow keys to adjust position!\n")
        
        step_xy = 5.0  # pixels for X/Y adjustment
        step_z = 5.0   # units for Z adjustment
        step_scale = 0.05  # scale increment
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.process_frame(rgb_frame)
            bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('Adjustable DICOM Overlay - Use arrow keys', bgr_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # Left arrow
                self.offset_x -= step_xy
                print(f"X offset: {self.offset_x:+.1f}")
            elif key == 83 or key == 3:  # Right arrow
                self.offset_x += step_xy
                print(f"X offset: {self.offset_x:+.1f}")
            elif key == 82 or key == 0:  # Up arrow
                self.offset_y -= step_xy
                print(f"Y offset: {self.offset_y:+.1f}")
            elif key == 84 or key == 1:  # Down arrow
                self.offset_y += step_xy
                print(f"Y offset: {self.offset_y:+.1f}")
            elif key == ord('w'):  # Forward
                self.offset_z -= step_z
                print(f"Z offset: {self.offset_z:+.1f}")
            elif key == ord('s'):  # Backward
                self.offset_z += step_z
                print(f"Z offset: {self.offset_z:+.1f}")
            elif key == ord('+') or key == ord('='):  # Bigger
                self.scale += step_scale
                print(f"Scale: {self.scale:.2f}")
            elif key == ord('-') or key == ord('_'):  # Smaller
                self.scale = max(0.1, self.scale - step_scale)
                print(f"Scale: {self.scale:.2f}")
            elif key == ord('r'):  # Reset
                self.offset_x = 0.0
                self.offset_y = 0.0
                self.offset_z = 0.0
                self.scale = 1.0
                print("Reset all adjustments")
        
        print("\n" + "="*70)
        print("FINAL ADJUSTMENTS:")
        print("="*70)
        print(f"offset_x = {self.offset_x}")
        print(f"offset_y = {self.offset_y}")
        print(f"offset_z = {self.offset_z}")
        print(f"scale = {self.scale}")
        print("\nCopy these values into your production script!")
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
        overlay = AdjustableDICOMOverlay(dicom_folder, landmarks_3d)
        overlay.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()