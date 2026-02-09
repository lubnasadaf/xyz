"""
FIXED DICOM Face Overlay - Proper transformation for PnP

The issue was centering the model before transforming.
PnP needs the actual spatial relationships preserved.
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

# WORKING PRESETS - Preset 4 should be the solution
PRESETS = {
    '0_original': {
        'flip_x': False, 'flip_y': False, 'flip_z': False,
        'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Original (baseline)'
    },
    '1_flip_x': {
        'flip_x': True, 'flip_y': False, 'flip_z': False,
        'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Just flip X (left-right mirror)'
    },
    '2_flip_z': {
        'flip_x': False, 'flip_y': False, 'flip_z': True,
        'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Just flip Z (front-back mirror)'
    },
    '3_flip_xz': {
        'flip_x': True, 'flip_y': False, 'flip_z': True,
        'rotate_x': 0, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Flip both X and Z'
    },
    '4_winner': {
        'flip_x': True, 'flip_y': False, 'flip_z': True,
        'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': '🏆 WINNER: Flip X+Z, Rotate X 90° (passed all checks)'
    },
    '5_rot90': {
        'flip_x': False, 'flip_y': False, 'flip_z': False,
        'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Just rotate X 90°'
    },
    '6_rot90_flipz': {
        'flip_x': False, 'flip_y': False, 'flip_z': True,
        'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Rotate X 90° + Flip Z'
    },
    '7_rot90_flipx': {
        'flip_x': True, 'flip_y': False, 'flip_z': False,
        'rotate_x': 90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Rotate X 90° + Flip X'
    },
    '8_rot_neg90': {
        'flip_x': False, 'flip_y': False, 'flip_z': False,
        'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Rotate X -90°'
    },
    '9_rot_neg90_flipxz': {
        'flip_x': True, 'flip_y': False, 'flip_z': True,
        'rotate_x': -90, 'rotate_y': 0, 'rotate_z': 0,
        'scale': 1.0,
        'description': 'Rotate X -90° + Flip X+Z'
    },
}

class FixedFaceOverlay:
    def __init__(self, dicom_folder, landmarks_3d):
        self.dicom_folder = dicom_folder
        self.landmarks_3d_original = landmarks_3d
        
        self.current_preset = '4_winner'  # Start with the winner
        
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
        
        # Apply winner preset
        self.apply_preset(self.current_preset)
        
        self.print_controls()
        
    def print_controls(self):
        print("\n" + "="*70)
        print("FIXED FACE OVERLAY - Starting with Preset 4 (Winner)")
        print("="*70)
        for key, preset in PRESETS.items():
            marker = "🏆" if "🏆" in preset['description'] else "  "
            print(f"{marker} {key[0]} - {preset['description']}")
        print("\nq - Quit")
        print("="*70)
        print("🎯 Currently using Preset 4 - the one that passed all checks!")
        print("   Try others if the overlay still doesn't look right.\n")
    
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
    
    def transform_point(self, point, preset_config):
        """
        Transform a point WITHOUT centering first.
        PnP needs the actual coordinate relationships.
        """
        p = point.copy()
        
        # Apply flips FIRST (in original coordinate space)
        if preset_config['flip_x']:
            p[0] = -p[0]
        if preset_config['flip_y']:
            p[1] = -p[1]
        if preset_config['flip_z']:
            p[2] = -p[2]
        
        # Then apply rotations
        if preset_config['rotate_x'] != 0:
            angle = np.radians(preset_config['rotate_x'])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            p = Rx @ p
        
        if preset_config['rotate_y'] != 0:
            angle = np.radians(preset_config['rotate_y'])
            Ry = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            p = Ry @ p
        
        if preset_config['rotate_z'] != 0:
            angle = np.radians(preset_config['rotate_z'])
            Rz = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            p = Rz @ p
        
        # Scale
        p = p * preset_config['scale']
        
        return p
    
    def apply_preset(self, preset_key):
        """Apply a transformation preset"""
        self.current_preset = preset_key
        preset_config = PRESETS[preset_key]
        
        print(f"\n{'='*70}")
        print(f"Applying Preset {preset_key[0]}: {preset_config['description']}")
        print(f"{'='*70}")
        
        # Transform landmarks - these are the KEY points for PnP
        original_points = np.array([
            self.landmarks_3d_original['right_eye'][0],
            self.landmarks_3d_original['left_eye'][0],
            self.landmarks_3d_original['nose_tip'][0],
            self.landmarks_3d_original['chin'][0],
            self.landmarks_3d_original['right_ear'][1],
            self.landmarks_3d_original['left_ear'][1],
        ], dtype=np.float32)
        
        print("\nOriginal landmarks:")
        print(f"  Right eye: {original_points[0]}")
        print(f"  Left eye:  {original_points[1]}")
        print(f"  Nose:      {original_points[2]}")
        
        self.model_points = np.array([
            self.transform_point(p, preset_config) for p in original_points
        ], dtype=np.float32)
        
        print("\nTransformed landmarks (these are used for PnP):")
        print(f"  Right eye: {self.model_points[0]}")
        print(f"  Left eye:  {self.model_points[1]}")
        print(f"  Nose:      {self.model_points[2]}")
        
        # Transform mesh
        self.mesh_3d = self.mesh_3d_original.copy()
        original_mesh_points = self.mesh_3d_original.points.copy()
        transformed_mesh_points = np.array([
            self.transform_point(p, preset_config) for p in original_mesh_points
        ])
        self.mesh_3d.points = transformed_mesh_points
        
        # Validation
        right_eye = self.model_points[0]
        left_eye = self.model_points[1]
        nose = self.model_points[2]
        chin = self.model_points[3]
        
        eye_center = (right_eye + left_eye) / 2
        
        check1 = left_eye[0] > right_eye[0]
        check2 = nose[2] < eye_center[2]
        check3 = chin[1] > nose[1]
        
        print("\nValidation:")
        print(f"  {'✓' if check1 else '✗'} Left eye X ({left_eye[0]:.1f}) > Right eye X ({right_eye[0]:.1f})")
        print(f"  {'✓' if check2 else '✗'} Nose Z ({nose[2]:.1f}) < Eye Z ({eye_center[2]:.1f})")
        print(f"  {'✓' if check3 else '✗'} Chin Y ({chin[1]:.1f}) > Nose Y ({nose[1]:.1f})")
        
        if check1 and check2 and check3:
            print("\n  🎉 This should work correctly!")
        print()
    
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
                    self.model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Project mesh
                    points_3d = self.mesh_3d.points
                    points_2d, _ = cv2.projectPoints(
                        points_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    points_2d = points_2d.reshape(-1, 2)
                    
                    # Draw overlay - reduced density for clarity
                    for i in range(0, len(points_2d), 10):
                        x, y = int(points_2d[i][0]), int(points_2d[i][1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    
                    # Draw 3D model landmarks (red) to verify transformation
                    model_2d, _ = cv2.projectPoints(
                        self.model_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    for pt in model_2d:
                        x, y = int(pt[0][0]), int(pt[0][1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                    
                    # Draw 2D face landmarks (green)
                    for point in image_points:
                        cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
                    
                    # Draw connection lines between corresponding points
                    for i in range(min(len(image_points), len(model_2d))):
                        pt1 = tuple(image_points[i].astype(int))
                        pt2 = tuple(model_2d[i][0].astype(int))
                        if (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(frame, pt1, pt2, (255, 0, 255), 1)
                    
            except Exception as e:
                cv2.putText(frame, f"PnP Error: {str(e)[:40]}", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Status
        preset_desc = PRESETS[self.current_preset]['description']
        cv2.putText(frame, f"Preset {self.current_preset[0]}: {preset_desc[:45]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Red=3D model pts | Green=2D face pts | Cyan=mesh", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🚀 Starting with Preset 4 (the winner)...")
        print("Watch for: Red dots (3D model) should align with Green dots (face)\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.process_frame(rgb_frame)
            bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('DICOM Face Overlay - Press 0-9 for presets, q to quit', bgr_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                preset_num = chr(key)
                matching_presets = [k for k in PRESETS.keys() if k.startswith(preset_num + '_')]
                if matching_presets:
                    self.apply_preset(matching_presets[0])
        
        print("\n" + "="*70)
        print("FINAL CONFIGURATION")
        print("="*70)
        print(f"Preset: {self.current_preset}")
        print(f"Config: {PRESETS[self.current_preset]}")
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
        overlay = FixedFaceOverlay(dicom_folder, landmarks_3d)
        overlay.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()