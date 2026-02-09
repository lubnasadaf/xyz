"""
FINAL PRODUCTION: DICOM Face Overlay
Correct transformation: Rotate Y -90°
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

class FinalDICOMFaceOverlay:
    def __init__(self, dicom_folder, landmarks_3d):
        self.dicom_folder = dicom_folder
        self.landmarks_3d_original = landmarks_3d
        
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
        
        # Calculate mesh center for centered transformations
        self.mesh_center = self.mesh_3d_original.points.mean(axis=0)
        print(f"Mesh center: [{self.mesh_center[0]:.1f}, {self.mesh_center[1]:.1f}, {self.mesh_center[2]:.1f}]")
        
        # Store ORIGINAL landmarks for PnP
        self.model_points_for_pnp = np.array([
            self.landmarks_3d_original['right_eye'][0],
            self.landmarks_3d_original['left_eye'][0],
            self.landmarks_3d_original['nose_tip'][0],
            self.landmarks_3d_original['chin'][0],
            self.landmarks_3d_original['right_ear'][1],
            self.landmarks_3d_original['left_ear'][1],
        ], dtype=np.float32)
        
        print("\n✓ DICOM Face Overlay initialized")
        print("  Transformation: Rotate Y -90°")
        print("  Mesh will correctly overlay on your face\n")
        print("Press 'q' to quit\n")
    
    def load_dicom_volume(self):
        """Load DICOM volume"""
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
        
        print(f"  Loaded {len(dicom_files)} slices")
        return volume, spacing
    
    def create_optimized_mesh(self):
        """Create 3D mesh from DICOM volume"""
        print("Creating 3D mesh...")
        
        z_dim, y_dim, x_dim = self.volume.shape
        grid = pv.ImageData()
        grid.dimensions = (x_dim, y_dim, z_dim)
        grid.spacing = (self.spacing[2], self.spacing[1], self.spacing[0])
        
        volume_xyz = np.transpose(self.volume, (2, 1, 0))
        grid.point_data["values"] = volume_xyz.flatten(order="F")
        
        # Create contour
        threshold = np.percentile(self.volume, 70)
        mesh = grid.contour([threshold], scalars="values")
        
        if mesh.n_points == 0:
            threshold = np.percentile(self.volume, 60)
            mesh = grid.contour([threshold], scalars="values")
        
        # Decimate for performance
        mesh = mesh.decimate(0.95)
        print(f"  Created mesh with {mesh.n_points} points")
        
        return mesh
    
    def transform_mesh_points(self, points_3d):
        """
        Apply the correct transformation: Rotate Y -90°
        Transform around the mesh center to keep it aligned
        """
        # Step 1: Center the points
        centered = points_3d - self.mesh_center
        
        # Step 2: Apply rotation around Y axis (-90 degrees)
        angle = np.radians(-90)
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        transformed = (Ry @ centered.T).T
        
        # Step 3: Translate back
        final_points = transformed + self.mesh_center
        
        return final_points
    
    def get_2d_landmarks(self, face_landmarks, w, h):
        """Extract 2D landmarks from MediaPipe detection"""
        image_points = []
        for key in ['right_eye', 'left_eye', 'nose_tip', 'chin', 'right_ear', 'left_ear']:
            idx = self.mp_landmarks[key]
            landmark = face_landmarks[idx]
            x = landmark.x * w
            y = landmark.y * h
            image_points.append([x, y])
        return np.array(image_points, dtype=np.float32)
    
    def process_frame(self, frame):
        """Process a single frame and overlay the DICOM mesh"""
        h, w, _ = frame.shape
        
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.face_landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            image_points = self.get_2d_landmarks(face_landmarks, w, h)
            
            # Camera parameters
            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
            
            try:
                # Solve PnP using original landmarks
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points_for_pnp,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Get original mesh points
                    original_mesh_points = self.mesh_3d_original.points
                    
                    # Apply the correct transformation
                    transformed_mesh_points = self.transform_mesh_points(original_mesh_points)
                    
                    # Project transformed mesh to 2D
                    points_2d, _ = cv2.projectPoints(
                        transformed_mesh_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    points_2d = points_2d.reshape(-1, 2)
                    
                    # Draw mesh overlay (sample every 3rd point for good density)
                    for i in range(0, len(points_2d), 3):
                        x, y = int(points_2d[i][0]), int(points_2d[i][1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                    
                    # Optionally draw face landmarks for reference
                    # Uncomment to show green dots on face landmarks:
                    # for point in image_points:
                    #     cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
                    
            except Exception as e:
                # Silently handle errors
                pass
        
        # Simple status display
        cv2.putText(frame, "DICOM Face Overlay Active", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Run the face overlay application"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam...\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.process_frame(rgb_frame)
            bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow('DICOM Face Overlay - Press q to quit', bgr_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\nClosing application...\n")
        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()


def main():
    """Main entry point"""
    
    # Your 3D landmarks from DICOM
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
    
    # Path to DICOM files
    dicom_folder = r"dicom_files"
    
    try:
        overlay = FinalDICOMFaceOverlay(dicom_folder, landmarks_3d)
        overlay.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()