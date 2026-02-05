import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

try:
    # Try newer MediaPipe API (v0.10+)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except (ImportError, AttributeError):
    # Fall back to older API
    import mediapipe as mp
    USE_NEW_API = False


class DicomAROverlay:
    def __init__(self, dicom_path):
        if USE_NEW_API:
            self.init_new_mediapipe()
        else:
            self.init_legacy_mediapipe()
        
        # Load and process DICOM image
        self.dicom_overlay = self.load_dicom(dicom_path)
        
    def init_new_mediapipe(self):
        """Initialize MediaPipe using new task API (v0.10+)"""
        print("Using new MediaPipe API")
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def init_legacy_mediapipe(self):
        """Initialize MediaPipe using legacy solutions API"""
        print("Using legacy MediaPipe API")
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except AttributeError:
            raise ImportError(
                "MediaPipe installation issue. Please reinstall:\n"
                "pip uninstall mediapipe\n"
                "pip install mediapipe==0.10.9"
            )
        
    def load_dicom(self, path):
        """Load DICOM file and convert to displayable image"""
        try:
            print(f"Loading DICOM from: {path}")
            ds = pydicom.dcmread(path)
            
            # Check if pixel data exists
            if not hasattr(ds, 'pixel_array'):
                raise AttributeError("DICOM file has no pixel data")
            
            # Get pixel array
            img = ds.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = np.max(img) - img  # Invert for MONOCHROME1
            
            # Apply modality LUT if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            # Handle multi-frame DICOM (take first frame)
            if len(img.shape) > 2:
                print(f"Multi-frame DICOM detected, using first frame")
                img = img[12] if img.shape[0] < img.shape[2] else img[:, :, 0]
            
            # Normalize to 0-255
            img = img.astype(float)
            if img.max() > img.min():
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
            
            # Convert grayscale to BGR for overlay
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            
            # Add transparency (create BGRA)
            img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            img_bgra[:, :, 3] = 180  # Set alpha channel (0-255, 180 = semi-transparent)
            
            print(f"✓ DICOM loaded successfully: {img_bgra.shape}")
            return img_bgra
            
        except FileNotFoundError:
            print(f"✗ Error: DICOM file not found at '{path}'")
            print("Please update DICOM_PATH with the correct file path")
            return self.create_placeholder()
        except Exception as e:
            print(f"✗ Error loading DICOM: {e}")
            print(f"Error type: {type(e).__name__}")
            return self.create_placeholder()
    
    def create_placeholder(self):
        """Create a placeholder image for testing"""
        print("Using placeholder image - Update DICOM_PATH to use real medical image")
        placeholder = np.zeros((400, 400, 4), dtype=np.uint8)
        
        # Create a brain-like circular pattern
        center = (200, 200)
        cv2.circle(placeholder, center, 150, (100, 100, 200, 180), -1)
        cv2.circle(placeholder, center, 100, (150, 150, 250, 180), -1)
        cv2.circle(placeholder, (180, 180), 30, (200, 100, 100, 180), -1)
        cv2.circle(placeholder, (220, 180), 30, (200, 100, 100, 180), -1)
        
        cv2.putText(placeholder, "DICOM", (130, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255, 255), 3)
        cv2.putText(placeholder, "PLACEHOLDER", (90, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200, 255), 2)
        
        return placeholder
    
    def get_top_head_region(self, landmarks, frame_shape):
        """Calculate region for top of head (forehead area) for MRI overlay"""
        h, w = frame_shape[:2]
        
        # Get all landmark coordinates
        if USE_NEW_API:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        else:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        
        # Get face dimensions
        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)
        face_top = min(y_coords)
        face_left = min(x_coords)
        face_right = max(x_coords)
        
        # Create region above forehead for top-view MRI
        # Position it above the face, centered horizontally
        overlay_height = int(face_height * 0.8)  # MRI overlay height
        overlay_width = int(face_width * 1.2)   # MRI overlay width
        
        # Center the overlay horizontally with the face
        x_center = (face_left + face_right) / 2
        x_min = int(x_center - overlay_width / 2)
        x_max = int(x_center + overlay_width / 2)
        
        # Position vertically - starting above the forehead
        y_max = int(face_top + face_height * 0.15)  # Slightly overlap with forehead
        y_min = int(y_max - overlay_height)
        
        # Ensure bounds are within frame
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        return x_min, y_min, x_max, y_max
    
    def overlay_image_alpha(self, background, overlay, x, y):
        """Overlay BGRA image on BGR background at position (x, y)"""
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        # Ensure overlay fits within background
        if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
            return background
        
        # Crop overlay if it extends beyond background
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)
        
        # Adjust overlay region
        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)
        
        overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        background_crop = background[y1:y2, x1:x2]
        
        # Extract alpha channel
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Blend images
        blended = (overlay_crop[:, :, :3] * alpha + 
                   background_crop * (1 - alpha)).astype(np.uint8)
        
        background[y1:y2, x1:x2] = blended
        return background
    
    def process_frame_new_api(self, frame):
        """Process frame using new MediaPipe API"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            return detection_result.face_landmarks[0]
        return None
    
    def process_frame_legacy_api(self, frame):
        """Process frame using legacy MediaPipe API"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
    
    def run(self):
        """Main loop for AR overlay"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("\n" + "="*50)
        print("AR DICOM Overlay - Top View MRI on Head")
        print("Controls:")
        print("  'q' - Quit")
        print("  '+' - Increase overlay opacity")
        print("  '-' - Decrease overlay opacity")
        print("  'h' - Toggle overlay height adjustment")
        print("  'w' - Toggle overlay width adjustment")
        print("="*50 + "\n")
        
        alpha_value = 180
        height_scale = 0.8
        width_scale = 1.2
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            if USE_NEW_API:
                landmarks = self.process_frame_new_api(frame)
            else:
                landmarks = self.process_frame_legacy_api(frame)
            
            if landmarks:
                # Get top head region for MRI overlay
                x_min, y_min, x_max, y_max = self.get_top_head_region(
                    landmarks, frame.shape
                )
                
                # Resize DICOM overlay to fit top of head region
                overlay_resized = cv2.resize(
                    self.dicom_overlay,
                    (x_max - x_min, y_max - y_min)
                )
                
                # Update alpha channel
                overlay_resized[:, :, 3] = alpha_value
                
                # Overlay DICOM on top of head
                frame = self.overlay_image_alpha(
                    frame, overlay_resized, x_min, y_min
                )
                
                # Draw bounding box (optional - shows overlay region)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                            (0, 255, 0), 2)
            
            # Display status
            cv2.putText(frame, f"Opacity: {int(alpha_value/255*100)}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('DICOM AR Head Overlay', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                alpha_value = min(255, alpha_value + 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
            elif key == ord('-') or key == ord('_'):
                alpha_value = max(0, alpha_value - 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace with your DICOM file path
    DICOM_PATH = "Brainscan.dcm"
    
    try:
        ar_app = DicomAROverlay(DICOM_PATH)
        ar_app.run()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Reinstall MediaPipe:")
        print("   pip uninstall mediapipe")
        print("   pip install mediapipe==0.10.9")
        print("\n2. Install other dependencies:")
        print("   pip install opencv-python pydicom numpy")