import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import os
import glob

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


class DicomFaceMapping:
    def __init__(self, dicom_path):
        if USE_NEW_API:
            self.init_new_mediapipe()
        else:
            self.init_legacy_mediapipe()
        
        # Load and process DICOM image stack
        self.dicom_stack = []
        self.current_slice = 0
        self.rotation_angle = 180  # Start with 180-degree rotation
        self.scale_factor = 2.5  # Scale factor for overlay size (increase this for bigger overlay)
        self.vertical_offset = 0  # Vertical position adjustment
        self.horizontal_offset = 0  # Horizontal position adjustment
        self.load_dicom(dicom_path)
        
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
        """Load DICOM file(s) and convert to displayable image stack"""
        try:
            # Check if path is a directory or a file
            if os.path.isdir(path):
                print(f"Loading DICOM files from folder: {path}")
                self.load_dicom_folder(path)
            else:
                print(f"Loading single DICOM file: {path}")
                self.load_single_dicom(path)
            
            if len(self.dicom_stack) == 0:
                raise ValueError("No valid slices found in DICOM")
            
            # Show a preview window of the first slice
            preview = cv2.resize(self.dicom_stack[0][:,:,:3], (300, 300))
            cv2.imshow('DICOM Preview - First Slice', preview)
            print(f"\n✓ Total slices loaded: {len(self.dicom_stack)}")
            print("  Press any key on the preview window to continue...")
            cv2.waitKey(2000)  # Wait 2 seconds
            cv2.destroyWindow('DICOM Preview - First Slice')
                
        except FileNotFoundError:
            print(f"✗ Error: DICOM path not found at '{path}'")
            print("Please update DICOM_PATH with the correct file or folder path")
            self.dicom_stack.append(self.create_placeholder())
        except Exception as e:
            print(f"✗ Error loading DICOM: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.dicom_stack.append(self.create_placeholder())
    
    def load_dicom_folder(self, folder_path):
        """Load all DICOM files from a folder"""
        # Find all .dcm files in the folder
        dicom_files = glob.glob(os.path.join(folder_path, "*.dcm"))
        
        if not dicom_files:
            # Try without extension
            dicom_files = glob.glob(os.path.join(folder_path, "*"))
            dicom_files = [f for f in dicom_files if os.path.isfile(f)]
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        # Sort files by name (usually corresponds to slice order)
        dicom_files.sort()
        
        print(f"Found {len(dicom_files)} files in folder")
        
        # Load each DICOM file
        for i, dicom_file in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(dicom_file)
                
                if not hasattr(ds, 'pixel_array'):
                    print(f"  Skipping {os.path.basename(dicom_file)} - no pixel data")
                    continue
                
                img = ds.pixel_array
                
                # Handle photometric interpretation
                if hasattr(ds, 'PhotometricInterpretation'):
                    if ds.PhotometricInterpretation == "MONOCHROME1":
                        img = np.max(img) - img
                
                # Apply modality LUT if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                
                # Process the slice
                processed_slice = self.process_slice(img)
                self.dicom_stack.append(processed_slice)
                
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{len(dicom_files)} files...")
                    
            except Exception as e:
                print(f"  Error loading {os.path.basename(dicom_file)}: {e}")
                continue
        
        print(f"✓ Successfully loaded {len(self.dicom_stack)} slices from folder")
    
    def load_single_dicom(self, path):
        """Load a single DICOM file (may contain multiple slices)"""
        print(f"Loading DICOM from: {path}")
        ds = pydicom.dcmread(path)
        
        # Print DICOM info for debugging
        print(f"DICOM Info:")
        print(f"  - Has pixel_array: {hasattr(ds, 'pixel_array')}")
        
        # Check if pixel data exists
        if not hasattr(ds, 'pixel_array'):
            raise AttributeError("DICOM file has no pixel data")
        
        # Get pixel array
        img = ds.pixel_array
        print(f"  - Original shape: {img.shape}")
        print(f"  - Data type: {img.dtype}")
        print(f"  - Min/Max values: {img.min()}/{img.max()}")
        
        # Handle different photometric interpretations
        if hasattr(ds, 'PhotometricInterpretation'):
            print(f"  - Photometric: {ds.PhotometricInterpretation}")
            if ds.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img  # Invert for MONOCHROME1
        
        # Apply modality LUT if available
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            print(f"  - Applying rescale: slope={ds.RescaleSlope}, intercept={ds.RescaleIntercept}")
            img = img * ds.RescaleSlope + ds.RescaleIntercept
        
        # Handle multi-frame/stacked DICOM
        if len(img.shape) > 2:
            num_slices = img.shape[0]
            print(f"✓ Multi-frame DICOM detected: {num_slices} slices")
            
            # Process each slice
            for i in range(num_slices):
                slice_img = img[i]
                processed_slice = self.process_slice(slice_img)
                self.dicom_stack.append(processed_slice)
                
            print(f"✓ Loaded {len(self.dicom_stack)} slices")
        else:
            # Single frame DICOM
            print(f"✓ Single-frame DICOM detected")
            processed_slice = self.process_slice(img)
            self.dicom_stack.append(processed_slice)
            print(f"  - Processed shape: {processed_slice.shape}")
    
    def process_slice(self, slice_img):
        """Process a single DICOM slice to BGRA format"""
        # Normalize to 0-255
        slice_img = slice_img.astype(float)
        if slice_img.max() > slice_img.min():
            slice_img = ((slice_img - slice_img.min()) / 
                        (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        else:
            slice_img = np.ones_like(slice_img, dtype=np.uint8) * 128
        
        # Convert grayscale to BGR
        if len(slice_img.shape) == 2:
            img_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = slice_img
        
        # Add transparency (create BGRA)
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = 255  # Full opacity initially
        
        return img_bgra
    
    def create_placeholder(self):
        """Create a placeholder face image for testing"""
        print("Using placeholder image - Update DICOM_PATH to use real medical image")
        placeholder = np.zeros((400, 400, 4), dtype=np.uint8)
        
        # Create a skull-like pattern
        # Skull outline
        cv2.ellipse(placeholder, (200, 180), (120, 140), 0, 0, 360, (200, 200, 200, 255), -1)
        # Eye sockets
        cv2.ellipse(placeholder, (160, 160), (25, 35), 0, 0, 360, (80, 80, 80, 255), -1)
        cv2.ellipse(placeholder, (240, 160), (25, 35), 0, 0, 360, (80, 80, 80, 255), -1)
        # Nose cavity
        cv2.ellipse(placeholder, (200, 200), (15, 20), 0, 0, 360, (80, 80, 80, 255), -1)
        # Jaw
        cv2.ellipse(placeholder, (200, 280), (80, 60), 0, 0, 360, (180, 180, 180, 255), -1)
        
        cv2.putText(placeholder, "FACE", (150, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255, 255), 3)
        cv2.putText(placeholder, "X-RAY", (145, 385), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200, 255), 2)
        
        return placeholder
    
    def rotate_image(self, image, angle):
        """Rotate image by specified angle (positive = anticlockwise)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate the image
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        
        return rotated
    
    def rotate_90_cw(self):
        """Rotate 90 degrees clockwise"""
        self.rotation_angle = (self.rotation_angle - 90) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def rotate_90_ccw(self):
        """Rotate 90 degrees counter-clockwise"""
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def rotate_180(self):
        """Toggle 180 degrees rotation"""
        self.rotation_angle = (self.rotation_angle + 180) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def adjust_scale(self, increment):
        """Adjust the scale factor of the overlay"""
        self.scale_factor = max(0.5, min(5.0, self.scale_factor + increment))
        print(f"Scale: {self.scale_factor:.2f}x")
    
    def adjust_vertical_offset(self, increment):
        """Adjust vertical position of overlay"""
        self.vertical_offset += increment
        print(f"Vertical offset: {self.vertical_offset}px")
    
    def adjust_horizontal_offset(self, increment):
        """Adjust horizontal position of overlay"""
        self.horizontal_offset += increment
        print(f"Horizontal offset: {self.horizontal_offset}px")
    
    def reset_adjustments(self):
        """Reset all adjustments to defaults"""
        self.scale_factor = 2.5  # Match the initial scale factor
        self.vertical_offset = 0
        self.horizontal_offset = 0
        print("Reset: Scale=2.5x, Offsets=0")
    
    def get_current_slice(self):
        """Get the current slice from the stack with rotation applied"""
        if len(self.dicom_stack) == 0:
            return self.create_placeholder()
        
        slice_img = self.dicom_stack[self.current_slice].copy()
        
        # Apply rotation
        if self.rotation_angle != 0:
            slice_img = self.rotate_image(slice_img, self.rotation_angle)
        
        return slice_img
    
    def next_slice(self):
        """Move to next slice"""
        if len(self.dicom_stack) > 1:
            self.current_slice = (self.current_slice + 1) % len(self.dicom_stack)
            print(f"Slice: {self.current_slice + 1}/{len(self.dicom_stack)}")
    
    def prev_slice(self):
        """Move to previous slice"""
        if len(self.dicom_stack) > 1:
            self.current_slice = (self.current_slice - 1) % len(self.dicom_stack)
            print(f"Slice: {self.current_slice + 1}/{len(self.dicom_stack)}")
    
    def reset_rotation(self):
        """Reset rotation to 0"""
        self.rotation_angle = 0
        print(f"Rotation reset: {self.rotation_angle}°")
    
    def get_face_region_scaled(self, landmarks, frame_shape):
        """Calculate face region with scaling and offset adjustments"""
        h, w = frame_shape[:2]
        
        # Get all landmark coordinates
        if USE_NEW_API:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        else:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        
        # Calculate face center and dimensions
        face_center_x = (min(x_coords) + max(x_coords)) / 2
        face_center_y = (min(y_coords) + max(y_coords)) / 2
        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)
        
        # Apply scale factor to determine overlay size
        overlay_width = int(face_width * self.scale_factor)
        overlay_height = int(face_height * self.scale_factor)
        
        # Calculate position with offsets
        x_min = int(face_center_x - overlay_width / 2 + self.horizontal_offset)
        x_max = int(face_center_x + overlay_width / 2 + self.horizontal_offset)
        y_min = int(face_center_y - overlay_height / 2 + self.vertical_offset)
        y_max = int(face_center_y + overlay_height / 2 + self.vertical_offset)
        
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
        
        # Check if crops are valid
        if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
            return background
        
        # Extract alpha channel and normalize
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
        
        print("\n" + "="*60)
        print("AR DICOM Face Mapping - Direct Face Overlay")
        print("Starting with 180° rotation and bounding box hidden")
        print("")
        print("Controls:")
        print("  'q'       - Quit")
        print("  '+'       - Increase overlay opacity")
        print("  '-'       - Decrease overlay opacity")
        print("  'RIGHT'   - Next slice")
        print("  'LEFT'    - Previous slice")
        print("  'UP'      - Jump forward 5 slices")
        print("  'DOWN'    - Jump backward 5 slices")
        print("")
        print("  SCALING & POSITIONING:")
        print("  '9'       - Increase overlay size")
        print("  '0'       - Decrease overlay size")
        print("  'i'       - Move overlay up")
        print("  'k'       - Move overlay down")
        print("  'j'       - Move overlay left")
        print("  'l'       - Move overlay right")
        print("  'h'       - Reset size and position")
        print("")
        print("  ROTATION:")
        print("  'r'       - Rotate 90° clockwise")
        print("  'e'       - Rotate 90° counter-clockwise")
        print("  't'       - Toggle 180° rotation")
        print("  'f'       - Reset rotation")
        print("")
        print("  DISPLAY:")
        print("  'b'       - Toggle bounding box")
        print("  'm'       - Toggle face landmarks")
        print("="*60 + "\n")
        
        alpha_value = 150  # Start with medium transparency
        show_bbox = False  # Hide bounding box by default
        show_landmarks = False
        
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
                # Get face region with scaling and offset adjustments
                x_min, y_min, x_max, y_max = self.get_face_region_scaled(
                    landmarks, frame.shape
                )
                
                # Get current slice from stack (with rotation applied)
                current_overlay = self.get_current_slice().copy()
                
                # Resize DICOM overlay to fit adjusted face region
                overlay_resized = cv2.resize(
                    current_overlay,
                    (x_max - x_min, y_max - y_min),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Update alpha channel for transparency control
                overlay_resized[:, :, 3] = (overlay_resized[:, :, 3].astype(float) * 
                                           (alpha_value / 255.0)).astype(np.uint8)
                
                # Overlay DICOM on face
                frame = self.overlay_image_alpha(
                    frame, overlay_resized, x_min, y_min
                )
                
                # Draw bounding box if enabled
                if show_bbox:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                (0, 255, 0), 2)
                
                # Draw landmarks if enabled
                if show_landmarks:
                    h, w = frame.shape[:2]
                    if USE_NEW_API:
                        for landmark in landmarks:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    else:
                        for landmark in landmarks:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            else:
                cv2.putText(frame, "Face not detected", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display status
            slice_info = f"Slice: {self.current_slice + 1}/{len(self.dicom_stack)}"
            cv2.putText(frame, slice_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Opacity: {int(alpha_value/255*100)}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Scale: {self.scale_factor:.2f}x", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rotation: {self.rotation_angle}°", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Offset: H={self.horizontal_offset} V={self.vertical_offset}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('DICOM Face Mapping', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Opacity controls
            elif key == ord('+') or key == ord('='):
                alpha_value = min(255, alpha_value + 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
            elif key == ord('-') or key == ord('_'):
                alpha_value = max(0, alpha_value - 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
            # Slice navigation
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                self.next_slice()
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                self.prev_slice()
            elif key == 82 or key == ord('w'):  # Up arrow or 'w'
                for _ in range(5):
                    self.next_slice()
            elif key == 84 or key == ord('x'):  # Down arrow or 'x'
                for _ in range(5):
                    self.prev_slice()
            # Scaling controls
            elif key == ord('9'):
                self.adjust_scale(0.1)
            elif key == ord('0'):
                self.adjust_scale(-0.1)
            # Position controls
            elif key == ord('i'):
                self.adjust_vertical_offset(-5)
            elif key == ord('k'):
                self.adjust_vertical_offset(5)
            elif key == ord('j'):
                self.adjust_horizontal_offset(-5)
            elif key == ord('l'):
                self.adjust_horizontal_offset(5)
            elif key == ord('h'):
                self.reset_adjustments()
            # Rotation controls
            elif key == ord('r'):
                self.rotate_90_cw()
            elif key == ord('e'):
                self.rotate_90_ccw()
            elif key == ord('t'):
                self.rotate_180()
            elif key == ord('f'):
                self.reset_rotation()
            # Display toggles
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"Bounding box: {'ON' if show_bbox else 'OFF'}")
            elif key == ord('m'):
                show_landmarks = not show_landmarks
                print(f"Face landmarks: {'ON' if show_landmarks else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # OPTION 1: Load from a folder containing multiple DICOM files
    DICOM_PATH = "D:/xyz/dicom_files"
    
    # OPTION 2: Load a single DICOM file (may contain multiple slices)
    # DICOM_PATH = "face_scan.dcm"
    
    try:
        ar_app = DicomFaceMapping(DICOM_PATH)
        ar_app.run()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Reinstall MediaPipe:")
        print("   pip uninstall mediapipe")
        print("   pip install mediapipe==0.10.9")
        print("\n2. Install other dependencies:")
        print("   pip install opencv-python pydicom numpy")