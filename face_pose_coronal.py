import cv2
import numpy as np
import pydicom
import os
import glob
import serial
import serial.tools.list_ports

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except (ImportError, AttributeError):
    import mediapipe as mp
    USE_NEW_API = False


class DicomCoronalFaceMapping:
    def __init__(self, dicom_folder, use_arduino=True, arduino_port=None):
        if USE_NEW_API:
            self.init_new_mediapipe()
        else:
            self.init_legacy_mediapipe()
        
        # Load axial slices
        self.axial_stack = []
        self.coronal_stack = []
        self.current_coronal_slice = 0
        
        # Settings
        self.rotation_angle = 270
        self.scale_factor = 2.0
        self.vertical_offset = 80
        self.horizontal_offset = 0
        
        # Arduino
        self.use_arduino = use_arduino
        self.arduino = None
        self.current_distance = 0
        self.distance_status = "UNKNOWN"
        
        if self.use_arduino:
            self.init_arduino(arduino_port)
        
        # Load and process DICOM files
        self.load_dicom_folder(dicom_folder)
        self.create_coronal_view()
        
    def init_new_mediapipe(self):
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
        print("Using legacy MediaPipe API")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def init_arduino(self, port=None):
        try:
            if port is None:
                ports = serial.tools.list_ports.comports()
                arduino_ports = [p.device for p in ports if 
                                'Arduino' in p.description or 
                                'UNO' in p.description or
                                'CH340' in p.description]
                
                if not arduino_ports:
                    print("⚠ No Arduino found. Distance guidance disabled.")
                    self.use_arduino = False
                    return
                
                port = arduino_ports[0]
            
            self.arduino = serial.Serial(port, 9600, timeout=1)
            print(f"✓ Connected to Arduino on {port}")
            import time
            time.sleep(2)
            self.arduino.reset_input_buffer()
            
        except Exception as e:
            print(f"⚠ Arduino connection failed: {e}")
            self.use_arduino = False
    
    def read_distance(self):
        if not self.use_arduino or self.arduino is None:
            return
        
        try:
            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode('utf-8').strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        self.current_distance = int(parts[0])
                        self.distance_status = parts[1]
        except:
            pass
    
    def get_distance_color(self):
        if self.distance_status == "OPTIMAL":
            return (0, 255, 0)
        elif self.distance_status == "TOO_CLOSE":
            return (0, 165, 255)
        elif self.distance_status == "TOO_FAR":
            return (0, 0, 255)
        else:
            return (128, 128, 128)
    
    def get_distance_message(self):
        if self.distance_status == "OPTIMAL":
            return f"Perfect! Distance: {self.current_distance}cm"
        elif self.distance_status == "TOO_CLOSE":
            return f"Move BACK - Distance: {self.current_distance}cm"
        elif self.distance_status == "TOO_FAR":
            return f"Move CLOSER - Distance: {self.current_distance}cm"
        else:
            return "Measuring distance..."
    
    def load_dicom_folder(self, folder_path):
        """Load all axial DICOM slices from folder"""
        print(f"\nLoading DICOM files from: {folder_path}")
        
        # Find all .dcm files
        dicom_files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
        
        if not dicom_files:
            dicom_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*")) 
                                 if os.path.isfile(f)])
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        print(f"Found {len(dicom_files)} DICOM files")
        print("Loading axial slices (top view)...")
        
        # Load each slice
        for i, dicom_file in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(dicom_file)
                
                if not hasattr(ds, 'pixel_array'):
                    continue
                
                img = ds.pixel_array
                
                # Handle photometric interpretation
                if hasattr(ds, 'PhotometricInterpretation'):
                    if ds.PhotometricInterpretation == "MONOCHROME1":
                        img = np.max(img) - img
                
                # Apply modality LUT
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                
                # Normalize to 0-255
                img = img.astype(float)
                if img.max() > img.min():
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                else:
                    img = np.ones_like(img, dtype=np.uint8) * 128
                
                self.axial_stack.append(img)
                
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(dicom_files)} slices...")
                    
            except Exception as e:
                print(f"  Error loading {os.path.basename(dicom_file)}: {e}")
                continue
        
        if len(self.axial_stack) == 0:
            raise ValueError("No valid slices loaded")
        
        print(f"✓ Successfully loaded {len(self.axial_stack)} axial slices")
        print(f"  Slice dimensions: {self.axial_stack[0].shape}")
    
    def create_coronal_view(self):
        """Create coronal slices from axial stack"""
        print("\nCreating coronal view (front view)...")
        
        # Stack all axial slices into 3D volume
        volume = np.stack(self.axial_stack, axis=0)
        print(f"  Volume shape: {volume.shape} (slices, height, width)")
        
        # Extract coronal slices (front view)
        # Coronal view shows left-right and superior-inferior (top-bottom)
        # We need to iterate through the anterior-posterior (front-back) dimension
        num_coronal_slices = volume.shape[1]  # Height becomes number of coronal slices
        
        print(f"  Extracting {num_coronal_slices} coronal slices...")
        
        for i in range(num_coronal_slices):
            # Extract coronal slice at anterior-posterior position i
            # This takes a horizontal slice through all axial images at height i
            coronal_slice = volume[:, i, :]  # (depth, width)
            
            # Rotate 90 degrees counter-clockwise to correct orientation
            coronal_slice = cv2.rotate(coronal_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Convert to BGR for display
            coronal_bgr = cv2.cvtColor(coronal_slice, cv2.COLOR_GRAY2BGR)
            
            # Add alpha channel
            coronal_bgra = cv2.cvtColor(coronal_bgr, cv2.COLOR_BGR2BGRA)
            coronal_bgra[:, :, 3] = 255
            
            self.coronal_stack.append(coronal_bgra)
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{num_coronal_slices} coronal slices...")
        
        # Start at middle slice (front of face)
        self.current_coronal_slice = len(self.coronal_stack) // 2
        
        print(f"✓ Created {len(self.coronal_stack)} coronal slices")
        print(f"  Coronal slice dimensions: {self.coronal_stack[0].shape}")
        print(f"  Starting at slice: {self.current_coronal_slice + 1}")
        print(f"  Orientation: Front view (looking at face)")
        print(f"  Slider: Front (nose) ← → Back (skull)")
    
    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        return rotated
    
    def get_current_slice(self):
        if len(self.coronal_stack) == 0:
            return self.create_placeholder()
        
        slice_img = self.coronal_stack[self.current_coronal_slice].copy()
        
        if self.rotation_angle != 0:
            slice_img = self.rotate_image(slice_img, self.rotation_angle)
        
        return slice_img
    
    def create_placeholder(self):
        placeholder = np.zeros((400, 400, 4), dtype=np.uint8)
        cv2.putText(placeholder, "NO DATA", (120, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255, 255), 3)
        return placeholder
    
    def slider_callback(self, val):
        """Callback for slider - updates current slice"""
        self.current_coronal_slice = val
        # Don't print here to avoid spam when dragging slider
    
    def adjust_scale(self, increment):
        self.scale_factor = max(0.5, min(5.0, self.scale_factor + increment))
        print(f"Scale: {self.scale_factor:.2f}x")
    
    def adjust_vertical_offset(self, increment):
        self.vertical_offset += increment
    
    def adjust_horizontal_offset(self, increment):
        self.horizontal_offset += increment
    
    def next_slice(self):
        """Move to next coronal slice"""
        if len(self.coronal_stack) > 1:
            self.current_coronal_slice = min(self.current_coronal_slice + 1, 
                                            len(self.coronal_stack) - 1)
            print(f"Coronal Slice: {self.current_coronal_slice + 1}/{len(self.coronal_stack)}")
    
    def prev_slice(self):
        """Move to previous coronal slice"""
        if len(self.coronal_stack) > 1:
            self.current_coronal_slice = max(self.current_coronal_slice - 1, 0)
            print(f"Coronal Slice: {self.current_coronal_slice + 1}/{len(self.coronal_stack)}")
    
    def jump_forward(self, steps=5):
        """Jump forward multiple slices"""
        if len(self.coronal_stack) > 1:
            self.current_coronal_slice = min(self.current_coronal_slice + steps, 
                                            len(self.coronal_stack) - 1)
            print(f"Coronal Slice: {self.current_coronal_slice + 1}/{len(self.coronal_stack)}")
    
    def jump_backward(self, steps=5):
        """Jump backward multiple slices"""
        if len(self.coronal_stack) > 1:
            self.current_coronal_slice = max(self.current_coronal_slice - steps, 0)
            print(f"Coronal Slice: {self.current_coronal_slice + 1}/{len(self.coronal_stack)}")
    
    def rotate_90_cw(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def rotate_90_ccw(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def rotate_180(self):
        self.rotation_angle = (self.rotation_angle + 180) % 360
        print(f"Rotation: {self.rotation_angle}°")
    
    def reset_rotation(self):
        self.rotation_angle = 0
        print(f"Rotation reset: {self.rotation_angle}°")
    
    def get_face_region_scaled(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        if USE_NEW_API:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        else:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
        
        face_center_x = (min(x_coords) + max(x_coords)) / 2
        face_center_y = (min(y_coords) + max(y_coords)) / 2
        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)
        
        overlay_width = int(face_width * self.scale_factor)
        overlay_height = int(face_height * self.scale_factor)
        
        x_min = int(face_center_x - overlay_width / 2 + self.horizontal_offset)
        x_max = int(face_center_x + overlay_width / 2 + self.horizontal_offset)
        y_min = int(face_center_y - overlay_height / 2 + self.vertical_offset)
        y_max = int(face_center_y + overlay_height / 2 + self.vertical_offset)
        
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        return x_min, y_min, x_max, y_max
    
    def overlay_image_alpha(self, background, overlay, x, y):
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
            return background
        
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)
        
        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)
        
        overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        background_crop = background[y1:y2, x1:x2]
        
        if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
            return background
        
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        blended = (overlay_crop[:, :, :3] * alpha + 
                   background_crop * (1 - alpha)).astype(np.uint8)
        
        background[y1:y2, x1:x2] = blended
        return background
    
    def process_frame_new_api(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            return detection_result.face_landmarks[0]
        return None
    
    def process_frame_legacy_api(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("\n" + "="*60)
        print("DICOM Coronal Face Mapping with Slider")
        print("="*60)
        print("\nControls:")
        print("  NAVIGATION:")
        print("    'RIGHT' or 'd' - Next slice (go deeper into head)")
        print("    'LEFT'  or 'a' - Previous slice (go toward face front)")
        print("    'UP'    or 'w' - Jump forward 5 slices")
        print("    'DOWN'  or 's' - Jump backward 5 slices")
        print("    Slider         - Direct navigation through all slices")
        print("")
        print("  DISPLAY:")
        print("    '+'       - Increase opacity")
        print("    '-'       - Decrease opacity")
        print("    '9'       - Increase overlay size")
        print("    '0'       - Decrease overlay size")
        print("")
        print("  POSITION:")
        print("    'i'       - Move overlay up")
        print("    'k'       - Move overlay down")
        print("    'j'       - Move overlay left")
        print("    'l'       - Move overlay right")
        print("")
        print("  ROTATION:")
        print("    'r'       - Rotate 90° clockwise")
        print("    'e'       - Rotate 90° counter-clockwise")
        print("    't'       - Rotate 180°")
        print("    'f'       - Reset rotation")
        print("")
        print("  OTHER:")
        print("    'b'       - Toggle bounding box")
        print("    'q'       - Quit")
        print("="*60 + "\n")
        
        # Create window with slider
        window_name = 'DICOM Coronal Face Mapping'
        cv2.namedWindow(window_name)
        cv2.createTrackbar('Coronal Slice', window_name, 
                          self.current_coronal_slice, 
                          len(self.coronal_stack) - 1, 
                          self.slider_callback)
        
        alpha_value = 150
        show_bbox = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.use_arduino:
                self.read_distance()
            
            frame = cv2.flip(frame, 1)
            
            if USE_NEW_API:
                landmarks = self.process_frame_new_api(frame)
            else:
                landmarks = self.process_frame_legacy_api(frame)
            
            if landmarks:
                x_min, y_min, x_max, y_max = self.get_face_region_scaled(
                    landmarks, frame.shape
                )
                
                current_overlay = self.get_current_slice().copy()
                
                overlay_resized = cv2.resize(
                    current_overlay,
                    (x_max - x_min, y_max - y_min),
                    interpolation=cv2.INTER_LINEAR
                )
                
                overlay_resized[:, :, 3] = (overlay_resized[:, :, 3].astype(float) * 
                                           (alpha_value / 255.0)).astype(np.uint8)
                
                frame = self.overlay_image_alpha(
                    frame, overlay_resized, x_min, y_min
                )
                
                if show_bbox:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                (0, 255, 0), 2)
            
            # Distance indicator
            if self.use_arduino:
                distance_msg = self.get_distance_message()
                distance_color = self.get_distance_color()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), distance_color, -1)
                cv2.putText(frame, distance_msg, (10, 27), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Status display
            status_y = 70 if self.use_arduino else 30
            cv2.putText(frame, f"Coronal Slice: {self.current_coronal_slice + 1}/{len(self.coronal_stack)}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Opacity: {int(alpha_value/255*100)}%", 
                       (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Scale: {self.scale_factor:.2f}x", 
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            # IMPORTANT: Update slider position to match current slice
            # This syncs slider with keyboard navigation
            current_pos = cv2.getTrackbarPos('Coronal Slice', window_name)
            if current_pos != self.current_coronal_slice:
                cv2.setTrackbarPos('Coronal Slice', window_name, self.current_coronal_slice)
            
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
                print(f"Key pressed: RIGHT/d")
                self.next_slice()
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                print(f"Key pressed: LEFT/a")
                self.prev_slice()
            elif key == 82 or key == ord('w'):  # Up arrow or 'w'
                print(f"Key pressed: UP/w - Jumping forward 5 slices")
                self.jump_forward(5)
            elif key == 84 or key == ord('s'):  # Down arrow or 's'
                print(f"Key pressed: DOWN/s - Jumping backward 5 slices")
                self.jump_backward(5)
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
            # Rotation controls
            elif key == ord('r'):
                self.rotate_90_cw()
            elif key == ord('e'):
                self.rotate_90_ccw()
            elif key == ord('t'):
                self.rotate_180()
            elif key == ord('f'):
                self.reset_rotation()
            # Display toggle
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"Bounding box: {'ON' if show_bbox else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.use_arduino and self.arduino is not None:
            self.arduino.close()


if __name__ == "__main__":
    # Path to folder containing 409 axial DICOM slices (top view, 1mm spacing)
    DICOM_FOLDER = "D:/xyz/dicom_files"
    
    # Arduino settings
    USE_ARDUINO = True
    ARDUINO_PORT = None  # Auto-detect
    
    try:
        print("\n" + "="*60)
        print("DICOM Coronal View Face Mapping")
        print("Converts axial (top view) → coronal (front view)")
        print("="*60)
        
        app = DicomCoronalFaceMapping(DICOM_FOLDER, 
                                      use_arduino=USE_ARDUINO, 
                                      arduino_port=ARDUINO_PORT)
        app.run()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n\nSetup Instructions:")
        print("1. Install dependencies: pip install opencv-python pydicom numpy mediapipe pyserial")
        print("2. Set DICOM_FOLDER to your folder with 409 axial slices")
        print("3. Ensure slices are sorted by filename (001.dcm, 002.dcm, etc.)")