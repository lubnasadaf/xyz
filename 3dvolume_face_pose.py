import cv2
import numpy as np
import pydicom
import os
import glob
import serial
import serial.tools.list_ports
from scipy import ndimage

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except (ImportError, AttributeError):
    import mediapipe as mp
    USE_NEW_API = False


class Dicom3DFaceMapping:
    def __init__(self, dicom_folder, use_arduino=True, arduino_port=None):
        if USE_NEW_API:
            self.init_new_mediapipe()
        else:
            self.init_legacy_mediapipe()
        
        # Load axial slices
        self.axial_stack = []
        self.volume_3d = None
        
        # Settings
        self.rotation_angle = 180
        self.scale_factor = 2.5
        self.vertical_offset = 0
        self.horizontal_offset = 0
        
        # 3D rendering settings
        self.render_angle_x = 0  # Pitch (up/down rotation)
        self.render_angle_y = 0  # Yaw (left/right rotation)
        self.render_angle_z = 0  # Roll (tilt)
        self.render_depth = 0.5  # Depth into volume for MIP (0.0 to 1.0)
        self.threshold = 100  # Intensity threshold for 3D rendering
        
        # Arduino
        self.use_arduino = use_arduino
        self.arduino = None
        self.current_distance = 0
        self.distance_status = "UNKNOWN"
        
        if self.use_arduino:
            self.init_arduino(arduino_port)
        
        # Load and process DICOM files
        self.load_dicom_folder(dicom_folder)
        self.create_3d_volume()
        
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
        
        dicom_files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
        
        if not dicom_files:
            dicom_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*")) 
                                 if os.path.isfile(f)])
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        print(f"Found {len(dicom_files)} DICOM files")
        print("Loading slices for 3D reconstruction...")
        
        for i, dicom_file in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(dicom_file)
                
                if not hasattr(ds, 'pixel_array'):
                    continue
                
                img = ds.pixel_array
                
                if hasattr(ds, 'PhotometricInterpretation'):
                    if ds.PhotometricInterpretation == "MONOCHROME1":
                        img = np.max(img) - img
                
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
        
        print(f"✓ Successfully loaded {len(self.axial_stack)} slices")
    
    def create_3d_volume(self):
        """Create 3D volume from axial stack"""
        print("\nCreating 3D volume...")
        
        # Stack all slices into 3D volume
        self.volume_3d = np.stack(self.axial_stack, axis=0)
        print(f"  Volume shape: {self.volume_3d.shape} (depth, height, width)")
        print(f"  Volume size: {self.volume_3d.nbytes / (1024*1024):.1f} MB")
        print("✓ 3D volume ready for rendering")
    
    def render_3d_mip(self, angle_x=0, angle_y=0, angle_z=0):
        """
        Render 3D volume using Maximum Intensity Projection (MIP)
        with rotation
        """
        # Rotate volume
        rotated_volume = self.volume_3d.copy()
        
        # Apply rotations (around center of volume)
        if angle_x != 0:
            rotated_volume = ndimage.rotate(rotated_volume, angle_x, axes=(1, 0), 
                                           reshape=False, order=1)
        if angle_y != 0:
            rotated_volume = ndimage.rotate(rotated_volume, angle_y, axes=(2, 0), 
                                           reshape=False, order=1)
        if angle_z != 0:
            rotated_volume = ndimage.rotate(rotated_volume, angle_z, axes=(2, 1), 
                                           reshape=False, order=1)
        
        # Apply threshold to highlight bone/high-density structures
        rotated_volume = np.where(rotated_volume > self.threshold, 
                                   rotated_volume, 0)
        
        # Maximum Intensity Projection along depth axis
        mip_image = np.max(rotated_volume, axis=0)
        
        # Normalize to 0-255
        if mip_image.max() > 0:
            mip_image = ((mip_image / mip_image.max()) * 255).astype(np.uint8)
        else:
            mip_image = np.zeros_like(mip_image, dtype=np.uint8)
        
        # Convert to BGRA
        mip_bgr = cv2.cvtColor(mip_image, cv2.COLOR_GRAY2BGR)
        mip_bgra = cv2.cvtColor(mip_bgr, cv2.COLOR_BGR2BGRA)
        mip_bgra[:, :, 3] = 255
        
        return mip_bgra
    
    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        return rotated
    
    def get_current_3d_render(self):
        """Get current 3D rendering with applied rotations"""
        rendered = self.render_3d_mip(self.render_angle_x, 
                                       self.render_angle_y, 
                                       self.render_angle_z)
        
        if self.rotation_angle != 0:
            rendered = self.rotate_image(rendered, self.rotation_angle)
        
        return rendered
    
    def adjust_scale(self, increment):
        self.scale_factor = max(0.5, min(5.0, self.scale_factor + increment))
        print(f"Scale: {self.scale_factor:.2f}x")
    
    def adjust_vertical_offset(self, increment):
        self.vertical_offset += increment
    
    def adjust_horizontal_offset(self, increment):
        self.horizontal_offset += increment
    
    def adjust_threshold(self, increment):
        self.threshold = max(0, min(255, self.threshold + increment))
        print(f"3D Threshold: {self.threshold} (higher = show only dense structures)")
    
    def rotate_3d_x(self, increment):
        self.render_angle_x = (self.render_angle_x + increment) % 360
        print(f"3D Rotation X (Pitch): {self.render_angle_x}°")
    
    def rotate_3d_y(self, increment):
        self.render_angle_y = (self.render_angle_y + increment) % 360
        print(f"3D Rotation Y (Yaw): {self.render_angle_y}°")
    
    def rotate_3d_z(self, increment):
        self.render_angle_z = (self.render_angle_z + increment) % 360
        print(f"3D Rotation Z (Roll): {self.render_angle_z}°")
    
    def rotate_90_cw(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        print(f"2D Rotation: {self.rotation_angle}°")
    
    def rotate_90_ccw(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"2D Rotation: {self.rotation_angle}°")
    
    def rotate_180(self):
        self.rotation_angle = (self.rotation_angle + 180) % 360
        print(f"2D Rotation: {self.rotation_angle}°")
    
    def reset_rotation(self):
        self.rotation_angle = 0
        self.render_angle_x = 0
        self.render_angle_y = 0
        self.render_angle_z = 0
        print("All rotations reset")
    
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
        print("DICOM 3D Volume Rendering Face Mapping")
        print("="*60)
        print("\nControls:")
        print("  DISPLAY:")
        print("    '+'       - Increase opacity")
        print("    '-'       - Decrease opacity")
        print("    '9'       - Increase overlay size")
        print("    '0'       - Decrease overlay size")
        print("")
        print("  3D ROTATION:")
        print("    'u/U'     - Rotate 3D up/down (pitch)")
        print("    'y/Y'     - Rotate 3D left/right (yaw)")
        print("    'h/H'     - Rotate 3D tilt (roll)")
        print("    'v'       - Increase bone threshold (show only dense)")
        print("    'V'       - Decrease bone threshold (show more)")
        print("")
        print("  POSITION:")
        print("    'i/k/j/l' - Move overlay (up/down/left/right)")
        print("")
        print("  2D ROTATION:")
        print("    'r/e/t/f' - 2D rotation controls")
        print("")
        print("  OTHER:")
        print("    'b'       - Toggle bounding box")
        print("    'q'       - Quit")
        print("="*60 + "\n")
        print("Rendering 3D volume... (this may take a moment per frame)")
        
        window_name = 'DICOM 3D Face Mapping'
        cv2.namedWindow(window_name)
        
        alpha_value = 150
        show_bbox = False
        
        # Pre-render initial 3D view
        print("Generating initial 3D render...")
        current_3d = self.get_current_3d_render()
        print("Ready!\n")
        
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
                
                # Use pre-rendered 3D view
                overlay_resized = cv2.resize(
                    current_3d,
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
            cv2.putText(frame, "3D Volume Rendering", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Opacity: {int(alpha_value/255*100)}%", 
                       (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Scale: {self.scale_factor:.2f}x", 
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {self.threshold}", 
                       (10, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"3D Rot: X{self.render_angle_x}° Y{self.render_angle_y}° Z{self.render_angle_z}°", 
                       (10, status_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            needs_rerender = False
            
            if key == ord('q'):
                break
            # Opacity controls
            elif key == ord('+') or key == ord('='):
                alpha_value = min(255, alpha_value + 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
            elif key == ord('-') or key == ord('_'):
                alpha_value = max(0, alpha_value - 20)
                print(f"Opacity: {int(alpha_value/255*100)}%")
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
            # 3D rotation controls
            elif key == ord('u'):
                self.rotate_3d_x(-5)
                needs_rerender = True
            elif key == ord('U'):
                self.rotate_3d_x(5)
                needs_rerender = True
            elif key == ord('y'):
                self.rotate_3d_y(-5)
                needs_rerender = True
            elif key == ord('Y'):
                self.rotate_3d_y(5)
                needs_rerender = True
            elif key == ord('h'):
                self.rotate_3d_z(-5)
                needs_rerender = True
            elif key == ord('H'):
                self.rotate_3d_z(5)
                needs_rerender = True
            # Threshold controls
            elif key == ord('v'):
                self.adjust_threshold(10)
                needs_rerender = True
            elif key == ord('V'):
                self.adjust_threshold(-10)
                needs_rerender = True
            # 2D rotation controls
            elif key == ord('r'):
                self.rotate_90_cw()
                needs_rerender = True
            elif key == ord('e'):
                self.rotate_90_ccw()
                needs_rerender = True
            elif key == ord('t'):
                self.rotate_180()
                needs_rerender = True
            elif key == ord('f'):
                self.reset_rotation()
                needs_rerender = True
            # Display toggle
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"Bounding box: {'ON' if show_bbox else 'OFF'}")
            
            # Re-render 3D if needed
            if needs_rerender:
                print("Re-rendering 3D volume...")
                current_3d = self.get_current_3d_render()
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.use_arduino and self.arduino is not None:
            self.arduino.close()


if __name__ == "__main__":
    # Path to folder containing axial DICOM slices
    DICOM_FOLDER = "D:/xyz/dicom_files"
    
    # Arduino settings
    USE_ARDUINO = True
    ARDUINO_PORT = None
    
    try:
        print("\n" + "="*60)
        print("DICOM 3D Volume Rendering")
        print("Maximum Intensity Projection (MIP)")
        print("="*60)
        
        # Check for scipy
        try:
            from scipy import ndimage
        except ImportError:
            print("\n⚠ SciPy not found!")
            print("Install with: pip install scipy")
            exit(1)
        
        app = Dicom3DFaceMapping(DICOM_FOLDER, 
                                 use_arduino=USE_ARDUINO, 
                                 arduino_port=ARDUINO_PORT)
        app.run()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n\nSetup:")
        print("pip install opencv-python pydicom numpy mediapipe scipy pyserial")