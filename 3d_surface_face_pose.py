import cv2
import numpy as np
import pydicom
import os
import glob
import pyvista as pv
from skimage import measure
import threading
import time

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except (ImportError, AttributeError):
    import mediapipe as mp
    USE_NEW_API = False


class PyVista3DFaceMapping:
    def __init__(self, dicom_folder):
        # Initialize MediaPipe
        if USE_NEW_API:
            self.init_new_mediapipe()
        else:
            self.init_legacy_mediapipe()
        
        # DICOM data
        self.axial_stack = []
        self.volume_3d = None
        self.mesh = None
        
        # Overlay positioning
        self.scale_factor = 5.0  # Larger for head to shoulders
        self.vertical_offset = 30
        self.horizontal_offset = 0
        
        # PyVista rendering
        self.plotter = None
        self.render_image = None
        self.rendering_active = True
        self.render_lock = threading.Lock()
        
        # 3D rotation controls (user adjustments)
        self.user_yaw = 0      # Left/right rotation (azimuth)
        self.user_pitch = 0    # Up/down rotation (elevation)
        self.user_roll = 0     # Tilt rotation
        self.zoom = 1.0
        
        # Fixed initial orientation (will be calibrated)
        self.base_yaw = 65
        self.base_pitch = 90   # Default, can be adjusted
        self.base_roll = -90    # Default, can be adjusted
        
        # Mouse state
        self.mouse_down = False
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_button = -1
        
        print("\n" + "="*70)
        print("PyVista 3D Surface Rendering - Face Mapping")
        print("="*70)
        
        # Load and process
        self.load_dicom_folder(dicom_folder)
        self.create_3d_surface()
        
        # Skip calibration - go directly to plotter
        self.init_pyvista_plotter()
        
        # Start rendering thread
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()
        
        print("\n✓ Initialization complete!")
        print(f"  Base orientation: Pitch={self.base_pitch}°, Roll={self.base_roll}°")
        print("  Ready for live face mapping\n")
        
    def init_new_mediapipe(self):
        print("Initializing MediaPipe (new API)...")
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def init_legacy_mediapipe(self):
        print("Initializing MediaPipe (legacy API)...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def load_dicom_folder(self, folder_path):
        """Load all DICOM slices"""
        print(f"\nLoading DICOM from: {folder_path}")
        
        dicom_files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
        if not dicom_files:
            dicom_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*")) 
                                 if os.path.isfile(f)])
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        print(f"Found {len(dicom_files)} files")
        
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
                
                self.axial_stack.append(img.astype(np.float32))
                
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(dicom_files)}")
                    
            except Exception as e:
                continue
        
        print(f"✓ Loaded {len(self.axial_stack)} slices")
        self.volume_3d = np.stack(self.axial_stack, axis=0)
    
    def create_3d_surface(self):
        """Create 3D mesh with marching cubes"""
        print("\nCreating 3D surface (may take 30-60 seconds)...")
        
        # Normalize
        volume_norm = self.volume_3d.copy()
        volume_norm = (volume_norm - volume_norm.min()) / (volume_norm.max() - volume_norm.min())
        
        # Marching cubes
        threshold = 0.3
        print(f"  Using threshold: {threshold}")
        verts, faces, normals, values = measure.marching_cubes(
            volume_norm, 
            level=threshold,
            step_size=1
        )
        
        print(f"  Generated {len(verts)} vertices, {len(faces)} faces")
        
        # Create mesh
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces])
        self.mesh = pv.PolyData(verts, faces_pv)
        self.mesh.compute_normals(inplace=True)
        
        print("✓ 3D surface created")
    
    def calibrate_orientation(self):
        """Interactive calibration to find correct orientation"""
        print("\n" + "="*70)
        print("ORIENTATION CALIBRATION")
        print("="*70)
        print("\nLet's find the correct orientation for your skull.")
        print("A window will open showing the 3D skull.")
        print("Adjust until it shows a FRONT VIEW (like looking at a face).\n")
        
        print("Starting values:")
        print(f"  Yaw:   {self.base_yaw}°")
        print(f"  Pitch: {self.base_pitch}°")
        print(f"  Roll:  {self.base_roll}°\n")
        
        # Create interactive calibration plotter
        cal_plotter = pv.Plotter(window_size=[1000, 1000])
        cal_plotter.add_mesh(self.mesh, color='white', smooth_shading=True)
        cal_plotter.background_color = 'black'
        
        # Apply current base orientation
        cal_plotter.camera.azimuth = self.base_yaw
        cal_plotter.camera.elevation = self.base_pitch
        cal_plotter.camera.roll = self.base_roll
        
        # Add text instructions
        instruction_text = cal_plotter.add_text(
            "CALIBRATION MODE\n\n"
            "Keyboard Controls:\n"
            "  W/S - Adjust Pitch (+10/-10)\n"
            "  A/D - Adjust Yaw (+10/-10)\n"
            "  Q/E - Adjust Roll (+10/-10)\n"
            "  Z/X - Fine adjust Pitch (+1/-1)\n"
            "  C/V - Fine adjust Yaw (+1/-1)\n"
            "  B/N - Fine adjust Roll (+1/-1)\n"
            "  R   - Reset to defaults\n"
            "  ENTER - Confirm and continue\n\n"
            f"Current: Y={self.base_yaw} P={self.base_pitch} R={self.base_roll}",
            position='upper_left',
            font_size=10,
            color='yellow'
        )
        
        print("Keyboard controls in calibration window:")
        print("  W/S - Pitch ±10°  |  Z/X - Pitch ±1°")
        print("  A/D - Yaw ±10°    |  C/V - Yaw ±1°")
        print("  Q/E - Roll ±10°   |  B/N - Roll ±1°")
        print("  R   - Reset to default (Yaw=0, Pitch=270, Roll=270)")
        print("  ENTER - Confirm and continue\n")
        print("Adjust until skull faces FORWARD, then press ENTER\n")
        
        # Calibration loop
        def key_press_callback(obj, event):
            key = obj.GetKeySym()
            
            if key == 'w':
                self.base_pitch += 10
            elif key == 's':
                self.base_pitch -= 10
            elif key == 'a':
                self.base_yaw -= 10
            elif key == 'd':
                self.base_yaw += 10
            elif key == 'q':
                self.base_roll -= 10
            elif key == 'e':
                self.base_roll += 10
            elif key == 'z':
                self.base_pitch += 1
            elif key == 'x':
                self.base_pitch -= 1
            elif key == 'c':
                self.base_yaw -= 1
            elif key == 'v':
                self.base_yaw += 1
            elif key == 'b':
                self.base_roll -= 1
            elif key == 'n':
                self.base_roll += 1
            elif key == 'r':
                self.base_yaw = 0
                self.base_pitch = 270
                self.base_roll = 270
                print("Reset to defaults")
            elif key == 'Return':
                cal_plotter.close()
                return
            
            # Update camera
            cal_plotter.camera.azimuth = self.base_yaw
            cal_plotter.camera.elevation = self.base_pitch
            cal_plotter.camera.roll = self.base_roll
            
            # Update text
            instruction_text.SetText(
                2,
                "CALIBRATION MODE\n\n"
                "Keyboard Controls:\n"
                "  W/S - Adjust Pitch (+10/-10)\n"
                "  A/D - Adjust Yaw (+10/-10)\n"
                "  Q/E - Adjust Roll (+10/-10)\n"
                "  Z/X - Fine adjust Pitch (+1/-1)\n"
                "  C/V - Fine adjust Yaw (+1/-1)\n"
                "  B/N - Fine adjust Roll (+1/-1)\n"
                "  R   - Reset to defaults\n"
                "  ENTER - Confirm and continue\n\n"
                f"Current: Y={self.base_yaw} P={self.base_pitch} R={self.base_roll}"
            )
            
            # Print to console
            print(f"Yaw={self.base_yaw}° Pitch={self.base_pitch}° Roll={self.base_roll}°")
            
            # Re-render
            cal_plotter.render()
        
        cal_plotter.iren.add_observer('KeyPressEvent', key_press_callback)
        cal_plotter.show()
        
        print(f"\n✓ Calibration complete!")
        print(f"  Final orientation:")
        print(f"    Yaw:   {self.base_yaw}°")
        print(f"    Pitch: {self.base_pitch}°")
        print(f"    Roll:  {self.base_roll}°")
        print("\nThese angles will be used for live face mapping.\n")
    
    def init_pyvista_plotter(self):
        """Setup off-screen renderer"""
        print("\nInitializing PyVista renderer...")
        
        self.plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024])
        self.plotter.add_mesh(
            self.mesh,
            color='white',
            opacity=1.0,
            smooth_shading=True
        )
        self.plotter.background_color = 'black'
        
        # Set initial camera with CORRECT orientation
        self.plotter.camera.elevation = self.base_pitch  # Pitch = elevation in PyVista
        self.plotter.camera.roll = self.base_roll
        self.plotter.camera.zoom(1.0)
        
        print("✓ Renderer ready")
    
    def render_loop(self):
        """Background rendering thread"""
        while self.rendering_active:
            try:
                with self.render_lock:
                    # Reset to base orientation
                    self.plotter.reset_camera()
                    # PyVista mapping: azimuth=yaw, elevation=pitch, roll=roll
                    self.plotter.camera.elevation = self.base_pitch + self.user_pitch
                    self.plotter.camera.azimuth = self.base_yaw + self.user_yaw
                    self.plotter.camera.roll = self.base_roll + self.user_roll
                    self.plotter.camera.zoom(self.zoom)
                    
                    # Render
                    img = self.plotter.screenshot(return_img=True, transparent_background=True)
                    self.render_image = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Render error: {e}")
                time.sleep(0.1)
    
    def get_current_render(self):
        """Get latest render"""
        with self.render_lock:
            if self.render_image is not None:
                return self.render_image.copy()
            return np.zeros((1024, 1024, 4), dtype=np.uint8)
    
    def get_face_region(self, landmarks, frame_shape):
        """Calculate region from eyes to shoulders"""
        h, w = frame_shape[:2]
        
        if USE_NEW_API:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
            
            # Get specific eye landmarks for more accurate positioning
            # Left eye: landmarks 33, 133 (corners)
            # Right eye: landmarks 362, 263 (corners)
            left_eye_y = (landmarks[33].y + landmarks[133].y) / 2 * h
            right_eye_y = (landmarks[362].y + landmarks[263].y) / 2 * h
            eye_level = (left_eye_y + right_eye_y) / 2
        else:
            x_coords = [landmark.x * w for landmark in landmarks]
            y_coords = [landmark.y * h for landmark in landmarks]
            
            # Legacy API - estimate eye level
            face_top = min(y_coords)
            face_height = max(y_coords) - min(y_coords)
            eye_level = face_top + (face_height * 0.25)  # Eyes are ~25% from top
        
        # Face dimensions
        face_left = min(x_coords)
        face_right = max(x_coords)
        face_bottom = max(y_coords)
        face_center_x = (face_left + face_right) / 2
        face_width = face_right - face_left
        face_height = face_bottom - min(y_coords)
        
        # Width: Scale for shoulders (wider than face)
        overlay_width = int(face_width * self.scale_factor)
        
        # Height: From eyes down to shoulders
        # Shoulders are roughly 2-2.5x face height below eyes
        overlay_height = int(face_height * self.scale_factor * 1.5)
        
        # Horizontal positioning (centered on face)
        x_min = int(face_center_x - overlay_width / 2 + self.horizontal_offset)
        x_max = int(face_center_x + overlay_width / 2 + self.horizontal_offset)
        
        # Vertical positioning (starts at eyes, extends down to shoulders)
        # Small overlap above eyes (10%) for better alignment
        # Large extension below (90%) for shoulders
        y_min = int(eye_level - overlay_height * 0.1 + self.vertical_offset)
        y_max = int(eye_level + overlay_height * 0.9 + self.vertical_offset)
        
        # Bounds check
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        return x_min, y_min, x_max, y_max
    
    def overlay_image_alpha(self, background, overlay, x, y):
        """Alpha blend overlay onto background"""
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
        result = self.detector.detect(mp_image)
        return result.face_landmarks[0] if result.face_landmarks else None
    
    def process_frame_legacy_api(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.mouse_button = 0
            self.mouse_last_x = x
            self.mouse_last_y = y
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_down = True
            self.mouse_button = 1
            self.mouse_last_x = x
            self.mouse_last_y = y
        
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            dx = x - self.mouse_last_x
            dy = y - self.mouse_last_y
            
            if self.mouse_button == 0:  # Left - rotate (yaw/pitch)
                self.user_yaw += dx * 0.5
                self.user_pitch -= dy * 0.5
                print(f"Rotation: Yaw={self.user_yaw:.0f}° Pitch={self.user_pitch:.0f}°")
            
            elif self.mouse_button == 1:  # Right - roll
                self.user_roll += dx * 0.5
                print(f"Roll: {self.user_roll:.0f}°")
            
            self.mouse_last_x = x
            self.mouse_last_y = y
        
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.mouse_down = False
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom *= 1.05
            else:
                self.zoom *= 0.95
            self.zoom = np.clip(self.zoom, 0.3, 3.0)
            print(f"Zoom: {self.zoom:.2f}x")
    
    def run(self):
        """Main video loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("\n" + "="*70)
        print("CONTROLS:")
        print("="*70)
        print("\nMOUSE:")
        print("  Left Drag    - Rotate 3D (yaw/pitch)")
        print("  Right Drag   - Roll rotation")
        print("  Wheel        - Zoom")
        print("\nKEYBOARD:")
        print("  9 / 0        - Increase/Decrease overlay size")
        print("  i / k        - Move overlay up/down")
        print("  j / l        - Move overlay left/right")
        print("  r            - Reset rotation")
        print("  c            - Reset position/size")
        print("  p            - Print current settings")
        print("  q            - Quit")
        print("="*70 + "\n")
        
        window_name = 'PyVista 3D Face Mapping'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        alpha = 200
        time.sleep(1)  # Wait for first render
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect face
            if USE_NEW_API:
                landmarks = self.process_frame_new_api(frame)
            else:
                landmarks = self.process_frame_legacy_api(frame)
            
            if landmarks:
                # Get overlay region
                x_min, y_min, x_max, y_max = self.get_face_region(landmarks, frame.shape)
                
                # Get 3D render
                render_3d = self.get_current_render()
                
                # Resize to fit
                overlay = cv2.resize(render_3d, (x_max - x_min, y_max - y_min),
                                    interpolation=cv2.INTER_LINEAR)
                
                # Apply alpha
                overlay[:, :, 3] = (overlay[:, :, 3].astype(float) * 
                                   (alpha / 255.0)).astype(np.uint8)
                
                # Overlay on video
                frame = self.overlay_image_alpha(frame, overlay, x_min, y_min)
            
            # Display info
            cv2.putText(frame, "PyVista 3D Face Mapping", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rot: Yaw={self.user_yaw:.0f} Pitch={self.user_pitch:.0f} Roll={self.user_roll:.0f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Zoom: {self.zoom:.2f} | Scale: {self.scale_factor:.2f} | Offset: ({self.horizontal_offset},{self.vertical_offset})", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Mouse: L-Rotate | R-Roll | Wheel-Zoom", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('9'):
                self.scale_factor += 0.1
                print(f"Scale: {self.scale_factor:.2f}")
            elif key == ord('0'):
                self.scale_factor = max(0.5, self.scale_factor - 0.1)
                print(f"Scale: {self.scale_factor:.2f}")
            elif key == ord('i'):
                self.vertical_offset -= 10
                print(f"Vertical offset: {self.vertical_offset}")
            elif key == ord('k'):
                self.vertical_offset += 10
                print(f"Vertical offset: {self.vertical_offset}")
            elif key == ord('j'):
                self.horizontal_offset -= 10
                print(f"Horizontal offset: {self.horizontal_offset}")
            elif key == ord('l'):
                self.horizontal_offset += 10
                print(f"Horizontal offset: {self.horizontal_offset}")
            elif key == ord('r'):
                self.user_yaw = 0
                self.user_pitch = 0
                self.user_roll = 0
                self.zoom = 1.0
                print("Rotation reset")
            elif key == ord('c'):
                self.scale_factor = 3.0
                self.vertical_offset = 0
                self.horizontal_offset = 0
                print("Position/size reset")
            elif key == ord('p'):
                print(f"\nCurrent settings:")
                print(f"  Scale: {self.scale_factor:.2f}")
                print(f"  Offset: H={self.horizontal_offset}, V={self.vertical_offset}")
                print(f"  Rotation: Yaw={self.user_yaw:.0f}, Pitch={self.user_pitch:.0f}, Roll={self.user_roll:.0f}")
                print(f"  Zoom: {self.zoom:.2f}\n")
        
        self.rendering_active = False
        cap.release()
        cv2.destroyAllWindows()
        self.plotter.close()


if __name__ == "__main__":
    DICOM_FOLDER = "D:/xyz/dicom_files"
    
    try:
        app = PyVista3DFaceMapping(DICOM_FOLDER)
        app.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()