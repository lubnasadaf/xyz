import pydicom
import numpy as np
import pyvista as pv
import cv2
import os
import urllib.request
from PIL import Image

# MediaPipe Tasks imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - manual mode only")

class DicomFaceOverlayOptimized:
    """
    Optimized 3D DICOM overlay directly on webcam feed
    Better performance with pre-rendered frames
    """
    def __init__(self, folder_path, threshold_percentile=50):
        self.folder_path = folder_path
        self.threshold_percentile = threshold_percentile
        
        # 3D rendering
        self.surface = None
        self.rendered_frames = {}  # Cache rendered frames
        
        # Face detection
        self.face_landmarker = None
        if MEDIAPIPE_AVAILABLE:
            self.setup_face_landmarker()
        
        # Camera
        self.cap = None
        self.running = False
        
        # 3D object control
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.scale = 1.0
        self.alpha = 0.7  # Transparency
        
        # Mouse control
        self.mouse_down = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Mode
        self.auto_track = True
    
    def setup_face_landmarker(self):
        """Setup MediaPipe Face Landmarker"""
        try:
            model_path = 'face_landmarker.task'
            
            # Download if needed
            if not os.path.exists(model_path):
                print("Downloading face landmarker model...")
                url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            print("Face tracking enabled")
        except Exception as e:
            print(f"Face tracking setup failed: {e}")
    
    def load_dicom_volume(self):
        """Load DICOM files and create 3D surface"""
        print("Loading DICOM files...")
        
        slices = []
        for filename in sorted(os.listdir(self.folder_path)):
            filepath = os.path.join(self.folder_path, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                ds = pydicom.dcmread(filepath, force=True)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except:
                continue
        
        if len(slices) == 0:
            raise ValueError("No DICOM files found")
        
        # Sort slices
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except:
                pass
        
        print(f"Loaded {len(slices)} slices")
        
        # Create volume
        volume = np.stack([s.pixel_array for s in slices])
        
        # Get spacing
        try:
            pixel_spacing = slices[0].PixelSpacing
            slice_thickness = slices[0].SliceThickness
            spacing = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
        except:
            spacing = [1.0, 1.0, 1.0]
        
        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = volume.shape
        grid.spacing = spacing
        grid.point_data["values"] = volume.flatten(order="F")
        
        # Create surface
        threshold_value = np.percentile(volume, self.threshold_percentile)
        print(f"Threshold: {threshold_value}")
        
        self.surface = grid.contour([threshold_value])
        
        # Center and scale
        center = self.surface.center
        self.surface.translate(-np.array(center), inplace=True)
        
        # Normalize size
        bounds = self.surface.bounds
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        self.surface.points = self.surface.points / max_dim * 200  # Scale to 200 units
        
        print(f"Surface: {self.surface.n_points} points, {self.surface.n_cells} cells")
    
    def render_3d_view(self, width, height, rotation_x=0, rotation_y=0, rotation_z=0, scale=1.0):
        """Render 3D surface to image with given parameters"""
        # Create off-screen plotter
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        
        # Add surface
        plotter.add_mesh(
            self.surface,
            color='lightblue',
            opacity=0.9,
            smooth_shading=True,
            specular=0.5
        )
        
        # Set camera
        plotter.camera_position = 'iso'
        plotter.camera.azimuth = rotation_y
        plotter.camera.elevation = rotation_x
        plotter.camera.roll = rotation_z
        plotter.camera.zoom(scale)
        
        # Set background to white for better masking
        plotter.set_background('white')
        
        # Render
        img = plotter.screenshot(transparent_background=False, return_img=True)
        plotter.close()
        
        return img
    
    def create_mask_from_white(self, img):
        """Create alpha mask from white background"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create mask where white (255) becomes transparent
        # Everything not pure white becomes opaque
        mask = np.where(gray < 250, 255, 0).astype(np.uint8)
        
        return mask
    
    def detect_face_landmarks(self, frame):
        """Detect face landmarks"""
        if self.face_landmarker is None:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_landmarker.detect(mp_image)
            
            if result.face_landmarks:
                return result.face_landmarks[0]
        except:
            pass
        
        return None
    
    def get_face_bbox(self, face_landmarks, frame_shape):
        """Get bounding box of face"""
        if not face_landmarks:
            return None
        
        h, w = frame_shape[:2]
        
        # Get all x, y coordinates
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        
        # Get bounding box
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        
        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def overlay_image_alpha(self, background, overlay, x, y, alpha_mask):
        """Overlay image with alpha mask"""
        h, w = overlay.shape[:2]
        bh, bw = background.shape[:2]
        
        # Ensure overlay fits in background
        if y + h > bh:
            h = bh - y
            overlay = overlay[:h, :]
            alpha_mask = alpha_mask[:h, :]
        if x + w > bw:
            w = bw - x
            overlay = overlay[:, :w]
            alpha_mask = alpha_mask[:, :w]
        
        if h <= 0 or w <= 0:
            return background
        
        # Get region of interest
        roi = background[y:y+h, x:x+w]
        
        # Normalize alpha mask
        alpha = alpha_mask.astype(float) / 255.0 * self.alpha
        alpha = np.expand_dims(alpha, axis=2)
        
        # Blend
        blended = (alpha * overlay + (1 - alpha) * roi).astype(np.uint8)
        background[y:y+h, x:x+w] = blended
        
        return background
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
            self.last_mouse_x = x
            self.last_mouse_y = y
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scale *= 1.1
            else:
                self.scale *= 0.9
            self.scale = max(0.3, min(3.0, self.scale))
    
    def run(self, camera_index=0):
        """Main loop"""
        print("="*60)
        print("DICOM Face Overlay - Optimized Version")
        print("="*60)
        
        # Load DICOM
        print("\nLoading DICOM data...")
        self.load_dicom_volume()
        
        # Open webcam
        print("Opening webcam...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ret, test_frame = self.cap.read()
        if not ret:
            print("Error: Cannot read from webcam")
            return
        
        print(f"Webcam: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # Create window
        window_name = 'DICOM Face Overlay'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("CONTROLS")
        print("="*60)
        print("Mouse:")
        print("  • Drag: Rotate 3D model")
        print("  • Wheel: Zoom in/out")
        print("\nKeyboard:")
        print("  • A: Toggle auto-tracking")
        print("  • W/S: Rotate up/down")
        print("  • A/D: Rotate left/right")
        print("  • +/-: Zoom in/out")
        print("  • R: Reset rotation")
        print("  • T: Adjust transparency")
        print("  • Q/ESC: Quit")
        print("="*60 + "\n")
        
        self.running = True
        frame_count = 0
        
        # Pre-render initial view
        print("Rendering initial 3D view...")
        overlay_size = 400
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect face
            face_landmarks = None
            if self.face_landmarker and self.auto_track:
                face_landmarks = self.detect_face_landmarks(frame)
            
            # Render 3D overlay
            if face_landmarks and self.auto_track:
                # Get face bounding box
                bbox = self.get_face_bbox(face_landmarks, frame.shape)
                
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Calculate rotation based on face position
                    face_center_x = (x_min + x_max) / 2
                    face_width = x_max - x_min
                    
                    # Auto-rotation based on face position
                    auto_rot_y = ((face_center_x / w) - 0.5) * 30
                    
                    # Render 3D view
                    rendered = self.render_3d_view(
                        face_width,
                        y_max - y_min,
                        rotation_x=self.rotation_x + auto_rot_y * 0.3,
                        rotation_y=self.rotation_y + auto_rot_y,
                        rotation_z=self.rotation_z,
                        scale=self.scale
                    )
                    
                    # Create mask
                    mask = self.create_mask_from_white(rendered)
                    
                    # Convert rendered to BGR
                    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                    
                    # Overlay on face
                    frame = self.overlay_image_alpha(frame, rendered_bgr, x_min, y_min, mask)
                    
                    # Draw face box (optional)
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            else:
                # Manual mode - render at center
                rendered = self.render_3d_view(
                    overlay_size,
                    overlay_size,
                    rotation_x=self.rotation_x,
                    rotation_y=self.rotation_y,
                    rotation_z=self.rotation_z,
                    scale=self.scale
                )
                
                mask = self.create_mask_from_white(rendered)
                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                
                # Position at center
                x_pos = (w - overlay_size) // 2
                y_pos = (h - overlay_size) // 2
                
                frame = self.overlay_image_alpha(frame, rendered_bgr, x_pos, y_pos, mask)
            
            # Display status
            mode_text = "AUTO-TRACK" if self.auto_track else "MANUAL"
            color = (0, 255, 0) if self.auto_track else (0, 165, 255)
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if face_landmarks and self.auto_track:
                cv2.putText(frame, "Face Detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Alpha: {self.alpha:.1f} | Scale: {self.scale:.1f}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('a'):  # Toggle auto-track
                self.auto_track = not self.auto_track
                print(f"Auto-track: {'ON' if self.auto_track else 'OFF'}")
            elif key == ord('w'):  # Rotate up
                self.rotation_x += 5
            elif key == ord('s'):  # Rotate down
                self.rotation_x -= 5
            elif key == ord('d'):  # Rotate right
                self.rotation_y += 5
            elif key == ord('a'):  # Rotate left
                self.rotation_y -= 5
            elif key == ord('+') or key == ord('='):  # Zoom in
                self.scale *= 1.1
            elif key == ord('-') or key == ord('_'):  # Zoom out
                self.scale *= 0.9
            elif key == ord('r'):  # Reset
                self.rotation_x = 0
                self.rotation_y = 0
                self.rotation_z = 0
                self.scale = 1.0
            elif key == ord('t'):  # Toggle transparency
                self.alpha = 0.3 if self.alpha > 0.5 else 0.7
            
            self.scale = max(0.3, min(3.0, self.scale))
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.face_landmarker:
            self.face_landmarker.close()
        
        print(f"\nProcessed {frame_count} frames")
        print("Done!")

if __name__ == "__main__":
    try:
        overlay = DicomFaceOverlayOptimized(
            folder_path="D:/xyz/dicom_files",
            threshold_percentile=50
        )
        
        overlay.run(camera_index=0)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()