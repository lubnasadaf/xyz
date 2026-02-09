import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv

# --- 1. SETUP 3D MODEL LANDMARKS ---
# These are the coordinates you provided from your DICOM model.
# Format: [x, y, z]
model_points_3d = np.array([
    [330.25, 52.24, 170.25],   # Nose Tip
    [226.89, 71.91, 178.41],   # Chin
    [376.84, 97.25, 213.39],   # Left Eye Center
    [376.91, 97.84, 146.96],   # Right Eye Center
    # We omit ears/shoulders for the PnP solver to keep it robust using 
    # the strong central features MediaPipe detects best.
], dtype=np.float32)

# --- 2. SETUP MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # Critical for Iris centers
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Indices corresponding to the 3D points above
# Order must match model_points_3d: Nose, Chin, Left Eye, Right Eye
# 1 = Nose Tip, 152 = Chin, 473 = Left Iris, 468 = Right Iris
mp_indices = [1, 152, 473, 468]

# --- 3. LOAD THE MESH ---
# Replace with your actual file path. 
# If you have a folder of DICOMs, you likely need to convert to STL/OBJ first
# or use pyvista to read the volume and extract_surface().
# For this example, we create a dummy sphere to represent the head if file not found.
try:
    mesh = pv.read("path_to_your_model.stl") 
    print("Mesh loaded successfully.")
except:
    print("Mesh file not found. Creating a dummy mesh for demonstration.")
    mesh = pv.Sphere(radius=100, center=(300, 100, 150))

# Decimate mesh for faster real-time projection (optional)
# Overlaying 100k points is slow; 5k is fast.
if mesh.n_points > 5000:
    mesh = mesh.decimate_pro(0.9) 

mesh_points = np.array(mesh.points)

# --- 4. VIDEO LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for mirror view and convert color
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Extract 2D Image Points
            image_points_2d = []
            for idx in mp_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points_2d.append([x, y])
            
            image_points_2d = np.array(image_points_2d, dtype=np.float32)

            # Camera Matrix (Approximate if not calibrated)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

            # Solve PnP
            # This finds the rotation (rvec) and translation (tvec) that 
            # maps the 3D model points to the 2D image points.
            success_pnp, rvec, tvec = cv2.solvePnP(
                model_points_3d, 
                image_points_2d, 
                camera_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                # --- VISUALIZATION ---
                
                # Option A: Project the mesh points onto the image (Wireframe effect)
                # We project the 3D mesh vertices to 2D image coordinates
                projected_points, _ = cv2.projectPoints(
                    mesh_points, rvec, tvec, camera_matrix, dist_coeffs
                )
                
                # Draw the points
                projected_points = projected_points.reshape(-1, 2).astype(int)
                
                # Optimization: Draw only every 10th point to speed up rendering
                for p in projected_points[::10]: 
                    # Check if point is inside frame
                    if 0 <= p[0] < w and 0 <= p[1] < h:
                        cv2.circle(image, (p[0], p[1]), 1, (0, 255, 0), -1)

                # Draw the specific anchor points (Nose/Eyes) in Red for debugging
                for p in image_points_2d.astype(int):
                    cv2.circle(image, tuple(p), 5, (0, 0, 255), -1)

    cv2.imshow('DICOM Overlay', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.closeAll()