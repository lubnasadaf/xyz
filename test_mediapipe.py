"""
MediaPipe Tasks API Test Script
Tests the new MediaPipe Tasks API (version 0.10+)
"""

import sys
import os

print("="*60)
print("MediaPipe Tasks API Diagnostic")
print("="*60)

# Test 1: Check MediaPipe version
print("\n1. Checking MediaPipe installation...")
try:
    import mediapipe as mp
    version = mp.__version__
    print(f"   ✓ MediaPipe version: {version}")
    
    major, minor = map(int, version.split('.')[:2])
    if major == 0 and minor >= 10:
        print(f"   ✓ Version {version} uses Tasks API")
    else:
        print(f"   ⚠ Version {version} may use old Solutions API")
except ImportError:
    print("   ✗ MediaPipe not installed")
    print("   Install: pip install mediapipe")
    sys.exit(1)

# Test 2: Check Tasks API
print("\n2. Checking MediaPipe Tasks API...")
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("   ✓ Tasks API accessible")
except ImportError as e:
    print(f"   ✗ Cannot import Tasks API: {e}")
    print("   Try: pip install mediapipe --upgrade")
    sys.exit(1)

# Test 3: Download face landmarker model
print("\n3. Checking face landmarker model...")
model_path = 'face_landmarker.task'

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✓ Model exists: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"   ⚠ Model not found, downloading...")
    try:
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        
        print("   Downloading from Google...")
        urllib.request.urlretrieve(url, model_path)
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✓ Model downloaded: {size_mb:.1f} MB")
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
        print("   Manual download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# Test 4: Initialize Face Landmarker
print("\n4. Testing Face Landmarker initialization...")
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    if not os.path.exists(model_path):
        print("   ✗ Model file missing, skipping test")
    else:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("   ✓ Face Landmarker initialized successfully")
        landmarker.close()
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")

# Test 5: Test with webcam
print("\n5. Testing face detection with webcam...")
try:
    import cv2
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ⚠ Cannot open webcam")
    else:
        ret, frame = cap.read()
        if not ret:
            print("   ⚠ Cannot read from webcam")
        else:
            print(f"   ✓ Webcam accessible: {frame.shape[1]}x{frame.shape[0]}")
            
            if os.path.exists(model_path):
                # Initialize landmarker
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                    min_face_detection_confidence=0.5
                )
                landmarker = vision.FaceLandmarker.create_from_options(options)
                
                # Detect face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                result = landmarker.detect(mp_image)
                
                if result.face_landmarks:
                    num_landmarks = len(result.face_landmarks[0])
                    print(f"   ✓ Face detected with {num_landmarks} landmarks!")
                else:
                    print("   ⚠ No face detected (make sure you're visible)")
                
                landmarker.close()
            
        cap.release()
except Exception as e:
    print(f"   ✗ Webcam test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check other dependencies
print("\n6. Checking other dependencies...")

deps = {
    'opencv-python': 'cv2',
    'pyvista': 'pyvista',
    'pydicom': 'pydicom',
    'numpy': 'numpy'
}

for package_name, import_name in deps.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {package_name}: {version}")
    except ImportError:
        print(f"   ✗ {package_name} not installed")
        print(f"      Install: pip install {package_name}")

print("\n" + "="*60)
print("Diagnostic Complete!")
print("="*60)

print("\n📋 Next Steps:")
print("\nIf all tests passed:")
print("  → Run: python dicom_face_overlay_tasks.py")
print("\nIf face detection test failed but model downloaded:")
print("  → Position yourself in front of the webcam")
print("  → Ensure good lighting")
print("\nIf model download failed:")
print("  → Download manually from:")
print("    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
print("  → Place in same folder as the script")

print(f"\nYour Python version: {sys.version}")
print("MediaPipe Tasks requires Python 3.8-3.11")