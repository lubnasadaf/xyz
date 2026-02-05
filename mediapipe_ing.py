import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_pose_estimation():
    # 0 usually refers to the default webcam (/dev/video0)
    # If you have multiple cameras, try changing this to 1 or 2
    cap = cv2.VideoCapture(0)

    # --- ADD THESE TWO LINES FOR 360p ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 
    # ------------------------------------
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Tip: Check if another application is using the camera or if your user is in the 'video' group.")
        sys.exit()

    # Setup MediaPipe Pose
    # min_detection_confidence: Threshold for the initial detection to be considered successful
    # min_tracking_confidence: Threshold to keep tracking the landmarks across frames
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        print("Press 'q' to quit.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            # MediaPipe expects RGB images, but OpenCV captures in BGR.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- Perform Pose Detection ---
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Body Pose (Debian)', cv2.flip(image, 1))

            # Break the loop when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation()