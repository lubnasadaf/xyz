import cv2
import sys

def open_debian_camera():
    # 0 is the default index for /dev/video0
    # cv2.CAP_V4L2 is the safest backend for Debian systems
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("troubleshoot: Run 'ls -l /dev/video0' to check if the device exists.")
        return

    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        # NOTE: This requires a GUI (Desktop Environment like GNOME/XFCE)
        cv2.imshow('Debian Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_debian_camera()