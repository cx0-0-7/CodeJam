import cv2
import sys

def run_camera(frame_processor=None):
    """
    Initializes the webcam, displays the feed, and handles cleanup.

    Args:
        frame_processor (callable, optional): A function to process each frame before display.
    """
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream or camera.")
        sys.exit(1)

    window_name = "Live Webcam Feed (Press 'q' or click X to close)"
    print(f"Opening window: '{window_name}'. Press 'q' key to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera.")
                continue

            # Apply optional processing function
            if frame_processor:
                frame = frame_processor(frame)

            cv2.imshow(window_name, frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting via 'q' key press.")
                break

            # Exit if window is closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Exiting via window close button ('X').")
                    break
            except cv2.error:
                print("Exiting: Window was closed externally.")
                break

    except KeyboardInterrupt:
        print("Exiting via KeyboardInterrupt (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and all windows closed.")


if __name__ == "__main__":
    run_camera()