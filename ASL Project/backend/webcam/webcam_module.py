import cv2
<<<<<<< HEAD
import mediapipe as mp
import numpy as np
=======
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
>>>>>>> 62a50cea118f67bb3163b575d8d20b5f06a9211b

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Function to classify basic gestures
def classify_gesture(landmarks):
    """
    Classifies hand gestures: Thumbs Up, Thumbs Down, and Fist
    """
    try:
        # Normalize using wrist as reference
        wrist = landmarks[0]
        
        # Normalize coordinates
        normalized_landmarks = np.array([
            [(lm.x - wrist.x), (lm.y - wrist.y)] for lm in landmarks
        ])

        # Distance calculations
        thumb_index_dist = np.linalg.norm(normalized_landmarks[4] - normalized_landmarks[8])
        thumb_pinky_dist = np.linalg.norm(normalized_landmarks[4] - normalized_landmarks[20])

        # **Gesture Classification**
        # Fist → All fingers are close together
        if thumb_index_dist < 0.1 and thumb_pinky_dist < 0.2:
            return "Fist "

        # Thumbs Up → Thumb above MCP joint
        elif normalized_landmarks[4][1] < normalized_landmarks[2][1]:
            return "Thumbs Up "

        # Thumbs Down → Thumb below MCP joint
        elif normalized_landmarks[4][1] > normalized_landmarks[2][1]:
            return "Thumbs Down "

        else:
            return "Unknown"
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Unknown"

# Real-time Webcam Feed
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 1280)   # Width
cap.set(4, 720)    # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the image for natural viewing
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and classify landmarks
            gesture = classify_gesture(hand_landmarks.landmark)

            # Display detected gesture
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()