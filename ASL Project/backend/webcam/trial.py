import cv2
import mediapipe as mp
import time
import sys
import numpy as np

# --- Configuration ---
WINDOW_NAME = "Pure Landmark Extractor (Press 'q' to quit)"
# Use high confidence for clean, accurate data
DETECTION_CONFIDENCE = 0.8 

# Initialize MediaPipe objects globally for simplicity in this pure module
mpHands = mp.solutions.hands
# Note: Using keyword arguments for robustness, even in procedural code
hands = mpHands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=DETECTION_CONFIDENCE
)

def extract_normalized_landmarks(img, results):
    """
    Extracts and flattens the 3D normalized coordinates (x, y, z) for one hand.
    
    Returns:
        np.array: A flat array of 63 values (21 landmarks * 3 dimensions) 
                  or an empty array if no hand is detected.
    """
    lm_array = np.array([])
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # Get the first detected hand
        myHand = results.multi_hand_landmarks[0] 
        lm_data = []
        
        # Extract 3D Normalized Coordinates (x, y, z)
        for lm in myHand.landmark:
            # We explicitly collect the normalized coordinates (0.0 to 1.0)
            lm_data.extend([lm.x, lm.y, lm.z]) 
        
        lm_array = np.array(lm_data)
        
    return lm_array


def main():
    """Main function to run the webcam stream and purely extract normalized data."""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream or camera.")
        sys.exit(1)
        
    print(f"Opening window: '{WINDOW_NAME}'.")
    print("When a hand is visible, the 63-value normalized vector will be printed.")
    print("Press 'q' key or Ctrl+C to quit.")

    try:
        while True:
            success, img = cap.read()
            if not success:
                cv2.waitKey(10)
                continue
            
            # 1. Process the frame (no drawing needed)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            
            # 2. Extract the normalized vector
            landmark_vector = extract_normalized_landmarks(img, results)
            
            if len(landmark_vector) == 63:
                # The raw, clean, 63-value vector is ready!
                # In your next module, you would save this to a file.
                print(landmark_vector)
            
            # Display a window (just for input positioning, no drawing on the image itself)
            cv2.putText(img, "EXTRACTING RAW DATA...", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
                
    except KeyboardInterrupt:
        print("\nExiting via KeyboardInterrupt (Ctrl+C).")

    finally:
        # Crucial cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and all windows closed.")


if __name__ == "__main__":
    main()