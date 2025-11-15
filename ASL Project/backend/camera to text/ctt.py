import cv2
import speech_recognition as sr
import time
# We still import MediaPipe and HandTracker, but they will be ignored in the main loop
# if we are only focusing on Speech-to-Text for now.
import mediapipe as mp 
import numpy as np 

# --- SPEECH RECOGNITION FUNCTION ---
def record_and_transcribe():
    """Records audio from the microphone and transcribes it using Google's API."""
    recognizer = sr.Recognizer()

    # Use microphone as input
    with sr.Microphone() as source:
        print("\n[🎤] Listening...")
        # Calibrate the recognizer
        recognizer.adjust_for_ambient_noise(source, duration=0.5) 
        
        # We will listen for a fixed, short duration to avoid blocking too long
        # This makes the app responsive while transcribing.
        try:
            # Listen for up to 3 seconds
            audio = recognizer.listen(source, phrase_time_limit=3) 
        except sr.WaitTimeoutError:
            print("[❌] No speech detected within the timeout.")
            return None
        except Exception as e:
            # Catch other potential issues during listening
            print(f"[❌] An error occurred during listening: {e}")
            return None

    print("[⚙️] Processing audio...")

    try:
        # Recognition call
        text = recognizer.recognize_google(audio)
        print(f"📝 Transcribed: {text}")
        return text

    except sr.UnknownValueError:
        # This is common if silence or noise is detected
        # We return an empty string or a special message to display on the screen
        return "[...Listening for speech...]"
    except sr.RequestError as e:
        # API connection error
        print(f"❌ Could not request results from Google Speech API; {e}")
        return "[ERROR: Check network connection]"

# --- HAND TRACKER CLASS (Included but NOT used in the main loop for this task) ---
class HandTracker:
    # Retain the class definition for future use, but it's optional here.
    # ... (Your original HandTracker class code goes here, but we omit the full content for brevity)
    def __init__(self, *args, **kwargs):
        pass # Placeholder for this example
    def findHands(self, img, draw=True):
        return img
    def findPosition(self, img, handNo=0, draw=True):
        return []

# --- MAIN APPLICATION LOOP (Revised for continuous transcription) ---
def main():
    cap = cv2.VideoCapture(0)
    
    # Transcription settings
    TIME_TO_LISTEN = 4.0 # How often to call the microphone function (in seconds)
    last_listen_time = time.time()
    current_transcription = "Press 'q' to quit. Starting voice transcription soon..."
    
    # Initialize HandTracker (it won't be used, but helps if you add ASL later)
    # tracker = HandTracker() # Uncomment to use the tracker later

    print(f"App running. New transcription cycle every {TIME_TO_LISTEN} seconds.")

    while True:
        # 1. Video Capture
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
            
        # Optional: Flip image for mirror effect
        img = cv2.flip(img, 1)

        # 2. Check if it's time to transcribe
        if time.time() - last_listen_time > TIME_TO_LISTEN:
            # Block the video feed briefly to listen and transcribe
            new_text = record_and_transcribe()
            
            # Only update the displayed text if the function returned a result
            if new_text is not None:
                current_transcription = new_text
                
            last_listen_time = time.time() # Reset the timer

        # 3. Display Current Transcription on Screen
        # Display the instructional prompt
        cv2.putText(img, "SAY SOMETHING NOW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the transcribed text
        # Use a large font at the bottom of the screen
        cv2.putText(img, current_transcription, 
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 4. Display Image
        cv2.imshow("Speech to Screen", img)
        
        # 5. Handle Quit Key Press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting program by pressing 'q'.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()