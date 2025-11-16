import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import queue
import time
import os
from datetime import datetime
import numpy as np

# -----------------------
# Configuration
# -----------------------
VIDEO_SOURCE = 0
FRAME_RATE = 20.0
LISTEN_PHRASE_TIME_LIMIT = 4 # Increased slightly for better voice command time
FLASH_DURATION = 0.5
DEFAULT_DRAW_COLOR = (0, 255, 0) # Green (BGR)

# Centralized voice commands for robustness
CUSTOM_ACTION_COMMANDS = {} # Will be populated by user input
COLOR_COMMANDS = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255) # Added red as an option
}
FILTER_COMMANDS = {
    "grayscale": "grayscale",
    "edge": "edge",
    "cartoon": "cartoon",
    "normal": None, # To remove filter
    "no filter": None
}
FILTER_LIST = ["normal", "grayscale", "edge", "cartoon"] # For keyboard cycling
CURRENT_FILTER_INDEX = 0

# -----------------------
# Globals
# -----------------------
command_queue = queue.Queue()
frame_lock = threading.Lock()
terminate_event = threading.Event()

all_strokes = []
current_drawing_points = []
current_draw_color = DEFAULT_DRAW_COLOR
current_filter = None

# -----------------------
# Utilities
# -----------------------
def get_desktop_path():
    """Returns the desktop path for cross-platform file saving."""
    try:
        # Windows
        return os.path.join(os.environ["USERPROFILE"], "Desktop")
    except KeyError:
        # Unix/Linux/macOS
        return os.path.join(os.path.expanduser("~"), "Desktop")

def ts(fmt="%Y%m%d_%H%M%S"):
    """Generates a timestamp string."""
    return datetime.now().strftime(fmt)

def unique_filename(prefix, ext):
    """Creates a unique timestamped filename on the user's desktop."""
    desktop = get_desktop_path()
    return os.path.join(desktop, f"{prefix}_{ts()}.{ext}")

# -----------------------
# Hand Tracker
# -----------------------
class HandTracker:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.6, trackCon=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        """Processes the image and finds hand landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        """Extracts the pixel coordinates of all 21 hand landmarks."""
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                # Convert normalized coordinates (0 to 1) to pixel coordinates
                lmList.append([id, int(lm.x * w), int(lm.y * h)])
        return lmList

    def fingersUp(self, lmList):
        """Determines which fingers are extended (Up)."""
        # Note: Thumb is slightly simplified here. More robust checks exist.
        fingers = []
        # Thumb (check x-coordinate for flip-friendly tracking)
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 Fingers (check y-coordinate)
        tipIds = [8, 12, 16, 20] # Index, Middle, Ring, Pinky tips
        for id in tipIds:
            # Check if tip is above the knuckle (lmList[id-2])
            fingers.append(1 if lmList[id][2] < lmList[id-2][2] else 0)
        return fingers

# -----------------------
# Voice Listener
# -----------------------
def voice_listener():
    """
    Listens for voice commands and puts recognized actions, filters, or colors
    into the command queue.
    """
    global CUSTOM_ACTION_COMMANDS
    r = sr.Recognizer()
    r.pause_threshold = 0.5 # A shorter pause helps responsiveness

    print("[VOICE]: Voice listener starting...")
    while not terminate_event.is_set():
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                print("[VOICE]: Listening...")
                audio = r.listen(source, phrase_time_limit=LISTEN_PHRASE_TIME_LIMIT)
            
            spoken_text = r.recognize_google(audio).lower()
            print(f"[VOICE DETECTED]: {spoken_text}")
            
            # 1. Check for custom action commands (e.g., "take a picture")
            action_found = False
            for key, cmd in CUSTOM_ACTION_COMMANDS.items():
                if key in spoken_text:
                    command_queue.put(("voice", cmd, spoken_text))
                    action_found = True
                    break
            
            # 2. Check for filter commands (e.g., "apply cartoon filter")
            for key, filter_name in FILTER_COMMANDS.items():
                if key in spoken_text and not action_found:
                    command_queue.put(("filter", filter_name, spoken_text))
                    action_found = True
                    break

            # 3. Check for color commands (e.g., "change color to blue")
            for key, color in COLOR_COMMANDS.items():
                if key in spoken_text and not action_found:
                    command_queue.put(("color", color, spoken_text))
                    break # Color command does not need to block other actions

        except sr.WaitTimeoutError:
            # No speech detected within the time limit
            continue
        except sr.UnknownValueError:
            # Speech was detected but could not be understood
            continue
        except sr.RequestError:
            # Could not request results from Google Speech Recognition service
            print("[VOICE ERROR]: API request failed. Check internet connection.")
        except Exception as e:
            # General exception (e.g., microphone issue)
            print(f"[VOICE ERROR]: {e}")
            time.sleep(1)

# -----------------------
# Filters
# -----------------------
def apply_filter(frame, filter_name):
    """Applies a specified visual filter to the frame."""
    if filter_name == "grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_name == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Canny returns 1-channel, convert back to 3-channel BGR for display
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "cartoon":
        # Simplified cartoon filter using bilateral filter and thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_3c)
    
    # 'normal' or None returns the original frame
    return frame

# -----------------------
# Main App
# -----------------------
def main():
    global current_draw_color, all_strokes, current_drawing_points
    global current_filter, CURRENT_FILTER_INDEX, CUSTOM_ACTION_COMMANDS

    # --- ASK USER FOR CUSTOM COMMANDS ---
    print("\n--- Multimodal Camera Setup ---")
    print("Please set your custom voice command phrases for core actions.")
    
    # Collect custom phrases and map them to internal commands
    setup_commands = {
        "Phrase to take a picture": "picture",
        "Phrase to start recording video": "start_video",
        "Phrase to stop recording video": "stop_video",
        "Phrase to quit program": "quit"
    }
    
    for prompt, cmd_key in setup_commands.items():
        phrase = input(f"{prompt} (e.g., 'cheese!'): ").strip().lower()
        if phrase:
            CUSTOM_ACTION_COMMANDS[phrase] = cmd_key
        else:
            print(f"[{cmd_key}] command not set.")

    # --- Setup Camera and Threads ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = HandTracker()
    threading.Thread(target=voice_listener, daemon=True).start()

    # --- State Variables ---
    recording = False
    video_writer = None
    is_drawing = False
    hud_status = "Initialized"
    detected_gesture = "None"
    last_voice_command = "..."
    flash_active = False
    flash_start_time = 0.0
    
    win_name = "Multimodal Camera (Q to Quit | F to cycle filter)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_with_drawing = frame.copy()

            # --- Hand Tracking ---
            img_hand = tracker.findHands(frame.copy(), draw=True)
            lmList = tracker.findPosition(img_hand)
            fingers_extended = 0
            
            if lmList:
                fingers = tracker.fingersUp(lmList)
                fingers_extended = sum(fingers)
                index_up = fingers[1] == 1
                middle_up = fingers[2] == 1
                index_x, index_y = lmList[8][1], lmList[8][2]


                # --- 1-Finger Gesture: Air Drawing ---
                if fingers_extended == 5:
                    command_queue.put(("gesture", "clear_drawing", "Palm Down Gesture"))
                    detected_gesture = "CANVAS CLEARED"
                elif index_up and not middle_up and fingers_extended == 1:
                    if not is_drawing:
                        is_drawing = True
                    current_drawing_points.append((index_x, index_y, current_draw_color))
                    cv2.circle(img_hand, (index_x, index_y), 8, current_draw_color, cv2.FILLED)
                    detected_gesture = f"DRAWING (Color: {current_draw_color})"
                
                # --- Drawing Session End / Idle ---
                else:
                    if is_drawing and len(current_drawing_points) > 3:
                        # Finalize the current stroke
                        with frame_lock:
                            all_strokes.append(current_drawing_points.copy())
                        current_drawing_points.clear()
                        detected_gesture = "STROKE ENDED"
                    
                    is_drawing = False
                    
                    if not detected_gesture.startswith("DRAWING"):
                        detected_gesture = f"{fingers_extended} fingers open" if fingers_extended > 0 else "Idle"


            # --- Render Drawing on the Frame ---
            with frame_lock:
                for stroke_points in all_strokes:
                    for i in range(1, len(stroke_points)):
                        p1 = (stroke_points[i-1][0], stroke_points[i-1][1])
                        p2 = (stroke_points[i][0], stroke_points[i][1])
                        color = stroke_points[i][2]
                        cv2.line(frame_with_drawing, p1, p2, color, 4)
                
                # Draw the current, unfinished stroke
                for i in range(1, len(current_drawing_points)):
                    p1 = (current_drawing_points[i-1][0], current_drawing_points[i-1][1])
                    p2 = (current_drawing_points[i][0], current_drawing_points[i][1])
                    color = current_drawing_points[i][2]
                    cv2.line(frame_with_drawing, p1, p2, color, 4)
                
                last_frame = frame_with_drawing.copy()

            # Blend the drawing canvas with the live hand tracking overlay
            display_frame = cv2.addWeighted(frame_with_drawing, 1.0, img_hand, 0.7, 0)

            # --- Process commands ---
            while not command_queue.empty():
                source, cmd, data = command_queue.get_nowait()
                
                if source == "voice":
                    last_voice_command = data
                
                if cmd == "picture":
                    fname = unique_filename("picture", "jpg")
                    cv2.imwrite(fname, last_frame)
                    hud_status = f"Picture Taken: {os.path.basename(fname)}"
                    flash_active = True
                    flash_start_time = time.time()
                elif cmd == "clear_drawing":
                    # Only clears the drawing; does NOT stop video
                    with frame_lock:
                        all_strokes.clear()
                        current_drawing_points.clear()
                    hud_status = "Drawing Cleared"
                
                elif cmd == "start_video" and not recording:
                    fname = unique_filename("video", "avi")
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    h, w = last_frame.shape[:2]
                    video_writer = cv2.VideoWriter(fname, fourcc, FRAME_RATE, (w, h))
                    if video_writer.isOpened():
                        recording = True
                        hud_status = f"Recording: {os.path.basename(fname)}"
                    else:
                        hud_status = "ERROR: VideoWriter failed to open"
                
                elif cmd == "stop_video" and recording:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    hud_status = "Stopped recording"
                
                elif cmd == "quit":
                    terminate_event.set()
                    hud_status = "Quitting..."
                    break
                
                # Filter commands from voice listener
                elif source == "filter":
                    current_filter = data
                    hud_status = f"Filter: {current_filter or 'None'}"
                
                # Color commands from voice listener
                elif source == "color":
                    current_draw_color = data
                    # Update filter index state to reflect 'normal' if a color was chosen
                    if current_filter:
                         current_filter = None
                         global CURRENT_FILTER_INDEX
                         CURRENT_FILTER_INDEX = 0

            # --- Apply filter ---
            if current_filter:
                tmp = apply_filter(display_frame, current_filter)
                # Handle conversion if filter outputs a single-channel (grayscale/edge) image
                if len(tmp.shape) == 2:
                    display_frame = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = tmp
            
            # --- Record video (record the filter-applied frame) ---
            if recording and video_writer:
                # Write the frame that includes the drawing but *before* the filter is applied
                # to keep the video consistent, or you can write the display_frame
                video_writer.write(last_frame)

            # --- Flash Effect ---
            if flash_active:
                if time.time() - flash_start_time < FLASH_DURATION:
                    display_frame[:] = 255 # White flash
                    h, w = display_frame.shape[:2]
                    cv2.putText(display_frame, "PICTURE TAKEN!", (w//2 - 150, h//2),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                else:
                    flash_active = False

            # --- HUD overlay ---
            if not flash_active:
                h, w = display_frame.shape[:2]
                
                # Dark transparent overlay for HUD
                cv2.rectangle(display_frame, (0,0), (w,60), (0,0,0), -1)
                cv2.rectangle(display_frame, (0,h-60), (w,h), (0,0,0), -1)

                cv2.putText(display_frame, f"STATUS: {hud_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                
                # Right side: Filter & Recording Status
                filter_name = current_filter.upper() if current_filter else "NORMAL"
                cv2.putText(display_frame, f"FILTER (F): {filter_name}", (w-250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                
                if recording:
                    cv2.putText(display_frame, "● REC", (w-90, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                
                # Bottom Left: Voice Command & Gesture
                cv2.putText(display_frame, f"GESTURE: {detected_gesture}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(display_frame, f"VOICE: {last_voice_command}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
                
                # Drawing Status
                color_name = next((k for k, v in COLOR_COMMANDS.items() if v == current_draw_color), "Custom")
                cv2.putText(display_frame, f"DRAW COLOR: {color_name.upper()}", (w-250, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_draw_color, 1)

            cv2.imshow(win_name, display_frame)
            
            # --- Keyboard Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or terminate_event.is_set():
                break
            
            # Filter Cycling (Keyboard)
            if key == ord('f'):
                CURRENT_FILTER_INDEX = (CURRENT_FILTER_INDEX + 1) % len(FILTER_LIST)
                current_filter = FILTER_LIST[CURRENT_FILTER_INDEX] if FILTER_LIST[CURRENT_FILTER_INDEX] != "normal" else None
                hud_status = f"Filter Switched: {current_filter or 'None'}"

    finally:
        # --- Cleanup ---
        if video_writer: video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        tracker.hands.close()
        terminate_event.set()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()