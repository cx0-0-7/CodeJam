import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import queue
import time
import os
from datetime import datetime

# -----------------------
# Configuration
# -----------------------
VIDEO_SOURCE = 0
FRAME_RATE = 20.0
LISTEN_PHRASE_TIME_LIMIT = 4
FLASH_DURATION = 0.5
DEFAULT_DRAW_COLOR = (0, 255, 0)  # Green

# Custom voice commands
CUSTOM_ACTION_COMMANDS = {}  # Will be populated by user input

# -----------------------
# Globals
# -----------------------
command_queue = queue.Queue()
frame_lock = threading.Lock()
terminate_event = threading.Event()

all_strokes = []
current_drawing_points = []
current_draw_color = DEFAULT_DRAW_COLOR

# -----------------------
# Utilities
# -----------------------
def get_desktop_path():
    try:
        return os.path.join(os.environ["USERPROFILE"], "Desktop")
    except KeyError:
        return os.path.join(os.path.expanduser("~"), "Desktop")

def ts(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)

def unique_filename(prefix, ext):
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
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h)])
        return lmList

    def fingersUp(self, lmList):
        fingers = []
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        tipIds = [8, 12, 16, 20]
        for id in tipIds:
            fingers.append(1 if lmList[id][2] < lmList[id-2][2] else 0)
        return fingers

# -----------------------
# Voice Listener
# -----------------------
def voice_listener():
    global CUSTOM_ACTION_COMMANDS
    r = sr.Recognizer()
    r.pause_threshold = 0.5

    print("[VOICE]: Voice listener starting...")
    while not terminate_event.is_set():
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, phrase_time_limit=LISTEN_PHRASE_TIME_LIMIT)
            
            spoken_text = r.recognize_google(audio).lower()
            print(f"[VOICE DETECTED]: {spoken_text}")
            
            for key, cmd in CUSTOM_ACTION_COMMANDS.items():
                if key in spoken_text:
                    command_queue.put(("voice", cmd, spoken_text))
                    break

        except (sr.WaitTimeoutError, sr.UnknownValueError):
            continue
        except sr.RequestError:
            print("[VOICE ERROR]: API request failed.")
        except Exception as e:
            print(f"[VOICE ERROR]: {e}")
            time.sleep(1)

# -----------------------
# Main App
# -----------------------
def main():
    global current_draw_color, all_strokes, current_drawing_points

    print("\n--- Multimodal Camera Setup ---")
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

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = HandTracker()
    threading.Thread(target=voice_listener, daemon=True).start()

    recording = False
    video_writer = None
    is_drawing = False
    hud_status = "Initialized"
    detected_gesture = "None"
    last_voice_command = "..."
    flash_active = False
    flash_start_time = 0.0

    cv2.namedWindow("Multimodal Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_with_drawing = frame.copy()

            img_hand = tracker.findHands(frame.copy(), draw=True)
            lmList = tracker.findPosition(img_hand)
            fingers_extended = 0

            if lmList:
                fingers = tracker.fingersUp(lmList)
                fingers_extended = sum(fingers)
                index_up = fingers[1] == 1
                middle_up = fingers[2] == 1
                index_x, index_y = lmList[8][1], lmList[8][2]

                # --- Gesture-based color selection ---
                if fingers_extended == 2:
                    current_draw_color = (255, 0, 0)  # Blue
                elif fingers_extended == 4:
                    current_draw_color = (0, 255, 255)  # Yellow

                # --- Clear drawing (5 fingers) ---
                if fingers_extended == 5:
                    command_queue.put(("gesture", "clear_drawing", "Palm Down Gesture"))
                    detected_gesture = "CANVAS CLEARED"

                # --- 1-Finger Drawing ---
                elif index_up and not middle_up and fingers_extended == 1:
                    if not is_drawing:
                        is_drawing = True
                    current_drawing_points.append((index_x, index_y, current_draw_color))
                    cv2.circle(img_hand, (index_x, index_y), 8, current_draw_color, cv2.FILLED)
                    detected_gesture = f"DRAWING (Color: {current_draw_color})"
                
                # --- End stroke ---
                else:
                    if is_drawing and len(current_drawing_points) > 3:
                        with frame_lock:
                            all_strokes.append(current_drawing_points.copy())
                        current_drawing_points.clear()
                        detected_gesture = "STROKE ENDED"
                    is_drawing = False
                    if not detected_gesture.startswith("DRAWING"):
                        detected_gesture = f"{fingers_extended} fingers open" if fingers_extended > 0 else "Idle"

            # --- Render Drawing ---
            with frame_lock:
                for stroke_points in all_strokes:
                    for i in range(1, len(stroke_points)):
                        p1 = (stroke_points[i-1][0], stroke_points[i-1][1])
                        p2 = (stroke_points[i][0], stroke_points[i][1])
                        color = stroke_points[i][2]
                        cv2.line(frame_with_drawing, p1, p2, color, 4)
                for i in range(1, len(current_drawing_points)):
                    p1 = (current_drawing_points[i-1][0], current_drawing_points[i-1][1])
                    p2 = (current_drawing_points[i][0], current_drawing_points[i][1])
                    color = current_drawing_points[i][2]
                    cv2.line(frame_with_drawing, p1, p2, color, 4)

            last_frame = frame_with_drawing.copy()
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
                        hud_status = "ERROR: VideoWriter failed"
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

            # --- Record video ---
            if recording and video_writer:
                video_writer.write(last_frame)

            # --- Flash effect ---
            if flash_active:
                if time.time() - flash_start_time < FLASH_DURATION:
                    display_frame[:] = 255
                    h, w = display_frame.shape[:2]
                    cv2.putText(display_frame, "PICTURE TAKEN!", (w//2 - 150, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
                else:
                    flash_active = False

            # --- HUD overlay ---
            if not flash_active:
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0,0), (w,60), (0,0,0), -1)
                cv2.rectangle(display_frame, (0,h-60), (w,h), (0,0,0), -1)
                cv2.putText(display_frame, f"STATUS: {hud_status}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                if recording:
                    cv2.putText(display_frame, "● REC", (w-90, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(display_frame, f"GESTURE: {detected_gesture}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(display_frame, f"VOICE: {last_voice_command}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
                color_name = "GREEN" if current_draw_color==(0,255,0) else "BLUE" if current_draw_color==(255,0,0) else "YELLOW"
                cv2.putText(display_frame, f"DRAW COLOR: {color_name}", (w-250, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_draw_color, 1)

            cv2.imshow("Multimodal Camera", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or terminate_event.is_set():
                break

    finally:
        if video_writer: video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        tracker.hands.close()
        terminate_event.set()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
