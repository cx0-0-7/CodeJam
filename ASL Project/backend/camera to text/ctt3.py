import cv2
import mediapipe as mp
import time
import threading
import speech_recognition as sr


# -------------------------------
# GLOBAL VARIABLES (shared between threads)
# -------------------------------
current_transcription = ""
message_text = ""
message_timer = 0
all_strokes = []
index_points = []
write_mode = False
img_for_photo = None
lock = threading.Lock()  # ensures thread-safe access


# -------------------------------
# HAND TRACKER
# -------------------------------
class handTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList


    def fingersUp(self, lmList):
        fingers = []
        # Thumb
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers: index, middle, ring, pinky
        tipIds = [8, 12, 16, 20]
        for id in tipIds:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


# -------------------------------
# SPEECH RECOGNITION FUNCTION
# -------------------------------
def record_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print("Listening error:", e)
            return None
    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        return "[...]"
    except sr.RequestError as e:
        print("Google API error:", e)
        return "[ERROR]"


# -------------------------------
# THREAD FUNCTION FOR CONTINUOUS LISTENING
# -------------------------------
def continuous_listen():
    global current_transcription, message_text, message_timer, all_strokes, index_points, img_for_photo
    while True:
        text = record_and_transcribe()
        if text:
            with lock:
                current_transcription = text
                # Voice commands
                if "photo" in text and img_for_photo is not None:
                    cv2.imwrite("snapshot.png", img_for_photo)
                    message_text = "PHOTO TAKEN"
                    message_timer = 50
                elif "clear" in text or "reset" in text:
                    all_strokes.clear()
                    index_points.clear()
                    message_text = "CANVAS CLEARED"
                    message_timer = 50
                elif "save" in text:
                    if img_for_photo is not None:
                        canvas = img_for_photo.copy()
                        for stroke in all_strokes:
                            for i in range(1, len(stroke)):
                                cv2.line(canvas, stroke[i-1], stroke[i], (0, 255, 0), 4)
                        cv2.imwrite("drawing.png", canvas)
                        message_text = "DRAWING SAVED"
                        message_timer = 50
                elif "quit" in text or "exit" in text:
                    print("Exiting via voice command...")
                    exit()


# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    global write_mode, index_points, all_strokes, current_transcription, message_text, message_timer, img_for_photo


    pTime = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()


    # Start voice listener thread
    listener_thread = threading.Thread(target=continuous_listen, daemon=True)
    listener_thread.start()


    while True:
        success, img = cap.read()
        if not success:
            break


        img = cv2.flip(img, 1)
        with lock:
            img_for_photo = img.copy()  # for photo saving


        # -----------------------
        # HAND DETECTION AND WRITING
        # -----------------------
        img = tracker.findHands(img)
        lmList = tracker.findPosition(img)
        if len(lmList) != 0:
            fingers = tracker.fingersUp(lmList)
            index_up = fingers[1] == 1
            middle_up = fingers[2] == 1
            fingers_extended = sum(fingers)
            # -----------------------
            # COLOR SELECTION BASED ON FINGERS
            # -----------------------
            if fingers_extended == 1:
                current_color = (0, 255, 0)   # Green
            elif fingers_extended == 3:
                current_color = (255, 0, 0)   # Blue
            elif fingers_extended == 4:
                current_color = (0, 255, 255) # Yellow
            else:
                current_color = (0, 255, 0)   # Default green


            ix, iy = lmList[8][1], lmList[8][2]


            # Reset canvas if full hand
            if fingers_extended == 5:
                all_strokes.clear()
                index_points.clear()
                write_mode = False
                with lock:
                    message_text = "RESET"
                    message_timer = 50


            # Write mode
            elif index_up and not middle_up:
                write_mode = True
                index_points.append((ix, iy))
                cv2.circle(img, (ix, iy), 8, current_color, cv2.FILLED)
                cv2.putText(img, "WRITING", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


            # Stop writing
            elif index_up and middle_up:
                if write_mode and len(index_points) > 3:
                    all_strokes.append(index_points.copy())
                write_mode = False
                index_points.clear()
                cv2.putText(img, "STOP", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        # Draw strokes
        for stroke in all_strokes:
            for i in range(1, len(stroke)):
                cv2.line(img, stroke[i-1], stroke[i], current_color, 3)
        for i in range(1, len(index_points)):
            cv2.line(img, index_points[i-1], index_points[i], current_color, 3)


        # -----------------------
        # DISPLAY TEXT
        # -----------------------
        with lock:
            transcription_to_display = current_transcription
            message_to_show = message_text
            message_timer_local = message_timer
            if message_timer_local > 0:
                message_timer -= 1


        cv2.putText(img, "SAY SOMETHING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(img, transcription_to_display, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


        if message_timer_local > 0:
            cv2.putText(img, message_to_show, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
