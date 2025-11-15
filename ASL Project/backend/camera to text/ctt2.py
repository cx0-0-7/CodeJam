import cv2
import speech_recognition as sr
import threading
import time
from datetime import datetime
import os

# -------------------------------
# DESKTOP PATH (CROSS-PLATFORM)
# -------------------------------
def get_desktop_path():
    try:
        return os.path.join(os.environ["USERPROFILE"], "Desktop")  # Windows
    except KeyError:
        return os.path.join(os.path.expanduser("~"), "Desktop")    # macOS/Linux


# -------------------------------
# SHARED COMMAND VARIABLE
# -------------------------------
last_command = ""


# -------------------------------
# BACKGROUND VOICE LISTENER
# -------------------------------
def background_listener():
    global last_command
    r = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, phrase_time_limit=3)

            text = r.recognize_google(audio).lower()
            print("You said:", text)
            last_command = text

        except:
            pass  # ignore errors but keep listening


# -------------------------------
# MAIN CAMERA APP
# -------------------------------
def run_camera_app():
    global last_command

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error.")
        return

    print("Camera started.")

    recording = False
    video_writer = None

    # Start background listener thread
    listener_thread = threading.Thread(target=background_listener, daemon=True)
    listener_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)

        # ------------------------------------
        # HANDLE COMMANDS FROM LISTENER THREAD
        # ------------------------------------
        command = last_command

        # ---- TAKE PICTURE ----
        if "picture" in command:
            desktop = get_desktop_path()
            filename = os.path.join(
                desktop,
                datetime.now().strftime("picture_%Y%m%d_%H%M%S.jpg")
            )
            cv2.imwrite(filename, frame)
            print("📸 Picture saved to:", filename)
            last_command = ""

        # ---- START VIDEO ----
        if "start video" in command and not recording:
            print("🎥 Starting video recording...")
            desktop = get_desktop_path()
            video_filename = os.path.join(desktop, "recorded_video.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                video_filename, fourcc, 20.0,
                (frame.shape[1], frame.shape[0])
            )

            recording = True
            print("🎬 Saving video to:", video_filename)
            last_command = ""

        # ---- STOP VIDEO ----
        if "stop video" in command and recording:
            print("🛑 Stopping video recording.")
            recording = False
            if video_writer:
                video_writer.release()
            video_writer = None
            last_command = ""

        # ------------------------------------
        # WRITE VIDEO FRAMES IF RECORDING
        # ------------------------------------
        if recording and video_writer is not None:
            video_writer.write(frame)

        # ---- Exit program ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


# -------------------------------
run_camera_app()
