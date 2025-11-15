# """
# Voice-Controlled Webcam — Full Rewrite

# Features:
# - Say the trigger word (default: "picture") to save the current webcam frame to Desktop.
# - Robust microphone detection with fallback scanning.
# - Non-blocking speech recognition in a worker thread.
# - Thread-safe sharing of the last captured frame (uses a lock).
# - Clear on-screen status and transcription display.
# - Proper thread-lifecycle handling to avoid overlapping recognition threads.
# - Desktop-compatible save path (Windows/macOS/Linux).
# """

# import cv2
# import speech_recognition as sr
# import time
# import threading
# import queue
# import numpy as np
# import pyaudio
# import os
# import sys
# import traceback

# # ---------------------------
# # CONFIGURATION (edit here)
# # ---------------------------
# TIME_TO_LISTEN = 4.0          # seconds between listening attempts
# PHRASE_TIME_LIMIT = 3         # seconds max to listen per attempt
# SCREENSHOT_COMMAND = "picture"
# VIDEO_SOURCE = 0              # cv2.VideoCapture index
# SHOW_WINDOW_NAME = "Speech Command Shooter"
# FONT = cv2.FONT_HERSHEY_SIMPLEX

# # ---------------------------
# # SHARED / THREAD-SAFE OBJECTS
# # ---------------------------
# transcription_queue = queue.Queue()  # status updates (strings)
# command_queue = queue.Queue()        # commands (e.g., "TAKE_PICTURE")
# frame_lock = threading.Lock()        # protects last_frame_to_save
# last_frame_to_save = None

# # This event indicates whether a speech-worker is currently running.
# # It prevents starting another recognition while one is active.
# speech_worker_running = threading.Event()

# # ---------------------------
# # UTILITIES
# # ---------------------------
# def find_default_mic_index():
#     """
#     Attempt to return a reasonable microphone index for pyaudio.
#     Priority:
#       1) p.get_default_input_device_info()
#       2) first device with maxInputChannels > 0
#     Returns None if nothing usable is found.
#     """
#     try:
#         p = pyaudio.PyAudio()
#         try:
#             default_info = p.get_default_input_device_info()
#             default_index = int(default_info.get("index"))
#             p.terminate()
#             return default_index
#         except Exception:
#             # fallback: scan all devices
#             for i in range(p.get_device_count()):
#                 try:
#                     info = p.get_device_info_by_index(i)
#                     if int(info.get("maxInputChannels", 0)) > 0:
#                         p.terminate()
#                         return i
#                 except Exception:
#                     continue
#             p.terminate()
#     except Exception:
#         pass
#     return None

# def desktop_path():
#     """Return an existing Desktop path for the current user (cross-platform)."""
#     # Standard approach: try USERPROFILE (Windows) then HOME
#     try:
#         if sys.platform.startswith("win"):
#             userprofile = os.environ.get("USERPROFILE")
#             if userprofile:
#                 path = os.path.join(userprofile, "Desktop")
#                 return path
#         # macOS / Linux or fallback
#         home = os.path.expanduser("~")
#         path = os.path.join(home, "Desktop")
#         return path
#     except Exception:
#         return os.path.expanduser("~")

# def take_screenshot(img):
#     """
#     Save the provided BGR image to Desktop with a timestamped filename.
#     Returns a short status string for UI.
#     """
#     if img is None:
#         return "[❌ No frame to save]"

#     desk = desktop_path()
#     try:
#         os.makedirs(desk, exist_ok=True)
#     except Exception:
#         # If we can't make the desktop dir, fallback to home
#         desk = os.path.expanduser("~")

#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     filename = f"webcam_shot_{timestamp}.jpg"
#     filepath = os.path.join(desk, filename)

#     try:
#         success = cv2.imwrite(filepath, img)
#         if success:
#             return f"[✅ Saved: {filename}]"
#         else:
#             return "[❌ cv2.imwrite failed]"
#     except Exception as e:
#         return f"[❌ Save error: {e}]"

# # ---------------------------
# # SPEECH WORKER (thread)
# # ---------------------------
# def record_and_transcribe_threaded(mic_index):
#     """
#     Worker function to capture audio and attempt to transcribe it.
#     Puts results into transcription_queue and commands into command_queue.
#     Uses speech_worker_running Event to signal running/finished.
#     """
#     recognizer = sr.Recognizer()
#     try:
#         # Mark as running
#         speech_worker_running.set()
#         transcription_queue.put("[🎤 Listening...]")

#         with sr.Microphone(device_index=mic_index) as source:
#             # Quick ambient calibration (short duration)
#             try:
#                 recognizer.adjust_for_ambient_noise(source, duration=0.4)
#             except Exception:
#                 # If calibration fails, continue — recognizer will still try
#                 pass

#             audio = recognizer.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
#     except Exception as e:
#         transcription_queue.put(f"[❌ Mic/listen error: {e}]")
#         speech_worker_running.clear()
#         return

#     transcription_queue.put("[⚙️ Processing audio...]")

#     try:
#         text = recognizer.recognize_google(audio).lower()
#         transcription_queue.put(f"📝 {text}")

#         # Command check
#         if SCREENSHOT_COMMAND in text:
#             command_queue.put("TAKE_PICTURE")
#             transcription_queue.put("[🖼️ Command received: saving picture]")

#     except sr.UnknownValueError:
#         transcription_queue.put("[...No audible speech detected...]")
#     except sr.RequestError as e:
#         transcription_queue.put(f"[API ERROR: {e}]")
#     except Exception as e:
#         # Capture unexpected errors
#         transcription_queue.put(f"[CRITICAL THREAD ERROR: {e}]")
#         traceback.print_exc()
#     finally:
#         # Ensure the running flag is cleared so main loop may spawn new workers
#         speech_worker_running.clear()

# # ---------------------------
# # MAIN LOOP
# # ---------------------------
# def main():
#     global last_frame_to_save

#     mic_index = find_default_mic_index()
#     if mic_index is None:
#         print("Fatal: No microphone detected. Exiting.")
#         return
#     else:
#         print(f"Using microphone index: {mic_index}")

#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     if not cap.isOpened():
#         print(f"Fatal: Cannot open webcam (index {VIDEO_SOURCE}). Exiting.")
#         return

#     last_listen_time = 0.0
#     status_text = f"Say '{SCREENSHOT_COMMAND}'. Press 'q' to quit."
#     is_listening = False
#     spinner = ["|", "/", "-", "\\"]
#     spin_idx = 0

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 transcription_queue.put("[❌ Camera read failed]")
#                 break

#             # Mirror for natural webcam feel
#             frame = cv2.flip(frame, 1)

#             # Store a copy of the current frame (thread-safe)
#             with frame_lock:
#                 last_frame_to_save = frame.copy()
#                 last_frame_to_save_local = last_frame_to_save  # local reference for safety

#             # Start speech worker if interval passed and no worker currently running
#             now = time.time()
#             if not speech_worker_running.is_set() and (now - last_listen_time) >= TIME_TO_LISTEN:
#                 last_listen_time = now
#                 is_listening = True
#                 transcription_queue.put(f"[🎤 Say '{SCREENSHOT_COMMAND}' now...]")
#                 t = threading.Thread(target=record_and_transcribe_threaded, args=(mic_index,), daemon=True)
#                 t.start()

#             # Process transcription status updates (non-blocking)
#             if not transcription_queue.empty():
#                 status_text = transcription_queue.get_nowait()
#                 # If the worker finished and reported something other than the initial listening message,
#                 # consider ourselves no longer "listening" for UI purposes.
#                 # We'll rely on the speech_worker_running Event for accurate state.
#                 if not speech_worker_running.is_set():
#                     is_listening = False

#             # Process commands
#             if not command_queue.empty():
#                 cmd = command_queue.get_nowait()
#                 if cmd == "TAKE_PICTURE":
#                     # Save using the frame we captured under lock
#                     with frame_lock:
#                         img_to_save = last_frame_to_save_local if 'last_frame_to_save_local' in locals() else None
#                     save_status = take_screenshot(img_to_save)
#                     status_text = save_status
#                     # Make sure listening resets
#                     is_listening = False
#                     # Clear any running worker flag just in case
#                     speech_worker_running.clear()

#             # Draw UI elements
#             status_color = (0, 200, 0) if is_listening or speech_worker_running.is_set() else (200, 200, 200)
#             cv2.putText(frame, f"STATUS: {'LISTENING' if (is_listening or speech_worker_running.is_set()) else 'IDLE'} {spinner[spin_idx]}",
#                         (10, 30), FONT, 0.9, status_color, 2, cv2.LINE_AA)
#             spin_idx = (spin_idx + 1) % len(spinner)

#             # Wrap status_text to fit screen bottom if necessary
#             txt = status_text
#             # We'll simply draw it once; user can adjust font/size if needed
#             cv2.putText(frame, txt, (10, frame.shape[0] - 20), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#             cv2.imshow(SHOW_WINDOW_NAME, frame)

#             # Quit key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
