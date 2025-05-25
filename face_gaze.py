import cv2
import mediapipe as mp
import pygame
import threading
import tkinter as tk
from gaze_tracking import GazeTracking

pygame.mixer.init()
sound = pygame.mixer.Sound("alert.mp3")

face = mp.solutions.face_detection
detector = face.FaceDetection(min_detection_confidence=0.6)

gaze = GazeTracking()

running = False
cam = None
alert = None
isPlaying = False

def start():
    global running, cam, alert, isPlaying
    running = True
    cam = cv2.VideoCapture(0)

    while running:
        success, frame = cam.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)
        count = len(results.detections) if results.detections else 0

        if count > 1 and not isPlaying:
            alert = sound.play(-1)
            isPlaying = True
        elif count <= 1 and isPlaying and alert:
            alert.stop()
            isPlaying = False

        gaze.refresh(frame)

        if gaze.is_center():
            gaze_text = "Looking center"
            display_frame = frame.copy()
        else:
            gaze_text = "Not looking - screen blurred"
            display_frame = cv2.GaussianBlur(frame, (51, 51), 0)

        cv2.putText(display_frame, gaze_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Privacy Shield", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop()

def stop():
    global running, cam, alert, isPlaying
    running = False
    if alert:
        alert.stop()
        isPlaying = False
    if cam:
        cam.release()
    cv2.destroyAllWindows()

app = tk.Tk()
app.title("Face + Gaze Tracker")
app.geometry("300x150")

tk.Button(app, text="Start", font=("Arial", 14), bg="green", fg="white",
          command=lambda: threading.Thread(target=start).start()).pack(pady=20)

tk.Button(app, text="Stop", font=("Arial", 14), bg="red", fg="white", command=stop).pack()

app.mainloop()
