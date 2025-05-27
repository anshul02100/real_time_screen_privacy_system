import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageGrab
import time
import threading
import winsound

class Eye:
    LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
    RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]

    def __init__(self, frame, landmarks, side):
        self.frame = frame
        self.landmarks = landmarks
        self.side = side
        self.height, self.width = frame.shape[:2]
        self.eye_points = self.LEFT_EYE_IDX if side == 0 else self.RIGHT_EYE_IDX
        self.eye_coords = self._get_eye_coords()
        self.pupil = self._get_pupil()

    def _get_eye_coords(self):
        coords = []
        try:
            for idx in self.eye_points:
                lm = self.landmarks.landmark[idx]
                x, y = int(lm.x * self.width), int(lm.y * self.height)
                coords.append((x, y))
        except Exception as e:
            print(f"[Eye._get_eye_coords ERROR]: {e}")
        return coords

    def _get_pupil(self):
        try:
            mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
            points = np.array(self.eye_coords, dtype=np.int32)
            if points.size == 0:
                return type('Point', (), {'x': 0, 'y': 0})
            cv2.fillPoly(mask, [points], 255)
            eye_region = cv2.bitwise_and(self.frame, self.frame, mask=mask)
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
            moments = cv2.moments(thresh)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = 0, 0
            return type('Point', (), {'x': cx, 'y': cy})
        except Exception as e:
            print(f"[Eye._get_pupil ERROR]: {e}")
            return type('Point', (), {'x': 0, 'y': 0})

class GazeTracking:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  # Allow up to 5 faces
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.eye_left = None
        self.eye_right = None
        self.frame = None
        self.landmarks = None
        self.num_faces = 0

    def refresh(self, frame):
        try:
            self.frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            self.num_faces = 0
            if results.multi_face_landmarks:
                self.num_faces = len(results.multi_face_landmarks)
                landmarks = results.multi_face_landmarks[0]  # Just use the first face for gaze
                self.eye_left = Eye(frame, landmarks, 0)
                self.eye_right = Eye(frame, landmarks, 1)
                self.landmarks = landmarks
            else:
                self.eye_left = None
                self.eye_right = None
                self.landmarks = None
        except Exception as e:
            print(f"[GazeTracking.refresh ERROR]: {e}")
            self.eye_left = None
            self.eye_right = None
            self.landmarks = None
            self.num_faces = 0

    def pupils_located(self):
        return self.eye_left is not None and self.eye_right is not None

    def horizontal_ratio_one_eye(self, eye):
        if eye.pupil.x == 0:
            return None
        xs = [p[0] for p in eye.eye_coords if p is not None]
        if not xs:
            return None
        min_x = min(xs)
        max_x = max(xs)
        eye_width = max_x - min_x
        if eye_width == 0:
            return None
        return (eye.pupil.x - min_x) / eye_width

    def get_average_horizontal_ratio(self):
        if not self.pupils_located():
            return None
        left = self.horizontal_ratio_one_eye(self.eye_left)
        right = self.horizontal_ratio_one_eye(self.eye_right)
        if left is None or right is None:
            return None
        return (left + right) / 2

    def get_head_yaw(self):
        if not self.pupils_located() or self.landmarks is None:
            return None
        try:
            nose_idx = 1
            left_eye_center = np.mean(self.eye_left.eye_coords, axis=0)
            right_eye_center = np.mean(self.eye_right.eye_coords, axis=0)
            nose = self.landmarks.landmark[nose_idx]
            nose_x = int(nose.x * self.eye_left.width)
            eye_center_x = int((left_eye_center[0] + right_eye_center[0]) / 2)
            dx = nose_x - eye_center_x
            return dx / self.eye_left.width
        except Exception as e:
            print(f"[GazeTracking.get_head_yaw ERROR]: {e}")
            return None

class FullScreenBlur:
    def __init__(self):
        self.overlay = None

    def show(self):
        if self.overlay:
            return
        screenshot = ImageGrab.grab(all_screens=True)
        small = screenshot.resize((screenshot.width // 4, screenshot.height // 4), Image.BILINEAR)
        blurred = small.filter(ImageFilter.GaussianBlur(radius=15))
        blurred = blurred.resize(screenshot.size, Image.BILINEAR)

        self.overlay = tk.Toplevel()
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg='black')

        self.overlay_img = ImageTk.PhotoImage(blurred)
        label = tk.Label(self.overlay, image=self.overlay_img)
        label.pack()

    def hide(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.gaze = GazeTracking()
        self.cap = cv2.VideoCapture(0)

        self.is_running = False
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_data = []

        self.calibrated_ratio = None
        self.blur_threshold = 0.15
        self.yaw_threshold = 0.07
        self.recent_ratios = []
        self.smoothing_frames = 7
        self.look_away_start = None
        self.blur_delay = 3

        self.overlay_shown = False
        self.alert_triggered = False

        self.blur_overlay = FullScreenBlur()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.start_btn = tk.Button(window, text="Start", width=10, command=self.start_gaze)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(window, text="Stop", width=10, command=self.stop_gaze)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.calib_btn = tk.Button(window, text="Calibrate", width=10, command=self.start_calibration)
        self.calib_btn.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(window, text="Status: Not running")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def start_gaze(self):
        self.is_running = True
        self.status_label.config(text="Status: Running")
        self.look_away_start = None
        self.recent_ratios = []

    def stop_gaze(self):
        self.is_running = False
        self.status_label.config(text="Status: Stopped")
        self.look_away_start = None
        self.recent_ratios = []
        self.blur_overlay.hide()
        self.alert_triggered = False

    def start_calibration(self):
        if not self.is_running:
            self.status_label.config(text="Start gaze blur first!")
            return
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_data = []
        self.recent_ratios = []
        self.status_label.config(text="Calibrating... Please look straight at screen")

    def show_blur_overlay(self):
        if self.overlay_shown:
            return
        self.blur_overlay.show()
        self.play_notification_sound()
        self.overlay_shown = True
        self.alert_triggered = True

    def remove_blur_overlay(self):
        if self.overlay_shown:
            self.blur_overlay.hide()
            self.overlay_shown = False
            self.alert_triggered = False

    def play_notification_sound(self):
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(self.delay, self.update)
            return

        frame = cv2.flip(frame, 1)

        if self.is_running:
            self.gaze.refresh(frame)
            if self.gaze.num_faces > 1:
                self._display_status(frame, f"Multiple Faces Detected! ({self.gaze.num_faces})", (0, 0, 255))
                self.play_notification_sound()
                self.show_blur_overlay()
            else:
                current_ratio = self.gaze.get_average_horizontal_ratio()
                smooth_ratio = self._get_smoothed_ratio(current_ratio)
                head_yaw = self.gaze.get_head_yaw()
                if head_yaw is not None and abs(head_yaw) > self.yaw_threshold:
                    self.look_away_start = None
                    self.remove_blur_overlay()
                    self._display_status(frame, "Head turned - ignoring gaze", (255, 255, 0))
                else:
                    if self.is_calibrating:
                        self._handle_calibration(smooth_ratio)
                    else:
                        if self.calibrated_ratio is None:
                            self._display_status(frame, 'Please calibrate gaze', (0, 255, 255))
                            self.look_away_start = None
                        else:
                            if smooth_ratio is not None:
                                self._handle_gaze_blur(smooth_ratio, frame)
                            else:
                                self.look_away_start = None
                                self.remove_blur_overlay()
                                self._display_status(frame, 'Face/Eyes not detected', (0, 255, 255))

                    if smooth_ratio is not None and self.calibrated_ratio is not None and abs(smooth_ratio - self.calibrated_ratio) <= self.blur_threshold:
                        self.remove_blur_overlay()
        else:
            self._display_status(frame, 'Gaze Blur OFF', (255, 255, 255))
            self.look_away_start = None
            self.remove_blur_overlay()

        self._update_canvas(frame)
        self.window.after(self.delay, self.update)

    def _get_smoothed_ratio(self, ratio):
        if ratio is not None:
            self.recent_ratios.append(ratio)
            if len(self.recent_ratios) > self.smoothing_frames:
                self.recent_ratios.pop(0)
            return sum(self.recent_ratios) / len(self.recent_ratios)
        return None

    def _handle_calibration(self, smooth_ratio):
        if smooth_ratio is not None:
            self.calibration_data.append(smooth_ratio)
            elapsed = time.time() - self.calibration_start_time
            self._display_status(None, f'Calibrating... {elapsed:.1f}s', (0, 255, 0))
            if elapsed >= 5:
                self.calibrated_ratio = sum(self.calibration_data) / len(self.calibration_data)
                self.is_calibrating = False
                self.status_label.config(text='Calibration complete')
        else:
            self._display_status(None, 'Face/Eyes not detected during calibration', (0, 255, 0))

    def _handle_gaze_blur(self, smooth_ratio, frame):
        diff = abs(smooth_ratio - self.calibrated_ratio)
        if diff > self.blur_threshold:
            if self.look_away_start is None:
                self.look_away_start = time.time()
            elif time.time() - self.look_away_start > self.blur_delay:
                self.show_blur_overlay()
                self._display_status(frame, 'Looking Away - Screen Blurred', (0, 0, 255))
        else:
            self.look_away_start = None
            self.remove_blur_overlay()
            self._display_status(frame, 'Looking Forward', (0, 255, 0))

    def _display_status(self, frame, text, color):
        if frame is not None:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _update_canvas(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def on_closing(self):
        self.cap.release()
        self.blur_overlay.hide()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "Gaze Blur with Face Alert")
