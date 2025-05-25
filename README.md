# Privacy Shield
A real-time privacy protection system using face detection and gaze tracking.
This project monitors the user's gaze and the number of faces visible to determine when to protect sensitive information. It blurs the screen when the user looks away and plays an alert sound if multiple faces are detected.

## Features :-
- Face detection using MediaPipe
- Gaze tracking using GazeTracking (built on dlib and OpenCV)
- Screen blur when user is not focused
- Audio alert when more than one face is detected
- Simple start/stop GUI using Tkinter

## Requirements :-
- Python 3.8–3.11
- opencv-python
- mediapipe
- pygame
- dlib
- GazeTracking (https://github.com/antoinelame/GazeTracking)
  
## Usage :-
1. Clone the repository and install the dependencies.
2. Place `alert.mp3` and the `gaze_tracking` folder in the project directory.
3. Run the script: python face_gaze.py
