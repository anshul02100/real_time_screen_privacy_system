# Privacy Shield

## Overview

A real-time privacy protection system using face detection and gaze tracking.  
This project monitors the user's gaze and the number of faces visible to determine when to protect sensitive information. It blurs the screen when the user looks away and plays an alert sound if multiple faces are detected.

---

## Features

- Face detection using MediaPipe  
- Gaze tracking using GazeTracking (built on dlib and OpenCV)  
- Screen blur when the user is not focused  
- Audio alert when more than one face is detected  
- Simple start/stop GUI using Tkinter  

---

## Requirements

- Python 3.8–3.11
- `opencv-python`
- `mediapipe`
- `pygame`
- `dlib`
- `GazeTracking` (https://github.com/antoinelame/GazeTracking)

---

## Installation

Ensure you are using **Python 3.8 – 3.11**.  
Install all dependencies using:

```bash
pip install -r requirements.txt

