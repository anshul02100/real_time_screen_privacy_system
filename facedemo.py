import cv2
import mediapipe as mp
import pygame
import threading
import tkinter as tk

pygame.mixer.init()                       
sound=pygame.mixer.Sound("alert.mp3")

face=mp.solutions.face_detection
detector=face.FaceDetection(min_detection_confidence=0.6)

running=False
cam=None
alert=None
isPlaying=False    

def start():
    global running,cam,alert,isPlaying
    running = True
    cam=cv2.VideoCapture(0)
    while running:
        success,frame=cam.read()                                #captures frame
        if not success:                                         #breaks the loop if frame capturing fails
            break
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)         #converts BGR to RGB
        results=detector.process(frame_rgb)                     #detect faces
        if results.detections:
            count=len(results.detections)                       #counts number of faces
        else: count=0
        cv2.putText(frame,f'Faces Detected:{count}',(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    #show number of faces
        cv2.imshow("FaceDemo",frame)
        if count>1:                                             #if number of faces >0, plays alert notification
            if not isPlaying:
                alert=sound.play(-1)
                isPlaying=True
        else:
            if isPlaying and alert:
                alert.stop()
                isPlaying=False
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    stop()

def stop():
    global running,cam,alert,isPlaying
    running=False
    if alert:
        alert.stop()
        isPlaying=False
    if cam:
        cam.release()
    cv2.destroyAllWindows()

app=tk.Tk()
app.title("FaceDemo")
app.geometry("300x150")
start_button = tk.Button(app,text="Start",font=("Arial",14),bg="green",fg="white",command=lambda:threading.Thread(target=start).start())
start_button.pack(pady=20)
stop_button = tk.Button(app,text="Stop",font=("Arial", 14),bg="red",fg="white",command=stop)
stop_button.pack()
app.mainloop()