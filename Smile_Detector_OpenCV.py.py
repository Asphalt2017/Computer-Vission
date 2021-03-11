# Computer-Vission
Smile Detector using OpenCV
# -*- coding: utf-8 -*-

@author: roudr
"""

import cv2

# Loading the cascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+h,x:x+w]
        
        roi_color = frame[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,20)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        smile = smile_cascade.detectMultiScale(roi_gray,1.8,30)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray,frame)
    cv2.imshow('video',canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
    
    
        