# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:58:36 2021

@author: roudr
"""


#pip install pafy
#pip install --upgrade youtube_dl

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:29:09 2021

@author: roudr
"""


import cv2, pafy # import opencv
#import matplotlib.pyplot as plt


url = input("Enter the youtube url in quotes : ")

classLabels = [] # empty list of python
file_name = 'coco_labels.txt'
with open(file_name, 'rt') as fpt:   #'rt :read'
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())
    
print(classLabels)
print("the number of classes are", len(classLabels))

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'   #' wights'

model = cv2.dnn_DetectionModel(frozen_model,config_file) # after dnn press tab this will give you options to comands
model.setInputSize(320,320) # as model is 320x320 in configuration file
model.setInputScale(1.0/127.5) ## 355/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) ## mobilenet takes input as [-1,1]
model.setInputSwapRB(True) # so automatic conversion from BGR (OpenCV default) to RGB



video = pafy.new(url)
best  = video.getbest(preftype="mp4")

cap = cv2.VideoCapture(best.url)


font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    
    ClassIndex,confidence,bbox = model.detect(frame,confThreshold = 0.4)
    
    #print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf,boxes in zip (ClassIndex.flatten(), confidence.flatten(),bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),1)
                cv2.putText(frame,classLabels[ClassInd-1].upper(),(boxes[0]+30,boxes[1]+4), font , fontScale = font_scale,color = (0,255,0),thickness=2)
                cv2.putText(frame,str(round(conf*100)),(boxes[0],boxes[1]+4), font , fontScale = font_scale,color = (0,0,255),thickness=1)
    
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows