import cv2
import time
import imutils
import numpy as np 
import os
import math
#img = cv2.imread('lena.png')

thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)

net.setInputScale(1.0/ 127.5)

net.setInputMean((127.5, 127.5, 127.5))

net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(type(bbox))

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            #print(classId)
            
            if classId == 77: #Detection Cell Phone
                # print(str(bbox[:,0]) + " , " + str(bbox[0:,1]))
                # print(len(bbox[:,0])) # x
                # print(len(bbox[:,1])) # y
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                # print(box)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                if (len(bbox[:,0]) == 1): #X
                    if(len(bbox[:,1]) == 1): #Y
                        if(len(bbox[:,2]) == 1): #W
                            if(len(bbox[:,3]) == 1): #H
                                x1 = bbox[:,0]
                                y1 = bbox[:,1]
                                x2 = bbox[:,2]
                                y2 = bbox[:,3]
                                #print(box)
                                #print(x1 , y1, x2 ,y2)
                                
                                # MiddleX = int(math.floor((x1 + y1) / 2))
                                # MiddleY = int(math.floor((x1 + x2) / 2))

                                x1 = int(math.floor(x1))
                                y1 = int(math.floor(y1))
                                x2 = int(math.floor(x2))
                                y2 = int(math.floor(y2))
                                
                                #cv2.circle(img, (x1,x1) , 100 , (255, 0 , 0), -1)
                                #cv2.circle(img, (y1,y1) , 100 , (255, 0 , 0), -1)
                                #cv2.circle(img, (x2,x2) , 100 , (255, 0 , 0), -1)
                                #cv2.circle(img, (y2,y2) , 100 , (255, 0 , 0), -1)

                                #print(MiddleX, MiddleY)
                                #cv2.circle(img, (MiddleX, MiddleY), 200, (0, 255, 0), -1)

                                FirstVertexPosition = (x1, y1)
                                SecondVertexPosition = (x1 + x2, y1)
                                ThirdVertexPosition = (x2 + x1 , y2)
                                FourthVertexPosition = (x1, y2)
                                
                                #print(FirstVertexPosition)
                                # cv2.circle(img, (FirstVertexPosition[0], FirstVertexPosition[1]) , 100, (255, 0, 0), -1)
                                # cv2.circle(img, (SecondVertexPosition[0], SecondVertexPosition[1]), 100, (255, 0 , 0), -1)
                                # cv2.circle(img, (ThirdVertexPosition[0], ThirdVertexPosition[1]), 100, (255, 0 , 0), -1)
                                # cv2.circle(img, (FourthVertexPosition[0], FourthVertexPosition[1]), 100, (255, 0 , 0), -1)

                                MiddleX = int( math.floor( (FirstVertexPosition[0] + SecondVertexPosition[0]) / 2) )
                                MiddleY = int( math.floor ( ( SecondVertexPosition[1] + ThirdVertexPosition[1] ) / 2 ) )
                                
                                print(MiddleX, MiddleY)
                                cv2.circle(img, (MiddleX, MiddleY), 30 , (0, 255, 0), -1)
    time.sleep(0.5)
    cv2.imshow("Output",img)
    #cv2.imshow("Test", img)
    cv2.waitKey(1)