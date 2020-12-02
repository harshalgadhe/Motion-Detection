import cv2 as cv
from datetime import datetime
import pandas
import time

first_frame=None
video=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    check,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray=cv.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv.absdiff(first_frame,gray)
    thresh_delta=cv.threshold(delta_frame,30,255,cv.THRESH_BINARY)[1]
    thresh_delta=cv.dilate(thresh_delta,None,iterations=0)
    cnts ,_ =cv.findContours(thresh_delta.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv.contourArea(contour)<10000:
            continue
        (x,y,w,h)=cv.boundingRect(contour)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv.imshow('frame',frame)

    key=cv.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv.destroyAllWindows()
