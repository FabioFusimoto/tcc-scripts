import cv2.cv2 as cv2
import numpy as np

def streamVideo():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if ret:        
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Frame', grayFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('No return from camera')
    
    cv2.destroyAllWindows()
    cam.release()