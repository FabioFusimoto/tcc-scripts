import cv2.cv2 as cv2
import numpy as np
from threading import Thread

class USBCamVideoStream:
    def __init__(self, camIndex=1):
        self.stream = cv2.VideoCapture(camIndex)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True