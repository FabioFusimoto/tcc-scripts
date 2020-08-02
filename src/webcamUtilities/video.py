import cv2.cv2 as cv2
import numpy as np
from threading import Thread
import urllib.request as urllib

class ThreadedWebCam:
    def __init__(self, baseUrl='http://192.168.42.129:5050', width=1280, height=720):
        self.URL = baseUrl
        self.shotEndpoint = '/shot.jpg'
        self.videoConfigEndpoint = '/settings/video_size?set={0}x{1}'.format(width, height)

        # Initialize with given size
        urllib.urlopen(self.URL + self.videoConfigEndpoint)

        # Read frame
        response = urllib.urlopen(self.URL + self.shotEndpoint)
        imgAsArray = np.array(bytearray(response.read()),dtype=np.uint8)
        self.frame = cv2.imdecode(imgAsArray, -1)
        
        self.active = True
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.active:
                response = urllib.urlopen(self.URL + self.shotEndpoint)
                imgAsArray = np.array(bytearray(response.read()),dtype=np.uint8)
                self.frame = cv2.imdecode(imgAsArray, -1)
            else:
                return

    def read(self):
        return self.frame

    def stop(self):
        self.active = False

    def stream(self):
        while True:
            image = self.read()
            cv2.imshow('Video stream', image)
            if cv2.waitKey(.5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()