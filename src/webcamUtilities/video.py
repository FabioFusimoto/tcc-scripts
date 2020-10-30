import cv2.cv2 as cv2
import numpy as np
from threading import Thread
import time
import urllib.request as urllib

class ThreadedWebCam:
    def __init__(self, baseUrl='http://192.168.0.9:8080', videoWidth=320, videoHeight=240, photoWidth=4128, photoHeight=2322):
        self.URL = baseUrl
        self.shotEndpoint = '/shot.jpg'
        self.photoEndpoint = '/photo.jpg'

        self.videoConfigEndpoint = '/settings/video_size?set={}x{}'.format(videoWidth, videoHeight)
        self.photoConfigEndpoint = '/settings/photo_size?set={}x{}'.format(photoWidth, photoHeight)

        # Initialize with given size
        urllib.urlopen(self.URL + self.videoConfigEndpoint)
        urllib.urlopen(self.URL + self.photoConfigEndpoint)

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
            time.sleep(1.0)

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

    def getPhoto(self):
        response = urllib.urlopen(self.URL + self.photoEndpoint)
        imgAsArray = np.array(bytearray(response.read()),dtype=np.uint8)
        return cv2.imdecode(imgAsArray, -1)
