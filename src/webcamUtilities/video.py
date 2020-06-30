import cv2.cv2 as cv2

from .constants import BASE_URL, VIDEO_STREAM_ENDPOINT
from .commons import decodedImageFromURL

def streamVideo():
    while True:
        img = decodedImageFromURL(BASE_URL + VIDEO_STREAM_ENDPOINT)
        cv2.imshow('Webcam',img)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break