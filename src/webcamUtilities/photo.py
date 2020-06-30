import time

import cv2.cv2 as cv2

from .constants import BASE_URL, PHOTO_ENDPOINT
from .commons import imageArrayFromURL, decodedImageFromURL

def takePicture():
    img = decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)
    
    cv2.imshow('Picture', img)

    while True:
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def savePhoto(filename):
    img = decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)

    return cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 100])