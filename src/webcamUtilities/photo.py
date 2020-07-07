import time

import cv2.cv2 as cv2

from src.utilities import rotateImages
from .constants import BASE_URL, PHOTO_ENDPOINT
from .commons import imageArrayFromURL, decodedImageFromURL

def takePicture():
    img = decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)
    
    cv2.imshow('Picture', img)

    while True:
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def savePhoto(filename, rotationType=None):
    img = decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)

    if(rotationType is not None):
        img = rotateImages.rotateImage(img, rotationType, filename, False)
    return cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def saveMultiple(path, prefix, extension, count):
    for i in range(count):
        input('Press ENTER key to take a picture\n')
        img = decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)
        filename = path + '/' + prefix + format(i, '02d') + '.' + extension
        cv2.imwrite(filename, img)
        print(filename + ' saved')