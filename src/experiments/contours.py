import numpy as np
from cv2 import cv2

import src.webcamUtilities.commons as webcam
from src.webcamUtilities.constants import BASE_URL, PHOTO_ENDPOINT

def displayContoursOnWebcamPhoto():
    photo = webcam.decodedImageFromURL(BASE_URL + PHOTO_ENDPOINT)
    grayPhoto = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(grayPhoto, 63, 255, 0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print(str(len(contours)) + ' contours found')

    photoWithCountours = cv2.drawContours(grayPhoto, contours, -1, (0,255,0), 3)

    cv2.imwrite('contours.png', photoWithCountours)
    cv2.imshow('Picture', photoWithCountours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()