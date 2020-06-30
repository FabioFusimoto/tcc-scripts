import urllib.request as urllib
import cv2.cv2 as cv2
import numpy as np

def imageArrayFromURL(url):
    response = urllib.urlopen(url)
    return np.array(bytearray(response.read()),dtype=np.uint8)

def decodedImageFromURL(url):
    imageAsArray = imageArrayFromURL(url)
    return cv2.imdecode(imageAsArray,-1)