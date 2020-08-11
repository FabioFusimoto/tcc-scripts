import cv2.cv2 as cv2
import numpy as np
from timeit import default_timer as timer

from src.USBCam.video import USBCamVideoStream as USBCam

def savePictureFromVideo(filename, display=False):
    cam = USBCam(camIndex=2).start()
    image = cam.read()

    try:
        if display:
            cv2.imshow('Frame', image)

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
            cv2.destroyAllWindows()
        cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    except:
        print('An error occurred while trying to create the image')
    
    cam.stop()
    return

def takeMultiplePicturesFromVideo(filename, repetitions):
    for i in range(50, 50 + repetitions):
        input('Press to take a picture')
        savePictureFromVideo(filename + '-' + f'{i:0>2d}' + '.jpg', False)

savePictureFromVideo('images/camera_position_estimation_01.jpg', True)