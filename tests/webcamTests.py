import cv2.cv2 as cv2
import time
from timeit import default_timer as timer

import src.webcamUtilities.video as video

def savePictureFromVideo(filename, display=False):
    cam = video.ThreadedWebCam().start()
    image = cam.read()

    if display:
        cv2.imshow('Frame', image)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
        cv2.destroyAllWindows()

    try:
        cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    except:
        print('An error occurred while trying to create the image')
    cam.stop()
    return

def takeMultiplePicturesFromVideo(filename, repetitions):
    for i in range(repetitions):
        input('Press to take a picture')
        savePictureFromVideo(filename + '-' + f'{i:0>2d}' + '.jpg', False)

def takePhoto(filename):
    cam = video.ThreadedWebCam().start()

    input('Press ENTER to take a picture')
    cv2.imwrite(filename, cam.getPhoto(), [cv2.IMWRITE_JPEG_QUALITY, 100])

    cam.stop()

def takePhotos(filePrefix, repetitions, offset):
    cam = video.ThreadedWebCam().start()

    for i in range(offset, offset + repetitions):
        input('Press ENTER to take a picture')
        image = cam.getPhoto()
        filename = filePrefix + '-' + f'{i:0>2d}' + '.jpg'

        try:
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        except:
            print('An error occurred while trying to create the image')
            cam.stop()

    cam.stop()

# testPhotoSave(filename='images/for-calibration/VR_02.jpg', rotationType=cv2.ROTATE_90_CLOCKWISE)
# multiplePhotoSave('images/for-calibration', 'X', 'jpg', 50)
# testVideoStream()
# takeMultiplePicturesFromVideo('images/test-1280x720', 5)
takePhoto('tests/precision/images/2322p-consistency-1-reference-and-pivot.jpg')
takePhotos('tests/precision/images/2322p-consistency-1-sample', 3, 0)