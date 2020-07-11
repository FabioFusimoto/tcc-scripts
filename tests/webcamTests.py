import cv2.cv2 as cv2
import time

import src.webcamUtilities.video as video
import src.webcamUtilities.photo as photo

def testVideoStream():
    print('Should display a video stream (press Q to exit)')
    video.streamVideo()

def testPhotoCapture():
    print('It should take a picture and display it on screen (press Q to exit)')
    photo.takePicture()

def testPhotoSave(filename='test.jpg', rotationType=None):
    print('It should take a picture and save it')
    time.sleep(3)
    print('Success? -> ' + str(photo.savePhoto(filename, rotationType=rotationType)))

def multiplePhotoSave(path, prefix, extension, count):
    print('It should take multiples pictures and save')
    input('Press ENTER key to start\n')
    photo.saveMultiple(path, prefix, extension, count)

testPhotoSave(filename='images/for-calibration/ARU_MISC_01.jpg', rotationType=cv2.ROTATE_90_CLOCKWISE)
# multiplePhotoSave('images/for-calibration', 'X', 'jpg', 50)