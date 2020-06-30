import src.webcamUtilities.video as video
import src.webcamUtilities.photo as photo

def testVideoStream():
    print('Should display a video stream (press Q to exit)')
    video.streamVideo()

def testPhotoCapture():
    print('It should take a picture and display it on screen (press Q to exit)')
    photo.takePicture()

def testPhotoSave(filename='test.jpg'):
    print('It should take a picture and save it')
    print('Success? -> ' + str(photo.savePhoto(filename)))