import cv2.cv2 as cv2
import glob

def rotateImage(image, rotationType, filename, overwrite):
    rotatedImage = cv2.rotate(image, rotationType)
    if(overwrite):
        cv2.imwrite(filename, rotatedImage)
    return rotatedImage

#imageFiles = glob.glob('images/for-calibration/Z100*.jpg')
#for i in imageFiles:
#    img = cv2.imread(i)
#    rotateImage(img, cv2.ROTATE_90_CLOCKWISE, i, True)