import cv2.cv2 as cv2
import glob

path = 'images/for-calibration'
prefix = 'IMG'
extension = 'jpg'

newPrefix = 'X'

files = glob.glob(path + '/' + prefix + '*.' + extension)

for i in range(len(files)):
    photo =  cv2.imread(files[i])
    filename = path + '/' + newPrefix + format(i, '02d') + '.' + extension
    print(filename)
    cv2.imwrite(filename, photo)

print(str(len(files)) + ' photos renamed')