from timeit import default_timer as timer

import cv2.cv2 as cv2
import glob
import numpy as np

# Inside corners in x
nx = 9
# Inside corners in y
ny = 6

# List of calibration images
chessImages = glob.glob('images/for-calibration/C*.jpg')

# Downsizing options
shouldDownsize = True
scale = 0.5

# Chessboard search flags
flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

imageTotal = len(chessImages)
print(str(imageTotal) + ' chessboard images found')

patternsFound = []
patternsNotFound = []

start = timer()

for i in range(len(chessImages)):
    print('\nProcessing image ' + str(chessImages[i]).split('\\')[1])

    # Read in the image
    readStart = timer()
    chessBoardImage = cv2.imread(chessImages[i])
    readEnd = timer()
    print('Time elapsed to read the image: ' + str(readEnd - readStart))

    # Downsizing the image
    if (shouldDownsize):
        chessBoardImage = cv2.resize(chessBoardImage, None, fx=scale, fy=scale) 

    # Convert to grayscale
    conversionStart = timer()
    gray = cv2.cvtColor(chessBoardImage, cv2.COLOR_RGB2GRAY)
    conversionEnd = timer()
    print('Time elapsed to convert to grayscale: ' + str(conversionEnd - conversionStart))

    # Find and draw the chessboard corners
    cornersStart = timer()
    
    ret, corners = cv2.findChessboardCorners(chessBoardImage, (nx, ny), flags)
    if(ret == True):
        patternsFound.append(str(chessImages[i]).split('\\')[1])
        cv2.drawChessboardCorners(chessBoardImage, (nx, ny), corners, True)
        result_name = 'images/calibration-output/patterns-' + str(chessImages[i]).split('\\')[1] + '.jpg'
        cv2.imwrite(result_name, chessBoardImage)
        print('Pattern found')
    else:
        patternsNotFound.append(str(chessImages[i]).split('\\')[1])
        print('Pattern not found')

    cornersEnd = timer()
    print('Time elapsed to find corners: ' + str(cornersEnd - cornersStart))

end = timer()

print('\n\n\nTime elapsed to process all images: ' + str(end - start))
print(str(len(patternsFound)) + '/' + str(imageTotal) + ' patterns found on images')

print('\nPatterns found on images: ' + str(patternsFound))
print('\nPatterns not found on images: ' + str(patternsNotFound))