import pprint
import glob

import src.calibration.chessboard as chess

def getCoefficients(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize, boardWidth, boardHeight, scale, calibrationFilename):
    '''Calculates the calibration coefficients, prints them and stores them as a YML file'''
    ret, mtx, dist, rvecs, tvecs = chess.calibrate(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize,
                                                   boardWidth, boardHeight, scale)

    print('\nReturn value:')
    pprint.pprint(ret)

    print('\nCamera intrinsic parameters matrix:')
    pprint.pprint(mtx)

    print('\nDistortion coefficients:')
    pprint.pprint(dist)

    print('\nRotation vectors:')
    pprint.pprint(rvecs)

    print('\nTranslation vectors:')
    pprint.pprint(tvecs)

    # Saving the calibration coefficients
    chess.saveCalibrationCoeficients(mtx, dist, calibrationFilename)

def undistortImage(imageSource, imageOutput, calibrationFile, scale):
    '''Undistorts image based on a calibration file. The calibration is resolution sensitive,
       so the chosen scale must be on par with the given calibration'''

    cameraMatrix, distortionCoefficients = chess.loadCalibrationCoeficients(calibrationFile)
    chess.undistortImage(imageSource, imageOutput, cameraMatrix, distortionCoefficients, scale)

def drawPositionVectors(imageSource, imageOutput, calibrationFile, squareSize, width, height, scale):
    cameraMatrix, distortionCoefficients = chess.loadCalibrationCoeficients(calibrationFile)
    chess.drawPositionVectors(imageSource, imageOutput, cameraMatrix, distortionCoefficients, squareSize, width, height, scale)

# getCoefficients('images/for-calibration', 'images/calibration-output', 'X', 'jpg', 0.228/9, 6, 9, 0.75,
#                 'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml') # measured 22.8cm - 9 squares

# undistortImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-undistorted.jpg', 
#                'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 0.75)

# findChessboardCoordinates('images/for-calibration/D100cm.jpg', 'tests/calibration-coefficients/g7-play-100-percent-resolution.yml',
#                           0.228/9, 6, 9, 1.0) # measured 22.8cm - 9 squares

poseCalibImages = glob.glob('images/for-calibration/Z*.jpg')
calibrationFile = 'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml'

for img in poseCalibImages:
    print('\nEstimating pose on ' + img.split('\\')[-1])
    outputFile = 'images/calibration-output/' + img.split('\\')[-1].split('.')[0] + '-coords.jpg'
    print('Saving coordinates to ' + outputFile)
    drawPositionVectors(img, outputFile, calibrationFile, 0.228/9, 6, 9, 0.75)