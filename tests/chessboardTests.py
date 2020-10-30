import pprint
import glob

import src.calibration.chessboard as chess
import src.calibration.commons as commons

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
    commons.saveCalibrationCoefficients(mtx, dist, calibrationFilename)

def undistortImage(imageSource, imageOutput, calibrationFile, scale):
    '''Undistorts image based on a calibration file. The calibration is resolution sensitive,
       so the chosen scale must be on par with the given calibration'''

    cameraMatrix, distortionCoefficients = commons.loadCalibrationCoefficients(calibrationFile)
    chess.undistortImage(imageSource, imageOutput, cameraMatrix, distortionCoefficients, scale)

def drawPositionVectors(imageSource, imageOutput, calibrationFile, squareSize, width, height, scale):
    cameraMatrix, distortionCoefficients = commons.loadCalibrationCoefficients(calibrationFile)
    chess.drawPositionVectors(imageSource, imageOutput, cameraMatrix, distortionCoefficients, squareSize, width, height, scale)

getCoefficients('images/for-calibration', 'images/calibration-output', '4128x2322', 'jpg', 22.8/9, 6, 9, 1.0,
                'tests/calibration-coefficients/J7-pro-4122x2322.yml') # measured 22.8cm - 9 squares

#undistortImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-undistorted.jpg', 
#               'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 0.75)

#findChessboardCoordinates('images/for-calibration/D100cm.jpg', 'tests/calibration-coefficients/g7-play-100-percent-resolution.yml',
#                          22.8/9, 6, 9, 1.0) # measured 22.8cm - 9 squares

#poseCalibImages = glob.glob('images/for-calibration/chessboard-1280x720*.jpg')
#calibrationFile = 'tests/calibration-coefficients/g7-play-landscape-1280x720.yml'

#for img in poseCalibImages:
#    print('\nEstimating pose on ' + img.split('\\')[-1])
#    outputFile = 'images/calibration-output/' + img.split('\\')[-1].split('.')[0] + '-coords.jpg'
#    print('Saving coordinates to ' + outputFile)
#    drawPositionVectors(img, outputFile, calibrationFile, 22.8/9, 6, 9, 1)