import pprint

import src.calibration.chessboard as chess

def getCoefficients(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize, boardWidth, boardHeight, downsize, scale, calibrationFilename):
    '''Calculates the calibration coefficients, prints them and stores them as a YML file'''
    ret, mtx, dist, rvecs, tvecs = chess.calibrate(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize,
                                                   boardWidth, boardHeight, downsize, scale)

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

def drawAxisOnChessboardImage(imageSource, imageOutput, axisLength, calibrationFile, width=9, height=6, scale=0.75):
    '''Draws an axis on a chessboard figure and optionally saves it to imageOutput'''
    
    cameraMatrix, distortionCoefficients = chess.loadCalibrationCoeficients(calibrationFile)
    chess.drawOnChessboard(imageSource, imageOutput, axisLength, cameraMatrix, distortionCoefficients,
                           width, height, scale)

def findChessboardCoordinates(imageSource, calibrationFile, squareSize, width, height, scale):
    cameraMatrix, distortionCoefficients = chess.loadCalibrationCoeficients(calibrationFile)
    rotationVectors, translationVectors = chess.getPositionVectors(imageSource, cameraMatrix, distortionCoefficients,
                                                                   squareSize, width, height, scale)
    
    print('\nRotation vectors:')
    pprint.pprint(rotationVectors)

    print('\nTranslation vectors:')
    pprint.pprint(translationVectors)

def drawPositionVectors(imageSource, imageOutput, calibrationFile, squareSize, width, height, scale):
    cameraMatrix, distortionCoefficients = chess.loadCalibrationCoeficients(calibrationFile)
    chess.drawPositionVectors(imageSource, imageOutput, cameraMatrix, distortionCoefficients, squareSize, width, height, scale)

# getCoefficients('images/for-calibration', 'images/calibration-output', 'X', 'jpg', 0.228/9, 9, 6, True, 0.75,
#                 'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml') # measured 22.8cm - 9 squares

# undistortImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-undistorted.jpg', 
#                'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 0.75)

# drawAxisOnChessboardImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-axis.jpg', 2,
#                           'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 9, 6 , 0.75)

# findChessboardCoordinates('images/for-calibration/D100cm.jpg', 'tests/calibration-coefficients/g7-play-100-percent-resolution.yml',
#                           0.228/9, 9, 6, 1.0) # measured 22.8cm - 9 squares

drawPositionVectors('images/for-calibration/Z100-00.jpg', 'images/calibration-output/Z100-00-coords.jpg',
                    'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml', 0.228/9, 9, 6, 0.75)