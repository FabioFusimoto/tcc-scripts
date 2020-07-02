import pprint

import src.calibration.chessboard as chess

def getCoefficients(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize, boardWidth, boardHeight, downsize, scale):
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
    scaleAsString = str(scale * 100).split('.')[0]
    filename = 'tests/calibration-coefficients/g7-play-' + scaleAsString + '-percent-resolution.yml'
    chess.saveCalibrationCoeficients(mtx, dist, filename)

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

# getCoefficients('images/for-calibration', 'images/calibration-output', 'C', 'jpg', 17.85 / 6, 9, 6, True, 0.25)

# undistortImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-undistorted.jpg', 
  #              'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 0.75)

drawAxisOnChessboardImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-axis.jpg', 2,
                          'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 9, 6 , 0.75)