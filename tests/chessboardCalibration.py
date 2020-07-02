import pprint

import src.calibration.chessboard as chess

def getCoefficients(imgSrcPath, imgOutPath, imgPrefix, imgFormat, sqrSize, boardWidth, boardHeight, downsize, scale):
    # Calculating coefficients
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

# getCoefficients('images/for-calibration', 'images/calibration-output', 'C', 'jpg', 17.85 / 6, 9, 6, True, 0.25)

undistortImage('images/for-calibration/C11.jpg', 'images/calibration-output/C11-undistorted.jpg', 
               'tests/calibration-coefficients/g7-play-75-percent-resolution.yml', 0.75)
