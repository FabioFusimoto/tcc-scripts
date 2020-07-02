import pprint

import src.calibration.chessboard as chess

calibrationImagesPath = 'images/for-calibration'
calibrationImagesPrefix = 'C'
calibrationImagesFormat = 'jpg'

realChessboardSquareSize = 17.85 / 6 # in meters - total length / square count
realChessboardWidth = 9
realChessboardHeight = 6

shouldDownsize = True
downsizeScale = 0.5

# Getting calibration coefficients
ret, mtx, dist, rvecs, tvecs = chess.calibrate(calibrationImagesPath,
                                               calibrationImagesPrefix,
                                               calibrationImagesFormat,
                                               realChessboardSquareSize,
                                               realChessboardWidth,
                                               realChessboardHeight,
                                               shouldDownsize,
                                               downsizeScale)

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
filename = 'tests/calibration-coefficients/g7-play.yml'
chess.saveCalibrationCoeficients(mtx, dist, filename)