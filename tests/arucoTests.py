import glob

import src.calibration.arucoMarkers as arucoMarkers
import src.calibration.chessboard as chess

def createArucoGrid(nx, ny, outputFile):
    print('It should generate a grid with size nx (width) x ny (height')
    arucoMarkers.generateMarkerGrid(nx, ny, outputFile)

def findMarkersOn(sourceFile, outputFile, shouldSave, scale):
    print('Finding markers on ' + sourceFile)
    arucoMarkers.highlightDetectedMarkers(sourceFile, outputFile, shouldSave, scale)

def estimateMarkersPose(sourceFile, outputFile, scale, markerId, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = chess.loadCalibrationCoeficients(calibrationFile)
    arucoMarkers.estimatePose(sourceFile, outputFile, scale, markerId, markerLength, cameraMatrix, distCoeffs)

#createArucoGrid(4, 3, 'images/arucoGrid.jpg')

#findMarkersOn('images/for-calibration/ARUZ50.jpg', 'images/calibration-output/ARUZ50_markers_found.jpg',
#              True, 0.75)

imageFiles = glob.glob('images/for-calibration/ARUZ*.jpg')
calibrationFile = 'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml'

for img in imageFiles:
    print('\nEstimating pose on ' + img.split('\\')[-1])
    outputFile = 'images/calibration-output/' + img.split('\\')[-1].split('.')[0] + '-pose.jpg'
    print('Saving coordinates to ' + outputFile)
    estimateMarkersPose(img, outputFile, 0.75, 6, 4.4, calibrationFile)