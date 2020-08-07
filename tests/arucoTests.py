import glob

import src.calibration.arucoMarkers as arucoMarkers
import src.calibration.commons as commons

def createArucoGrid(nx, ny, outputFile):
    print('It should generate a grid with size nx (width) x ny (height)')
    arucoMarkers.generateMarkerGrid(nx, ny, outputFile)

def findMarkersOn(sourceFile, outputFile, shouldSave, scale):
    print('Finding markers on ' + sourceFile)
    arucoMarkers.highlightDetectedMarkers(sourceFile, outputFile, shouldSave, scale)

def estimateMarkersPose(sourceFile, outputFile, scale, markerId, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = commons.loadCalibrationCoefficients(calibrationFile)
    arucoMarkers.estimatePose(sourceFile, outputFile, scale, markerId, markerLength, cameraMatrix, distCoeffs)

def writeCoordinatesToCSV(sourceFile, outputFile, scale, markerIds, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = commons.loadCalibrationCoefficients(calibrationFile)
    arucoMarkers.exportCoordinatesToFile(sourceFile, outputFile, scale, markerIds, markerLength, cameraMatrix, distCoeffs)

#createArucoGrid(2, 4, 'images/arucoGrid.jpg')

#findMarkersOn('images/for-calibration/ARUZ50.jpg', 'images/calibration-output/ARUZ50_markers_found.jpg',
#              True, 0.75)

imageFiles = glob.glob('images/pivot*.jpg')
calibrationFile = 'tests/calibration-coefficients/g7-play-1280x720.yml'

for img in imageFiles:
    print('\nEstimating pose on ' + img.split('\\')[-1])
    outputFile = 'images/calibration-output/' + img.split('\\')[-1].split('.')[0] + '-pose.jpg'
    print('Saving coordinates to ' + outputFile)
    estimateMarkersPose(img, outputFile, 1, 3, 3.78, calibrationFile)

#for img in imageFiles:
#    print('\nEstimating pose on ' + img.split('\\')[-1])
#    outputFile = 'Z:/workspace/OpenCV_Integration/Content/' + img.split('\\')[-1].split('.')[0] + '-coords.csv'
#    print('Saving coordinates to ' + outputFile)
#    writeCoordinatesToCSV(img, outputFile, 0.75, [0, 1, 3], 3.78, calibrationFile)