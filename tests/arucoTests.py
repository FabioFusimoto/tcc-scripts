import cv2.cv2 as cv2
import glob
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
import src.calibration.commons as commons
from src.server.coordinatesEstimation import estimatePosesFromPivot

def createArucoGrid(nx, ny, firstMarker, outputFile):
    print('It should generate a grid with size nx (width) x ny (height)')
    arucoMarkers.generateMarkerGrid(nx, ny, firstMarker, outputFile)

def findMarkersOn(sourceFile, outputFile, shouldSave, scale):
    print('Finding markers on ' + sourceFile)
    arucoMarkers.highlightDetectedMarkers(sourceFile, outputFile, shouldSave, scale)

def estimateMarkersPose(sourceFile, outputFile, scale, markerId, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = commons.loadCalibrationCoefficients(calibrationFile)
    arucoMarkers.estimatePose(sourceFile, outputFile, scale, markerId, markerLength, cameraMatrix, distCoeffs)

def writeCoordinatesToCSV(sourceFile, outputFile, scale, markerIds, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = commons.loadCalibrationCoefficients(calibrationFile)
    arucoMarkers.exportCoordinatesToFile(sourceFile, outputFile, scale, markerIds, markerLength, cameraMatrix, distCoeffs)

def estimateMarkerPoseFromPivot(sourceFile, markersToEstimate, pivotMarkerId, markerLength, calibrationFile):
    cameraMatrix, distCoeffs = commons.loadCalibrationCoefficients(calibrationFile)

    poses = estimatePosesFromPivot(markersToEstimate, pivotMarkerId, markerLength, cameraMatrix, distCoeffs, image=cv2.imread(sourceFile))

    for _, data in poses.items():
        if data['found']:
            data['pose']['roll'] *= 180/np.pi
            data['pose']['pitch'] *= 180/np.pi
            data['pose']['yaw'] *= 180/np.pi

    np.set_printoptions(precision=4, suppress=True)
    pprint.pprint(poses)

createArucoGrid(2, 5, 0, 'images/arucoGrid.jpg')

#findMarkersOn('images/for-calibration/ARUZ50.jpg', 'images/calibration-output/ARUZ50_markers_found.jpg',
#              True, 0.75)

#imageFiles = glob.glob('images/img*.jpg')
#calibrationFile = 'src/server/files/J7-pro.yml'

#for img in imageFiles:
#    print('\nEstimating pose on ' + img.split('\\')[-1])
#    outputFile = 'images/calibration-output/' + img.split('\\')[-1].split('.')[0] + '-pose.jpg'
#    print('Saving coordinates to ' + outputFile)
#    estimateMarkersPose(img, outputFile, 1, 3, 3.78, calibrationFile)

#for img in imageFiles:
#    print('\nEstimating pose on ' + img.split('\\')[-1])
#    outputFile = 'Z:/workspace/OpenCV_Integration/Content/' + img.split('\\')[-1].split('.')[0] + '-coords.csv'
#    print('Saving coordinates to ' + outputFile)
#    writeCoordinatesToCSV(img, outputFile, 0.75, [0, 1, 3], 3.78, calibrationFile)

#for img in imageFiles:
#    print('\n\n\n>>>>>Estimating pose on: ' + img + '<<<<<')
#    estimateMarkerPoseFromPivot(img, [], 3, 5.28, calibrationFile)