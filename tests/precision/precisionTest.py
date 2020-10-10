import glob
import math
import pprint

import src.calibration.commons as commons
from tests.precision.helpers import markerPose, exportResultsToFile

def precisionTest(markerId, resolution, field, expectedValue, cameraMatrix, distortionCoefficients, isAngle=False):
    imageFiles = glob.glob('tests/precision/images/{}-{}-{}*.jpg'.format(resolution, field, expectedValue))
    results = []
    for imageFile in imageFiles:
        print('\nEstimating pose on file: {}'.format(imageFile))
        coordinates = markerPose(markerId, cameraMatrix, distortionCoefficients, imageFile)
        if isAngle:
            results.append(math.degrees(coordinates[field]))
        else:
            results.append(coordinates[field])

    return results   

def zTests(markerId, resolution):
    calibrationFile = 'tests/calibration-coefficients/J7-pro-{}.yml'.format(resolution)
    cameraMatrix, distortionCoefficients = commons.loadCalibrationCoefficients(calibrationFile)

    distances = [15, 20, 25, 30, 35, 40, 45]
    results = {}
    for d in distances:
        results[d] = precisionTest(markerId, resolution, 'z', d, cameraMatrix, distortionCoefficients)

    outputFile = 'tests/precision/results/{}-{}.csv'.format(resolution, 'z')
    exportResultsToFile(outputFile, results)

def pitchTests(markerId, resolution):
    calibrationFile = 'tests/calibration-coefficients/J7-pro-{}.yml'.format(resolution)
    cameraMatrix, distortionCoefficients = commons.loadCalibrationCoefficients(calibrationFile)

    angles = [-70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60, 70]

    results = {}
    for a in angles:
        results[a] = precisionTest(markerId, resolution, 'pitch', a, cameraMatrix, distortionCoefficients, isAngle=True)

    outputFile = 'tests/precision/results/{}-{}.csv'.format(resolution, 'pitch')
    exportResultsToFile(outputFile, results)
    
#zTests(7, '720p')
pitchTests(7, '720p')