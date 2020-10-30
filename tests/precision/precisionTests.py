import glob
import math
import pprint

import src.calibration.commons as commons
from tests.precision.helpers import discoverPivot, exportPrecisionResultsToFile, getHMDPoseDifference, markerPose 

def precisionTest(markerId, resolution, field, expectedValue, cameraMatrix, distortionCoefficients, isAngle=False):
    imageFiles = glob.glob('tests/precision/images/{}-{}-{}*.jpg'.format(resolution, field, expectedValue))
    results = []
    for imageFile in imageFiles:
        print('\nExecuting precision test on file: {}'.format(imageFile))
        coordinates = markerPose(markerId, cameraMatrix, distortionCoefficients, imageFile)
        if isAngle:
            results.append(math.degrees(coordinates[field]))
        else:
            results.append(coordinates[field])

    return results   

def zTests(markerId, resolution, cameraMatrix, distortionCoefficients):
    distances = [15, 20, 25, 30, 35, 40, 45]
    results = {}
    for d in distances:
        results[d] = precisionTest(markerId, resolution, 'z', d, cameraMatrix, distortionCoefficients)

    outputFile = 'tests/precision/results/{}-{}.csv'.format(resolution, 'z')
    exportPrecisionResultsToFile(outputFile, results)

def pitchTests(referenceMarkerId, resolution, cameraMatrix, distortionCoefficients):
    angles = [-70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60, 70]

    results = {}
    for a in angles:
        results[a] = precisionTest(referenceMarkerId, resolution, 'pitch', a, cameraMatrix, distortionCoefficients, isAngle=True)

    outputFile = 'tests/precision/results/{}-{}.csv'.format(resolution, 'pitch')
    exportPrecisionResultsToFile(outputFile, results)

def consistencyTest(referenceMarkerId, pivotMarkerId, resolution, cameraMatrix, distortionCoefficients, repetitions):
    samples = []

    for i in range(repetitions):
        referenceAndPivotImage = glob.glob('tests/precision/images/{}-consistency-{}-reference-and-pivot.jpg'.format(resolution, i))[0]
        referencePoseRelativeToPivot = discoverPivot(pivotMarkerId, referenceMarkerId, cameraMatrix, distortionCoefficients, referenceAndPivotImage)

        differences = []
        
        for imageFile in glob.glob('tests/precision/images/{}-consistency-{}-sample-*.jpg'.format(resolution, i)):
            differences.append(getHMDPoseDifference(pivotMarkerId, referenceMarkerId, referencePoseRelativeToPivot, 
                                                    cameraMatrix, distortionCoefficients, imageFile))
        
        samples.append(differences)

    print('\nSamples')
    pprint.pprint(samples)

def testMultiple(tests, referenceMarkerId, pivotMarkerId, resolution, consistencyRepetitions):
    calibrationFile = 'tests/calibration-coefficients/J7-pro-{}.yml'.format(resolution)
    cameraMatrix, distortionCoefficients = commons.loadCalibrationCoefficients(calibrationFile)

    if 'z' in tests:
        zTests(referenceMarkerId, resolution, cameraMatrix, distortionCoefficients)
    if 'pitch' in tests:
        pitchTests(referenceMarkerId, resolution, cameraMatrix, distortionCoefficients)
    if 'consistency' in tests:
        consistencyTest(referenceMarkerId, pivotMarkerId, resolution, cameraMatrix, distortionCoefficients, consistencyRepetitions)

testMultiple(['pitch'], 7, 3, '2322p', 5)