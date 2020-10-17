from collections import OrderedDict
import csv
import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, composeRotations, getEulerAnglesFromRVector, getRVectorFromEulerAngles, \
                                    getTransformationMatrix, inversePerspective, relativePosition, transformCoordinates

MARKER_LENGTH = 5.3

###########################
# PRECISION TESTS HELPERS #
###########################
def markerPose(markerId, cameraMatrix, distCoeffs, imageFile):
    image = cv2.imread(imageFile)

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    indexes = np.where(ids == markerId)[0]

    if indexes.size > 0:
        i = indexes[0]
        coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=MARKER_LENGTH)
        return coords
    else:
        return {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}

def exportPrecisionResultsToFile(outputFile, results):
    header = ['Valor esperado', 'Amostra 1', 'Amostra 2', 'Amostra 3', 'Erro medio absoluto', 'Erro medio relativo (%)']

    rows = [header]

    for expectedValue, samples in results.items():
        sampleAverage = sum(samples) / len(samples)
        averageError = abs(sampleAverage - expectedValue)
        relativeErrorPercentage = abs(averageError/expectedValue) * 100

        rows.append(['{:.2f}'.format(expectedValue), '{:.2f}'.format(samples[0]), '{:.2f}'.format(samples[1]), 
                     '{:.2f}'.format(samples[2]), '{:.2f}'.format(averageError), '{:.2f}'.format(relativeErrorPercentage)])

    with open(outputFile, mode='w+', newline='') as resultsCSV:
        writer = csv.writer(resultsCSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)

############################
# CONSISTENCY TEST HELPERS #
############################

def discoverPivot(targetPivotId, referenceId, cameraMatrix, distCoeffs, imageFile):
    image = cv2.imread(imageFile)
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    pivotIndexes = np.where(ids == int(targetPivotId))[0]
    referenceIndexes = np.where(ids == int(referenceId))[0]

    pivotIndex = pivotIndexes[0]
    referenceIndex = referenceIndexes[0]

    # Pivot pose
    pivotRVec = rVecs[pivotIndex]
    pivotTVec = tVecs[pivotIndex]

    # Reference pose relative to pivot coords
    referenceRVec = rVecs[referenceIndex]
    referenceTVec = tVecs[referenceIndex]

    referenceRotationRelativeToPivotAsEuler, referenceTranslationRelativeToPivot = relativePosition(referenceRVec, 
                                                                                                    referenceTVec, 
                                                                                                    pivotRVec,
                                                                                                    pivotTVec, 
                                                                                                    scale=MARKER_LENGTH, 
                                                                                                    asEuler=True)

    referencePoseRelativeToPivot = {
        'x': referenceTranslationRelativeToPivot[0],
        'y': referenceTranslationRelativeToPivot[1],
        'z': referenceTranslationRelativeToPivot[2],
        'roll': referenceRotationRelativeToPivotAsEuler[0],
        'pitch': referenceRotationRelativeToPivotAsEuler[1],
        'yaw': referenceRotationRelativeToPivotAsEuler[2]
    }

    return {targetPivotId: referencePoseRelativeToPivot}

def normalizeCoordinates(coords):
    newCoords = coords

    for k in ['roll', 'pitch', 'yaw']:
        while abs(newCoords[k]) > np.pi: # Make it between -pi (-180°) and +pi (+180°)
            if newCoords[k] >= np.pi:
                newCoords[k] -= 2*np.pi
            elif newCoords[k] < -np.pi:
                newCoords[k] += 2*np.pi
    
    return newCoords

def getHMDPoseDifference(pivotId, referenceId, referencePoseRelativeToPivot, cameraMatrix, distCoeffs, imageFile):
    image = cv2.imread(imageFile)
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    hmdPoses = {}
    
    for targetPivotId in [pivotId, referenceId]:
        targetPivotIndex = np.where(ids == targetPivotId)[0][0]

        pivotRVecRelativeToCamera = rVecs[targetPivotIndex]
        pivotTVecRelativeToCamera = np.dot(MARKER_LENGTH, tVecs[targetPivotIndex])

        cameraRVecRelativeToPivot, cameraTVecRelativeToPivot = inversePerspective(pivotRVecRelativeToCamera, 
                                                                                  pivotTVecRelativeToCamera)

        referencePoseRelativeToPivot = referencePoseRelativeToPivot.get(targetPivotId, {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        })

        referenceRVecRelativeToPivot = np.array(getRVectorFromEulerAngles(
            referencePoseRelativeToPivot['roll'],
            referencePoseRelativeToPivot['pitch'],
            referencePoseRelativeToPivot['yaw']
        ))

        referenceTVecRelativeToPivot = np.array([
            [referencePoseRelativeToPivot['x']],
            [referencePoseRelativeToPivot['y']],
            [referencePoseRelativeToPivot['z']]
        ])

        cameraRotationRelativeToReference, cameraTranslationRelativeToReference = relativePosition(cameraRVecRelativeToPivot,
                                                                                                   cameraTVecRelativeToPivot,
                                                                                                   referenceRVecRelativeToPivot,
                                                                                                   referenceTVecRelativeToPivot,
                                                                                                   asEuler=True)

        cameraPoseRelativeToReference = {
            'x': cameraTranslationRelativeToReference[0],
            'y': cameraTranslationRelativeToReference[1],
            'z': cameraTranslationRelativeToReference[2],
            'roll': cameraRotationRelativeToReference[0],
            'pitch': cameraRotationRelativeToReference[1],
            'yaw': cameraRotationRelativeToReference[2]
        }

        hmdPoses[targetPivotId] = cameraPoseRelativeToReference

    keys = ['roll', 'pitch', 'yaw', 'x', 'y', 'z']
    differencesAsList = [abs(hmdPoses[pivotId][k] - hmdPoses[referenceId][k]) for k in keys]

    differencesAsDict = OrderedDict([(keys[0], differencesAsList[0]), (keys[1], differencesAsList[1]), (keys[2], differencesAsList[2]), 
                                     (keys[3], differencesAsList[3]), (keys[4], differencesAsList[4]), (keys[5], differencesAsList[5])])

    return differencesAsDict