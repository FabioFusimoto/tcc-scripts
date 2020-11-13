import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, getRMatrixFromVector, getRMatrixFromEulerAngles, getRVectorFromEulerAngles, getEulerAnglesFromRVector, \
                                    calculateRelativePoseFromVectors, calculateRelativePoseFromPose, getTransformationMatrix, relativePosition, \
                                    inversePerspective, transformCoordinates
from src.server.objects import OBJECT_DESCRIPTION

def estimatePoses(markerIds, cameraMatrix, distCoeffs, cam, camType):
    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    poses = {}
    for markerId in markerIds:
        indexes = np.where(ids == markerId)[0]
        if indexes.size > 0:
            i = indexes[0]
            pose = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=OBJECT_DESCRIPTION[str(markerId)]['length'])
            poses[str(markerId)] = pose

    return poses

def estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam=None, camType=None, image=None):
    np.set_printoptions(precision=4, suppress=True)

    if image is None:
        image = cam.read()
        while (camType == 'USB') and (cam.grabbed == False):
            image = cam.read()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    if ids is None or pivotMarkerId not in ids:
        return {}

    posesFound = {}

    pivotIndexes = np.where(ids == pivotMarkerId)[0]

    if pivotIndexes.size > 0:
        i = pivotIndexes[0]

        pivotLength = OBJECT_DESCRIPTION[str(pivotMarkerId)]['length']

        pivotRVecRelativeToCamera = rVecs[i]
        pivotTVecRelativeToCamera = np.dot(pivotLength, tVecs[i])

        cameraRVecRelativeToPivot, cameraTVecRelativeToPivot = inversePerspective(pivotRVecRelativeToCamera, 
                                                                                  pivotTVecRelativeToCamera)

        cameraRotationRelativeToReference, cameraTranslationRelativeToReference = relativePosition(
            cameraRVecRelativeToPivot,
            cameraTVecRelativeToPivot,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            asEuler=True
        )

        cameraPoseRelativeToReference = {
            'x': cameraTranslationRelativeToReference[0],
            'y': cameraTranslationRelativeToReference[1],
            'z': cameraTranslationRelativeToReference[2],
            'roll': cameraRotationRelativeToReference[0],
            'pitch': cameraRotationRelativeToReference[1],
            'yaw': cameraRotationRelativeToReference[2]
        }

        posesFound = {'hmd': cameraPoseRelativeToReference,
                      str(pivotMarkerId): {
                            'x': 0.0,
                            'y': 0.0,
                            'z': 0.0,
                            'roll': 0.0,
                            'pitch': 0.0,
                            'yaw': 0.0
                        }}

        for markerId in markerIds:
            indexes = np.where(ids == markerId)[0]
            if indexes.size > 0:
                i = indexes[0]

                markerLength = OBJECT_DESCRIPTION[str(markerId)]['length']

                markerRVecRelativeToCamera = rVecs[i]
                markerTVecRelativeToCamera = np.dot(markerLength, tVecs[i])

                markerRotationRelativeToReference, markerTranslationRelativeToReference = relativePosition(
                    markerRVecRelativeToCamera,
                    markerTVecRelativeToCamera,
                    pivotRVecRelativeToCamera,
                    pivotTVecRelativeToCamera,
                    asEuler=True
                )

                markerPoseRelativeToReference = {
                    'x': markerTranslationRelativeToReference[0],
                    'y': markerTranslationRelativeToReference[1],
                    'z': markerTranslationRelativeToReference[2],
                    'roll': markerRotationRelativeToReference[0],
                    'pitch': markerRotationRelativeToReference[1],
                    'yaw': markerRotationRelativeToReference[2]
                }

                posesFound[str(markerId)] = markerPoseRelativeToReference

        return posesFound

    return {}

def discoverPivot(targetPivotId, referenceId, cameraMatrix, distCoeffs, cam=None):
    np.set_printoptions(precision=4, suppress=True)

    image = cam.read()

    while cam.grabbed == False:
        image = cam.read()

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    pivotIndexes = np.where(ids == int(targetPivotId))[0]
    referenceIndexes = np.where(ids == int(referenceId))[0]

    if pivotIndexes.size == 0 or referenceIndexes.size == 0:
        return {}

    pivotIndex = pivotIndexes[0]
    referenceIndex = referenceIndexes[0]

    # Pivot pose relative to camera
    pivotLength = OBJECT_DESCRIPTION[str(targetPivotId)]['length']
    pivotRVec = rVecs[pivotIndex]
    pivotTVec = np.dot(pivotLength, tVecs[pivotIndex])
    

    # Reference pose relative to pivot coords
    referenceLength = OBJECT_DESCRIPTION[str(referenceId)]['length']
    referenceRVec = rVecs[referenceIndex]
    referenceTVec = np.dot(referenceLength, tVecs[referenceIndex])

    referenceRotationRelativeToPivot, referenceTranslationRelativeToPivot = relativePosition(referenceRVec, 
                                                                                             referenceTVec, 
                                                                                             pivotRVec,
                                                                                             pivotTVec, 
                                                                                             asEuler=True)

    pivotRotationRelativeToReference, pivotTranslationRelativeToReference = relativePosition(pivotRVec,
                                                                                             pivotTVec,
                                                                                             referenceRVec, 
                                                                                             referenceTVec,
                                                                                             asEuler=True)

    referencePoseRelativeToPivot = {
        'x': referenceTranslationRelativeToPivot[0],
        'y': referenceTranslationRelativeToPivot[1],
        'z': referenceTranslationRelativeToPivot[2],
        'roll': referenceRotationRelativeToPivot[0],
        'pitch': referenceRotationRelativeToPivot[1],
        'yaw': referenceRotationRelativeToPivot[2]
    }

    pivotPoseRelativeRelativeToReference = {
        'x': pivotTranslationRelativeToReference[0],
        'y': pivotTranslationRelativeToReference[1],
        'z': pivotTranslationRelativeToReference[2],
        'roll': pivotRotationRelativeToReference[0],
        'pitch': pivotRotationRelativeToReference[1],
        'yaw': pivotRotationRelativeToReference[2]
    }

    return referencePoseRelativeToPivot, pivotPoseRelativeRelativeToReference

def selectPivot(possiblePivotIds, foundMarkerIds):
    for pivotId in possiblePivotIds:
        if pivotId in foundMarkerIds:
            return pivotId
    
    return None

def estimatePosesFromMultiplePivots(markerIds, pivotIds, referenceId, referencePoseRelativeToPivots, \
                                    cameraMatrix, distCoeffs, cam=None, camType='USB'):
    np.set_printoptions(precision=4, suppress=True)

    # print('\n\n\n\nReference pose relative to pivots')
    # pprint.pprint(referencePoseRelativeToPivots)

    image = cam.read()
    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    if ids is None:
        return {}

    targetPivotId = selectPivot(pivotIds, ids) # Target is the one found on image

    if targetPivotId is None: # Meaning no pivot was found on image
        return {}

    # print('\nTarget pivot: {}'.format(targetPivotId))

    targetPivotLength = OBJECT_DESCRIPTION[str(targetPivotId)]['length']
    targetPivotIndexes = np.where(ids == int(targetPivotId))[0]

    if targetPivotIndexes.size == 0:
        return {}

    targetPivotIndex = targetPivotIndexes[0]

    pivotRVecRelativeToCamera = rVecs[targetPivotIndex]
    pivotTVecRelativeToCamera = np.dot(targetPivotLength, tVecs[targetPivotIndex])

    cameraRVecRelativeToPivot, cameraTVecRelativeToPivot = inversePerspective(pivotRVecRelativeToCamera, 
                                                                              pivotTVecRelativeToCamera)

    referencePoseRelativeToPivot = referencePoseRelativeToPivots.get(str(targetPivotId), {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0
    })

    # print('\nReference pose relative to pivot')
    # pprint.pprint(referencePoseRelativeToPivot)

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

    posesFound = {'hmd': cameraPoseRelativeToReference}

    for markerId in markerIds:
        indexes = np.where(ids == markerId)[0]
        if indexes.size > 0:
            i = indexes[0]

            markerLength = OBJECT_DESCRIPTION[str(markerId)]['length']

            markerRVecRelativeToCamera = rVecs[i]
            markerTVecRelativeToCamera = np.dot(markerLength, tVecs[i])

            markerRVecRelativeToPivot, markerTVecRelativeToPivot = relativePosition(
                markerRVecRelativeToCamera,
                markerTVecRelativeToCamera,
                pivotRVecRelativeToCamera,
                pivotTVecRelativeToCamera,
                asEuler=False
            )
            
            markerRotationRelativeToReference, markerTranslationRelativeToReference = relativePosition(
                markerRVecRelativeToPivot,
                markerTVecRelativeToPivot,
                referenceRVecRelativeToPivot,
                referenceTVecRelativeToPivot,
                asEuler=True
            )
            
            markerPoseRelativeToReference = {
                'x': markerTranslationRelativeToReference[0],
                'y': markerTranslationRelativeToReference[1],
                'z': markerTranslationRelativeToReference[2],
                'roll': markerRotationRelativeToReference[0],
                'pitch': markerRotationRelativeToReference[1],
                'yaw': markerRotationRelativeToReference[2]
            }

            posesFound[str(markerId)] = markerPoseRelativeToReference

    return posesFound