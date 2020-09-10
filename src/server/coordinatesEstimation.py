import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, getRMatrixFromVector, getRMatrixFromEulerAngles, getRVectorFromEulerAngles, getEulerAnglesFromRVector, \
                                    calculateRelativePoseFromVectors, calculateRelativePoseFromPose, getTransformationMatrix, transformCoordinates
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
            coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=OBJECT_DESCRIPTION[str(markerId)]['length'])
            poses[OBJECT_DESCRIPTION[str(markerId)]['objectName']] = {'found': True,
                                                                      'pose':  coords}
        else:
            poses[OBJECT_DESCRIPTION[str(markerId)]['objectName']] = {'found': False}

    return poses

def estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam=None, camType=None, image=None):
    np.set_printoptions(precision=4, suppress=True)

    if image is None:
        image = cam.read()

        # Displaying the image
        # while True:
        #     cv2.imshow('Press Q to quit', image)
        #     if cv2.waitKey(5) & 0xFF == ord('q'):
        #         break

        while (camType == 'USB') and (cam.grabbed == False):
            image = cam.read()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    indexes = np.where(ids == pivotMarkerId)[0]

    if indexes.size > 0:
        i = indexes[0]

        pivotLength = OBJECT_DESCRIPTION[str(pivotMarkerId)]['length']

        pivotTVec = tVecs[i]
        pivotRVec = rVecs[i]

        pivotPose = calculateCoordinates(np.reshape(pivotRVec, (3,1)), np.reshape(pivotTVec, (3,1)), scale=pivotLength)

        RT = getRMatrixFromVector(pivotRVec).T
        hmdOffset = OBJECT_DESCRIPTION['hmd']['offset']
        hmdOffsetAsVec = np.array([[hmdOffset['x']],
                                   [hmdOffset['y']],
                                   [hmdOffset['z']]])

        hmdPose = calculateRelativePoseFromVectors(np.zeros((1,3)), hmdOffsetAsVec, pivotPose, RT) # 1.0 because the offset is given in cm

        poses = {'hmd':              hmdPose,
                 str(pivotMarkerId): {'roll':  0,
                                      'pitch': 0,
                                      'yaw':   0,
                                      'x':     0,
                                      'y':     0,
                                      'z':     0}}

        for markerId in markerIds:
            markerIdIndexes = np.where(ids == markerId)[0]
            
            if markerIdIndexes.size > 0:
                j = markerIdIndexes[0]
                markerRVec = rVecs[j]
                markerTVec = tVecs[j]

                markerPoseRelativeToCamera = calculateCoordinates(np.reshape(markerRVec, (3,1)), np.reshape(markerTVec, (3,1)), scale=pivotLength)

                relativeMarkerPose = calculateRelativePoseFromPose(markerPoseRelativeToCamera, pivotPose, RT, 
                                        objOffset=OBJECT_DESCRIPTION[str(markerId)].get('offset', None))

                poses[str(markerId)] = relativeMarkerPose

        return poses

    return {}

def getCoordinateTransformationMatrixes(targetPivotId, referenceId, cameraMatrix, distCoeffs, cam=None):
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

    # Pivot pose
    pivotRVec = rVecs[pivotIndex]
    pivotTVec = tVecs[pivotIndex]
    pivotLength = OBJECT_DESCRIPTION[str(targetPivotId)]['length']

    pivotPoseRelativeToCamera = calculateCoordinates(np.reshape(pivotRVec, (3,1)), np.reshape(pivotTVec, (3,1)), scale=pivotLength)

    print('\nPIVOT POSE RELATIVE TO CAMERA')
    pprint.pprint(pivotPoseRelativeToCamera)

    MCameraToPivot = getTransformationMatrix(pivotPoseRelativeToCamera)
    MCameraToPivotInverted = np.linalg.inv(MCameraToPivot)
    MCameraToPivotInverted[0, 3] = 0
    MCameraToPivotInverted[1, 3] = 0
    MCameraToPivotInverted[2, 3] = 0
    MPivotToCamera = MCameraToPivotInverted

    print('\nTRANSFORMATION MATRIX: PIVOT -> CAMERA')
    pprint.pprint(MPivotToCamera)

    # Reference pose relative to pivot coords
    referenceRVec = rVecs[referenceIndex]
    referenceTVec = tVecs[referenceIndex]
    referenceLength = OBJECT_DESCRIPTION[str(referenceId)]['length']

    referencePoseRelativeToCamera = calculateCoordinates(np.reshape(referenceRVec, (3,1)), np.reshape(referenceTVec, (3,1)), scale=referenceLength)

    referenceTVecRelativeToCamera = np.array([referencePoseRelativeToCamera['x'],
                                              referencePoseRelativeToCamera['y'],
                                              referencePoseRelativeToCamera['z'],
                                                                             1.0])

    referenceTranslationRelativeToPivot = transformCoordinates(referenceTVecRelativeToCamera, MCameraToPivot)

    referenceRotationRelativeToPivot = {'roll':  referencePoseRelativeToCamera['roll'] - pivotPoseRelativeToCamera['roll'],
                                        'pitch': referencePoseRelativeToCamera['pitch'] - pivotPoseRelativeToCamera['pitch'],
                                        'yaw':   referencePoseRelativeToCamera['yaw'] - pivotPoseRelativeToCamera['yaw']}

    referencePoseRelativeToPivot = {**referenceRotationRelativeToPivot, **referenceTranslationRelativeToPivot}

    # Tests
    referenceTVecRelativeToPivotOnPivotCoords = np.array([
        referencePoseRelativeToPivot['x'],
        referencePoseRelativeToPivot['y'],
        referencePoseRelativeToPivot['z'],
        1
    ])

    print('\nREFERENCE TRANSLATION RELATIVE TO PIVOT ON PIVOT COORDS')
    pprint.pprint(referenceTVecRelativeToPivotOnPivotCoords)

    referenceTranslationRelativeToPivotOnCameraCoords = transformCoordinates(referenceTVecRelativeToPivotOnPivotCoords, MPivotToCamera)

    print('\nREFERENCE TRANSLATION RELATIVE TO PIVOT ON CAMERA COORDS')
    pprint.pprint(referenceTranslationRelativeToPivotOnCameraCoords)

    referencePoseRelativeToCameraOnCameraCoords = {
        'x': pivotPoseRelativeToCamera['x'] + referenceTranslationRelativeToPivotOnCameraCoords['x'],
        'y': pivotPoseRelativeToCamera['y'] + referenceTranslationRelativeToPivotOnCameraCoords['y'],
        'z': pivotPoseRelativeToCamera['z'] + referenceTranslationRelativeToPivotOnCameraCoords['z'],
        'roll': pivotPoseRelativeToCamera['roll'] + referencePoseRelativeToPivot['roll'],
        'pitch': pivotPoseRelativeToCamera['pitch'] + referencePoseRelativeToPivot['pitch'],
        'yaw': pivotPoseRelativeToCamera['yaw'] + referencePoseRelativeToPivot['yaw']
    }

    # Assertions
    print('\nREFERENCE POSE RELATIVE TO CAMERA - REAL')
    pprint.pprint(referencePoseRelativeToCamera)

    print('\nREFERENCE POSE RELATIVE TO CAMERA - CALCULATED')
    pprint.pprint(referencePoseRelativeToCameraOnCameraCoords)

    return {'{}->{}-pose'.format(targetPivotId, referenceId): referencePoseRelativeToPivot}

def selectPivot(possiblePivotIds, foundMarkerIds):
    # print('\n\n\n>>>>MARKERS FOUND: {}<<<'.format(foundMarkerIds))
    # print('\n\n\n>>>>POSSIBLE PIVOTS: {}<<<'.format(possiblePivotIds))
    for pivotId in possiblePivotIds:
        if pivotId in foundMarkerIds:
            return pivotId
    
    return None

def estimatePosesFromMultiplePivots(markerIds, referencePosesRelativeToPivots, pivotPosesRelativeToReference, cameraMatrix, distCoeffs, cam=None, camType=None):
    # print('\n\n\n----------------------\n>>>>KNOWN PIVOT POSES<<<')
    # pprint.pprint(pivotPosesRelativeToReference)

    np.set_printoptions(precision=4, suppress=True)

    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    if ids is None:
        return {}

    targetPivotId = selectPivot(referencePosesRelativeToPivots.keys(), list(map(str, ids))) # Target is the one found on image

    if targetPivotId is None: # Meaning no pivot was found on image
        return {}
    
    print('\n\n\n>>>>TARGET PIVOT FOUND: {}'.format(targetPivotId))

    targetPivotLength = OBJECT_DESCRIPTION[targetPivotId]['length']

    indexes = np.where(ids == int(targetPivotId))[0]

    if indexes.size > 0:
        i = indexes[0]
        pivotTVecRelativeToCamera = tVecs[i]
        pivotRVecRelativeToCamera = rVecs[i]

        pivotPoseRelativeToCamera = calculateCoordinates(np.reshape(pivotRVecRelativeToCamera, (3,1)), np.reshape(pivotTVecRelativeToCamera, (3,1)), scale=targetPivotLength)

        print('\nPIVOT POSE RELATIVE TO CAMERA')
        pprint.pprint(pivotPoseRelativeToCamera)

        RTCameraToTargetPivot = getRMatrixFromVector(pivotRVecRelativeToCamera).T
        hmdOffset = OBJECT_DESCRIPTION['hmd']['offset']
        hmdOffsetAsVec = np.array([[hmdOffset['x']],
                                   [hmdOffset['y']],
                                   [hmdOffset['z']]])

        print('\n>>>HMD POSE RELATIVE TO TARGET PIVOT')

        hmdPoseRelativeToTargetPivot = calculateRelativePoseFromVectors(np.zeros((1,3)), hmdOffsetAsVec,
                                        pivotPoseRelativeToCamera, RTCameraToTargetPivot) # 1.0 because the offset is given in cm

        print('')
        pprint.pprint(hmdPoseRelativeToTargetPivot)

        referencePoseRelativeToTargetPivot = referencePosesRelativeToPivots[targetPivotId]
        targetPivotPoseRelativeToReference = pivotPosesRelativeToReference[targetPivotId]

        print('\nReference pose relative to target pivot')
        pprint.pprint(referencePoseRelativeToTargetPivot)

        RTTargetPivotToReference = getRMatrixFromEulerAngles(referencePoseRelativeToTargetPivot['roll'],
                                    referencePoseRelativeToTargetPivot['pitch'],
                                    referencePoseRelativeToTargetPivot['yaw']).T

        print('\nRT target pivot to reference:')
        pprint.pprint(RTTargetPivotToReference)

        # hmdRVecRelativeToTargetPivot = getRVectorFromEulerAngles(hmdPoseRelativeToTargetPivot['roll'],
        #                                                          hmdPoseRelativeToTargetPivot['pitch'],
        #                                                          hmdPoseRelativeToTargetPivot['yaw'])

        # print('\nHMD Rvec relative to target:')
        # pprint.pprint(hmdRVecRelativeToTargetPivot)

        # hmdEulerAnglesRelativeToTargetPivot = getEulerAnglesFromRVector(hmdRVecRelativeToTargetPivot)
        # print('\nHMD Euler angles relative to target:')
        # pprint.pprint(hmdEulerAnglesRelativeToTargetPivot)

        # hmdTVecRelativeToTargetPivot = np.array([[hmdPoseRelativeToTargetPivot['x']],
        #                                          [hmdPoseRelativeToTargetPivot['y']],
        #                                          [hmdPoseRelativeToTargetPivot['z']]])

        # print('\nHMD Tvec relative to target:')
        # pprint.pprint(hmdTVecRelativeToTargetPivot)

        print('\nHMD OFFSET RELATIVE TO REFERENCE PIVOT IN REFERENCE COORDINATES')

        hmdOffsetRelativeToReferencePivotInReferenceCoordinates = calculateRelativePoseFromPose(hmdPoseRelativeToTargetPivot, referencePoseRelativeToTargetPivot,
                                                                    RTTargetPivotToReference) # 1.0 because the scale has been previously applied

        hmdPoseRelativeToReferencePivot = {'x': hmdOffsetRelativeToReferencePivotInReferenceCoordinates['x'] - targetPivotPoseRelativeToReference['x'],
                                           'y': hmdOffsetRelativeToReferencePivotInReferenceCoordinates['y'] - targetPivotPoseRelativeToReference['y'],
                                           'z': hmdOffsetRelativeToReferencePivotInReferenceCoordinates['z'] - targetPivotPoseRelativeToReference['z'],
                                           'roll':  hmdOffsetRelativeToReferencePivotInReferenceCoordinates['roll'],
                                           'pitch': hmdOffsetRelativeToReferencePivotInReferenceCoordinates['pitch'],
                                           'yaw':   hmdOffsetRelativeToReferencePivotInReferenceCoordinates['yaw']}

        print('\nHMD POSE RELATIVE TO REFERENCE PIVOT')
        pprint.pprint(hmdPoseRelativeToReferencePivot)

        # difference = {}
        # for k in hmdPoseRelativeToReferencePivot.keys():
        #     difference[k] = abs(hmdPoseRelativeToReferencePivot[k] - hmdPoseRelativeToTargetPivot[k])
        # print('\n\n\nDIFFERENCE:')
        # pprint.pprint(difference)

        poses = {'hmd': hmdPoseRelativeToReferencePivot}

        # for markerId in markerIds:
        #     markerIdIndexes = np.where(ids == markerId)[0]
            
        #     if markerIdIndexes.size > 0:
        #         j = markerIdIndexes[0]
        #         markerRVecRelativeToCamera = rVecs[j]
        #         markerTVecRelativeToCamera = tVecs[j]

        #         markerPoseRelativeToTargetPivot = calculateRelativePoseFromVectors(pivotPoseRelativeToCamera, RTTargetPivotRelativeToCamera, 
        #                                                                 markerRVecRelativeToCamera, markerTVecRelativeToCamera, 
        #                                                                 OBJECT_DESCRIPTION[str(targetPivotId)]['length'],
        #                                                                 OBJECT_DESCRIPTION[str(markerId)]['length'],
        #                                                                 offset=OBJECT_DESCRIPTION[str(markerId)].get('offset', None))

        #         markerRVecRelativeToTargetPivot = getRVectorFromEulerAngles(markerPoseRelativeToTargetPivot['roll'],
        #                                                                     markerPoseRelativeToTargetPivot['pitch'],
        #                                                                     markerPoseRelativeToTargetPivot['yaw'])

        #         markerTVecRelativeToTargetPivot = np.array([[markerPoseRelativeToTargetPivot['x']],
        #                                                     [markerPoseRelativeToTargetPivot['y']],
        #                                                     [markerPoseRelativeToTargetPivot['z']]])

        #         markerPoseRelativeToReferencePivot = calculateRelativePoseFromVectors(referencePoseRelativeToTargetPivot, RTTargetPivotRelativeToReference,
        #                                                                    markerRVecRelativeToTargetPivot, markerTVecRelativeToTargetPivot, 1.0, 1.0)

        #         poses[str(markerId)] = markerPoseRelativeToReferencePivot
        
        # print('\nPOSES RELATIVE TO REFERENCE')
        # pprint.pprint(poses)
        return poses

    return {}

def estimatePosesFromMultiplePivots2(markerIds, pivotIds, referenceId, transformationParameters, cameraMatrix, distCoeffs, cam=None, camType='USB'):
    np.set_printoptions(precision=4, suppress=True)

    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    if ids is None:
        return {}

    # targetPivotId = selectPivot(pivotIds, list(map(str, ids))) # Target is the one found on image
    targetPivotId = '7'

    if targetPivotId is None: # Meaning no pivot was found on image
        return {}
    
    print('\n\n\n>>>>TARGET PIVOT FOUND: {}'.format(targetPivotId))

    targetPivotLength = OBJECT_DESCRIPTION[str(targetPivotId)]['length']

    targetPivotIndexes = np.where(ids == int(targetPivotId))[0]

    if targetPivotIndexes.size == 0:
        return {}

    targetPivotIndex = targetPivotIndexes[0]

    pivotRVecRelativeToCamera = rVecs[targetPivotIndex]
    pivotTVecRelativeToCamera = tVecs[targetPivotIndex]

    pivotPoseRelativeToCamera = calculateCoordinates(np.reshape(pivotRVecRelativeToCamera, (3,1)), np.reshape(pivotTVecRelativeToCamera, (3,1)), scale=targetPivotLength)

    MCameraToPivot = getTransformationMatrix(pivotPoseRelativeToCamera)
    
    print('\nCAMERA -> PIVOT TRANSFORMATION MATRIX')
    pprint.pprint(MCameraToPivot)

    MCameraToPivotInverted = np.linalg.inv(MCameraToPivot)
    MCameraToPivotInverted[0, 3] = 0
    MCameraToPivotInverted[1, 3] = 0
    MCameraToPivotInverted[2, 3] = 0
    MPivotToCamera = MCameraToPivotInverted

    print('\nTRANSFORMATION MATRIX: PIVOT -> CAMERA')
    pprint.pprint(MPivotToCamera)

    referencePoseRelativeToPivot = transformationParameters['{}->{}-pose'.format(targetPivotId, referenceId)]
    referenceTVecRelativeToPivotOnPivotCoordinates = np.array([
        referencePoseRelativeToPivot['x'],
        referencePoseRelativeToPivot['y'],
        referencePoseRelativeToPivot['z'],
        1
    ])

    referenceTranslationRelativeToPivotOnCameraCoords = transformCoordinates(referenceTVecRelativeToPivotOnPivotCoordinates, MPivotToCamera)

    print('\nREFERENCE TRANSLATION RELATIVE TO PIVOT ON CAMERA COORDS')
    pprint.pprint(referenceTranslationRelativeToPivotOnCameraCoords)

    referencePoseRelativeToCameraOnCameraCoords = {
        'x': pivotPoseRelativeToCamera['x'] + referenceTranslationRelativeToPivotOnCameraCoords['x'],
        'y': pivotPoseRelativeToCamera['y'] + referenceTranslationRelativeToPivotOnCameraCoords['y'],
        'z': pivotPoseRelativeToCamera['z'] + referenceTranslationRelativeToPivotOnCameraCoords['z'],
        'roll': pivotPoseRelativeToCamera['roll'] + referencePoseRelativeToPivot['roll'],
        'pitch': pivotPoseRelativeToCamera['pitch'] + referencePoseRelativeToPivot['pitch'],
        'yaw': pivotPoseRelativeToCamera['yaw'] + referencePoseRelativeToPivot['yaw']
    }

    MCameraToReference = getTransformationMatrix(referencePoseRelativeToCameraOnCameraCoords)

    cameraTranslationRelativeToReference = transformCoordinates(np.array([0, 0, 0, 1]), MCameraToReference)

    cameraRotationRelativeToReference = {
        'roll': -referencePoseRelativeToCameraOnCameraCoords['roll'],
        'pitch': -referencePoseRelativeToCameraOnCameraCoords['pitch'],
        'yaw': -referencePoseRelativeToCameraOnCameraCoords['yaw']
    }

    cameraPoseRelativeToReference = {**cameraTranslationRelativeToReference, **cameraRotationRelativeToReference}

    print('\nCAMERA POSE RELATIVE TO REFERENCE\n')
    pprint.pprint(cameraPoseRelativeToReference)

    posesFound = {'hmd': cameraPoseRelativeToReference}

    return posesFound