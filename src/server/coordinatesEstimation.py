import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, getRMatrixFromVector, getRMatrixFromEulerAngles, getRVectorFromEulerAngles, getEulerAnglesFromRVector, calculateRelativePose
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
    if image is None:
        image = cam.read()

        # # Displaying the image
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
        pivotTVec = tVecs[i]
        pivotRVec = rVecs[i]

        pivotPose = calculateCoordinates(np.reshape(pivotRVec, (3,1)), np.reshape(pivotTVec, (3,1)))

        RT = getRMatrixFromVector(pivotRVec).T
        hmdOffset = OBJECT_DESCRIPTION['hmd']['offset']

        hmdPose = calculateRelativePose(pivotPose, RT, np.zeros((1,3)), np.array([[hmdOffset['x']],
                                                                                  [hmdOffset['y']],
                                                                                  [hmdOffset['z']]]),
                                        OBJECT_DESCRIPTION[str(pivotMarkerId)]['length'], 1.0) # 1.0 because the offset is given in cm

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

                relativeMarkerPose = calculateRelativePose(pivotPose, RT, markerRVec, markerTVec, 
                                                           OBJECT_DESCRIPTION[str(pivotMarkerId)]['length'], 
                                                           OBJECT_DESCRIPTION[str(markerId)]['length'],
                                                           offset=OBJECT_DESCRIPTION[str(markerId)].get('offset', None))

                poses[str(markerId)] = relativeMarkerPose
        
        return poses

    return {}

def selectPivot(possiblePivotIds, foundMarkerIds):
    # print('\n\n\n>>>>MARKERS FOUND: {}<<<'.format(foundMarkerIds))
    # print('\n\n\n>>>>POSSIBLE PIVOTS: {}<<<'.format(possiblePivotIds))
    for pivotId in possiblePivotIds:
        if pivotId in foundMarkerIds:
            return pivotId
    
    return None

def estimatePosesFromMultiplePivots(markerIds, referencePosesRelativeToPivots, cameraMatrix, distCoeffs, cam=None, camType=None):
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

    indexes = np.where(ids == int(targetPivotId))[0]

    if indexes.size > 0:
        i = indexes[0]
        pivotTVecRelativeToCamera = tVecs[i]
        pivotRVecRelativeToCamera = rVecs[i]

        pivotPoseRelativeToCamera = calculateCoordinates(np.reshape(pivotRVecRelativeToCamera, (3,1)), np.reshape(pivotTVecRelativeToCamera, (3,1)))

        RTTargetPivotRelativeToCamera = getRMatrixFromVector(pivotRVecRelativeToCamera).T
        hmdOffset = OBJECT_DESCRIPTION['hmd']['offset']

        print('\n>>>HMD POSE RELATIVE TO TARGET PIVOT {}'.format(targetPivotId))

        hmdPoseRelativeToTargetPivot = calculateRelativePose(pivotPoseRelativeToCamera, RTTargetPivotRelativeToCamera, np.zeros((1,3)), 
                                                             np.array([[hmdOffset['x']],
                                                                       [hmdOffset['y']],
                                                                       [hmdOffset['z']]]),
                                                             OBJECT_DESCRIPTION[str(targetPivotId)]['length'], 1.0) # 1.0 because the offset is given in cm

        print('')
        pprint.pprint(hmdPoseRelativeToTargetPivot)

        referencePoseRelativeToTargetPivot = referencePosesRelativeToPivots[targetPivotId]
        RTTargetPivotRelativeToReference = getRMatrixFromEulerAngles(referencePoseRelativeToTargetPivot['roll'],
                                                                     referencePoseRelativeToTargetPivot['pitch'],
                                                                     referencePoseRelativeToTargetPivot['yaw']).T

        print('\n>>>>RT target pivot relative to reference:')
        pprint.pprint(RTTargetPivotRelativeToReference)

        hmdRVecRelativeToTargetPivot = getRVectorFromEulerAngles(hmdPoseRelativeToTargetPivot['roll'],
                                                                 hmdPoseRelativeToTargetPivot['pitch'],
                                                                 hmdPoseRelativeToTargetPivot['yaw'])

        print('\n>>>HMD Rvec relative to target:')
        pprint.pprint(hmdRVecRelativeToTargetPivot)

        hmdEulerAnglesRelativeToTargetPivot = getEulerAnglesFromRVector(hmdRVecRelativeToTargetPivot)
        print('\n>>>HMD Euler angles relative to target:')
        pprint.pprint(hmdEulerAnglesRelativeToTargetPivot)

        hmdTVecRelativeToTargetPivot = np.array([[hmdPoseRelativeToTargetPivot['x']],
                                                 [hmdPoseRelativeToTargetPivot['y']],
                                                 [hmdPoseRelativeToTargetPivot['z']]])

        print('\n>>>HMD Tvec relative to target:')
        pprint.pprint(hmdTVecRelativeToTargetPivot)

        print('\n>>>HMD POSE RELATIVE TO REFERENCE PIVOT')

        hmdPoseRelativeToReferencePivot = calculateRelativePose(referencePoseRelativeToTargetPivot, RTTargetPivotRelativeToReference,
                                                                hmdRVecRelativeToTargetPivot, hmdTVecRelativeToTargetPivot,
                                                                1.0, 1.0) # 1.0 because the scale has been previously applied

        if str(targetPivotId) != '3':
            for k in ['x', 'y', 'z']:
                hmdPoseRelativeToReferencePivot[k] *= -1

        print('')
        pprint.pprint(hmdPoseRelativeToReferencePivot)

        # difference = {}
        # for k in hmdPoseRelativeToReferencePivot.keys():
        #     difference[k] = abs(hmdPoseRelativeToReferencePivot[k] - hmdPoseRelativeToTargetPivot[k])
        # print('\n\n\n>>>>DIFFERENCE:')
        # pprint.pprint(difference)

        poses = {'hmd': hmdPoseRelativeToReferencePivot}

        # for markerId in markerIds:
        #     markerIdIndexes = np.where(ids == markerId)[0]
            
        #     if markerIdIndexes.size > 0:
        #         j = markerIdIndexes[0]
        #         markerRVec = rVecs[j]
        #         markerTVec = tVecs[j]

        #         print('\n>>>MARKER {} POSE RELATIVE TO REFERENCE PIVOT'.format(markerId))
        #         markerPoseRelativeToTargetPivot = calculateRelativePose(pivotPoseRelativeToCamera, RTTargetPivotRelativeToReference, markerRVec, markerTVec, 
        #                                                                 OBJECT_DESCRIPTION[str(targetPivotId)]['length'], 
        #                                                                 OBJECT_DESCRIPTION[str(markerId)]['length'],
        #                                                                 offset=OBJECT_DESCRIPTION[str(markerId)].get('offset', None))

        #         markerPoseRelativeToReferencePivot = markerPoseRelativeToTargetPivot

        #         poses[str(markerId)] = markerPoseRelativeToReferencePivot
        
        return poses

    return {}