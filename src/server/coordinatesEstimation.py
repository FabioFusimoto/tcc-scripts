import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, getRMatrixFromVector, calculateRelativePose
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

        poses = {'hmd':              {'found': True,
                                      'pose':  hmdPose},
                 str(pivotMarkerId): {'found': True,
                                      'pose':  {'roll':  0,
                                                'pitch': 0,
                                                'yaw':   0,
                                                'x':     0,
                                                'y':     0,
                                                'z':     0}}}

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

                poses[str(markerId)] = {'found': True,
                                        'pose':  relativeMarkerPose}
        
        return poses

    return {}

def selectPivot(possiblePivotIds, foundMarkerIds):
    # print('\n\n\n>>>>MARKERS FOUND: {}<<<'.format(foundMarkerIds))
    # print('\n\n\n>>>>POSSIBLE PIVOTS: {}<<<'.format(possiblePivotIds))
    for pivotId in possiblePivotIds:
        if pivotId in foundMarkerIds:
            return pivotId
    
    return None

def estimatePosesFromMultiplePivots(markerIds, pivotPosesRelativeToReference, cameraMatrix, distCoeffs, cam=None, camType=None):
    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    if ids is None:
        return {}

    targetPivotId = selectPivot(pivotPosesRelativeToReference.keys(), list(map(str, ids)))

    if targetPivotId is None: # Meaning no pivot was found on image
        return {}
    
    # print('\n\n\n>>>>PIVOT FOUND: {}<<<'.format(targetPivotId))

    indexes = np.where(ids == int(targetPivotId))[0]

    if indexes.size > 0:
        i = indexes[0]
        pivotTVecRelativeToCamera = tVecs[i]
        pivotRVecRelativeToCamera = rVecs[i]

        pivotPoseRelativeToCamera = calculateCoordinates(np.reshape(pivotRVecRelativeToCamera, (3,1)), np.reshape(pivotTVecRelativeToCamera, (3,1)))

        RTCameraRelativeToTargetPivot = getRMatrixFromVector(pivotRVecRelativeToCamera).T
        hmdOffset = OBJECT_DESCRIPTION['hmd']['offset']

        hmdPoseRelativeToTargetPivot = calculateRelativePose(pivotPoseRelativeToCamera, RTCameraRelativeToTargetPivot, np.zeros((1,3)), 
                                                             np.array([[hmdOffset['x']],
                                                                       [hmdOffset['y']],
                                                                       [hmdOffset['z']]]),
                                                             OBJECT_DESCRIPTION[str(targetPivotId)]['length'], 1.0) # 1.0 because the offset is given in cm

        poses = {'hmd':              {'found': True,
                                      'pose':  hmdPoseRelativeToTargetPivot},
                 str(targetPivotId): {'found': True,
                                      'pose':  pivotPosesRelativeToReference[targetPivotId]}}

        for markerId in markerIds:
            markerIdIndexes = np.where(ids == markerId)[0]
            
            if markerIdIndexes.size > 0:
                j = markerIdIndexes[0]
                markerRVec = rVecs[j]
                markerTVec = tVecs[j]

                markerPoseRelativeToTargetPivot = calculateRelativePose(pivotPoseRelativeToCamera, RTCameraRelativeToTargetPivot, markerRVec, markerTVec, 
                                                                        OBJECT_DESCRIPTION[str(targetPivotId)]['length'], 
                                                                        OBJECT_DESCRIPTION[str(markerId)]['length'],
                                                                        offset=OBJECT_DESCRIPTION[str(markerId)].get('offset', None))

                poses[str(markerId)] = {'found': True,
                                        'pose':  markerPoseRelativeToTargetPivot}
        
        return poses

    return {}