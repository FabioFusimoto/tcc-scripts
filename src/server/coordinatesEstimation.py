import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, getRMatrixFromVector, calculateRelativePose
from src.server.objects import MARKER_DESCRIPTION

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
            coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=MARKER_DESCRIPTION[str(markerId)]['length'])
            poses[MARKER_DESCRIPTION[str(markerId)]['objectName']] = {'found': True,
                                                                      'pose':  coords}
        else:
            poses[MARKER_DESCRIPTION[str(markerId)]['objectName']] = {'found': False}

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
        hmdOffset = MARKER_DESCRIPTION[str(pivotMarkerId)]['offset']

        hmdPose = calculateRelativePose(pivotPose, RT, np.zeros((1,3)), np.array([[hmdOffset['x']],
                                                                                  [hmdOffset['y']],
                                                                                  [hmdOffset['z']]]),
                                        MARKER_DESCRIPTION[str(pivotMarkerId)]['length'], 1.0) # 1.0 because the offset is given in cm

        poses = {'hmd':          {'found': True,
                                  'pose':  hmdPose},
                 'marker_pivot': {'found': True,
                                  'pose':  {'roll':  0,
                                            'pitch': 0,
                                            'yaw':   0,
                                            'x':     0,
                                            'y':     0,
                                            'z':     0}}}

        for markerId in np.setdiff1d(ids, np.array([pivotMarkerId])): # check all ids, excluding the markerId
            markerIdIndexes = np.where(ids == markerId)[0]
            
            if markerIdIndexes.size > 0:
                j = markerIdIndexes[0]
                markerRVec = rVecs[j]
                markerTVec = tVecs[j]

                relativeMarkerPose = calculateRelativePose(pivotPose, RT, markerRVec, markerTVec, 
                                                           MARKER_DESCRIPTION[str(pivotMarkerId)]['length'], 
                                                           MARKER_DESCRIPTION[str(markerId)]['length'],
                                                           offset=MARKER_DESCRIPTION[str(markerId)].get('offset', None))

                poses[MARKER_DESCRIPTION[str(markerId)]['objectName']] = {'found': True,
                                                                          'pose':  relativeMarkerPose}
        
        return poses

    return {'hmd':          {'found': False},
            'marker_pivot': {'found': False}}