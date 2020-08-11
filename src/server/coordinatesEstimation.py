import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, calculateCameraCoordinates, getRMatrixFromVector, calculateRelativePose
from src.server.objects import MARKER_TO_OBJECT

markerRFlip = arucoMarkers.getMarkerRFlip()
pivotRFlip = arucoMarkers.getRFlip()

def estimatePoses(markerIds, markerLength, cameraMatrix, distCoeffs, cam, camType):
    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, markerLength, cameraMatrix, distCoeffs)

    poses = {}
    for markerId in markerIds:
        indexes = np.where(ids == markerId)[0]
        if indexes.size > 0:
            i = indexes[0]
            coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)))
            poses[MARKER_TO_OBJECT[str(markerId)]] = {'found': True,
                                                      'pose':  {'roll':  coords[0],
                                                                'pitch': coords[1],
                                                                'yaw':   coords[2],
                                                                'x':     coords[3],
                                                                'y':     coords[4],
                                                                'z':     coords[5]}}
        else:
            poses[MARKER_TO_OBJECT[str(markerId)]] = {'found': False}

    return poses

def estimatePosesFromPivot(markerIds, pivotMarkerId, markerLength, cameraMatrix, distCoeffs, cam=None, camType=None, image=None):
    if image is None:
        image = cam.read()

        while (camType == 'USB') and (cam.grabbed == False):
            image = cam.read()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, markerLength, cameraMatrix, distCoeffs)

    indexes = np.where(ids == pivotMarkerId)[0]

    if indexes.size > 0:
        i = indexes[0]
        pivotTVec = tVecs[i]
        pivotRVec = rVecs[i]

        pivotPose = calculateCoordinates(np.reshape(pivotRVec, (3,1)), np.reshape(pivotTVec, (3,1)))

        RT = getRMatrixFromVector(pivotRVec).T

        hmdPose = calculateRelativePose(pivotPose, RT, np.zeros((1,3)), np.zeros((1,3)))

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

                relativeMarkerPose = calculateRelativePose(pivotPose, RT, markerRVec, markerTVec)

                poses[MARKER_TO_OBJECT[str(markerId)]] = {'found': True,
                                                          'pose':  relativeMarkerPose}
        
        return poses

    return {'hmd':          {'found': False},
            'marker_pivot': {'found': False}}