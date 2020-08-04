import cv2.cv2 as cv2
import numpy as np

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates
from src.server.objects import MARKER_TO_OBJECT

def estimatePoses(markerIds, markerLength, cameraMatrix, distCoeffs, cam, camType):
    image = cam.read()

    while (camType == 'USB') and (cam.grabbed == False):
        image = cam.read()

    #cv2.imshow('Frame', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, markerLength, cameraMatrix, distCoeffs)

    poses = {}
    for markerId in markerIds:
        indexes = np.where(ids == markerId)[0]
        if indexes.size > 0:
            i = indexes[0]
            coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), arucoMarkers.getRFlip())
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