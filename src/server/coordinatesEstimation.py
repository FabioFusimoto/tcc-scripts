import cv2.cv2 as cv2
import numpy as np

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates

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
            poses[markerId] = {'0_roll':  coords[0],
                               '1_pitch': coords[1],
                               '2_yaw':   coords[2],
                               '3_x':     coords[3],
                               '4_y':     coords[4],
                               '5_z':     coords[5]}
        else:
            poses[markerId] = {'0_roll':  None,
                               '1_pitch': None,
                               '2_yaw':   None,
                               '3_x':     None,
                               '4_y':     None,
                               '5_z':     None}

    return poses