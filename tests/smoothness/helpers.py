import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import getEulerAnglesFromRVector, inversePerspective

MARKER_LENGTH = 5.3

def getCameraPosition(cam, markerId, cameraMatrix, distCoeffs):
    image = cam.read()
    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    markerIndexes = np.where(ids == int(markerId))[0]

    if len(markerIndexes) < 1:
        return False, {}
    
    markerIndex = markerIndexes[0]

    markerRVec = rVecs[markerIndex]
    markerTVec = tVecs[markerIndex]

    invRVec, invTVec = inversePerspective(markerRVec, markerTVec, scale=MARKER_LENGTH)
    eulerAngles = getEulerAnglesFromRVector(invRVec)

    return True, {
        'x': invTVec.item((0,0)),
        'y': invTVec.item((1,0)),
        'z': invTVec.item((2,0)),
        'roll': eulerAngles[0],
        'pitch': eulerAngles[1],
        'yaw': eulerAngles[2]
    }

def plotPoints(X, Y, XLabel, YLabel):
    plt.plot(X, Y, 'ro')
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.show()

def plotMultiple(X, Y1, Y1Legend, Y2, Y2Legend, XLabel, YLabel):
    plt.plot(X, Y1, 'r--', X, Y2, 'b--')
    Y1Patch = mpatches.Patch(color='red', label=Y1Legend)
    Y2Patch = mpatches.Patch(color='blue', label=Y2Legend)
    plt.legend(handles=[Y1Patch, Y2Patch])
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.show()