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
        return np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    
    markerIndex = markerIndexes[0]

    markerRVec = rVecs[markerIndex]
    markerTVec = tVecs[markerIndex]

    invRVec, invTVec = inversePerspective(markerRVec, markerTVec, scale=MARKER_LENGTH)
    eulerAngles = getEulerAnglesFromRVector(invRVec)

    return np.array([[
        invTVec.item((0,0)), # x
        invTVec.item((1,0)), # y
        invTVec.item((2,0)), # z
        eulerAngles[0],      # roll
        eulerAngles[1],      # pitch
        eulerAngles[2]       # yaw
    ]], dtype=np.float32)    

def plotPoints(X, Y, XLabel, YLabel):
    plt.plot(X, Y, 'ro')
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.show()

def plotMultiple(X, Y1, Y1Legend, Y2, Y2Legend, XLabel, YLabel):
    plt.plot(X, Y1, 'r--', X, Y2, 'b--', markersize=3)
    Y1Patch = mpatches.Patch(color='red', label=Y1Legend)
    Y2Patch = mpatches.Patch(color='blue', label=Y2Legend)
    plt.legend(handles=[Y1Patch, Y2Patch])
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.show()