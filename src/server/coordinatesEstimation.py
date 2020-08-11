import cv2.cv2 as cv2
import numpy as np
import pprint

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates, calculateCameraCoordinates, getRMatrixFromVector, rotationMatrixToEulerAngles
from src.server.objects import MARKER_TO_OBJECT

markerRFlip = arucoMarkers.getMarkerRFlip()
pivotRFlip = arucoMarkers.getRFlip()

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
    np.set_printoptions(precision=4, suppress=True)

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
        
        print('---------------------\nPivot pose (from camera\'s perspective)')
        pprint.pprint(pivotPose)
        print('---------------------')

        # poses = {'hmd':          {'found': True,
        #                           'pose':  {'roll':  cameraPose['roll'],
        #                                     'pitch': cameraPose['pitch'],
        #                                     'yaw':   cameraPose['yaw'],
        #                                     'x':     cameraPose['x'],
        #                                     'y':     cameraPose['y'],
        #                                     'z':     cameraPose['z']}},
        #          'marker_pivot': {'found': True,
        #                           'pose':  {'roll':  0,
        #                                     'pitch': 0,
        #                                     'yaw':   0,
        #                                     'x':     0,
        #                                     'y':     0,
        #                                     'z':     0}}}

        RT = getRMatrixFromVector(pivotRVec).T

        # TMatrix = np.array([[  RT.item(0,0),   RT.item(1,0),   RT.item(2,0), 0], 
        #                     [  RT.item(0,1),   RT.item(1,1),   RT.item(2,1), 0], 
        #                     [  RT.item(0,2),   RT.item(1,2),   RT.item(2,2), 0],
        #                     [pivotPose['x'], pivotPose['y'], pivotPose['z'], 1]])

        # print('---------------------\nTransformation matrix')
        # print(TMatrix)
        # print('---------------------')

        for markerId in np.setdiff1d(ids, np.array([pivotMarkerId])): # check all ids, excluding the markerId one
            markerIdIndexes = np.where(ids == markerId)[0]
            
            if markerIdIndexes.size > 0:
                j = markerIdIndexes[0]
                markerRVec = rVecs[j]
                markerTVec = tVecs[j]

                markerPose = calculateCoordinates(np.reshape(markerRVec, (3,1)), np.reshape(markerTVec, (3,1))) # from camera's perspective

                print('\nMarker ID = {}'.format(markerId))

                print('\nTranslation related to pivot position')
                firstElement = np.dot(RT, np.array([[markerPose['x']],
                                                    [markerPose['y']],
                                                    [markerPose['z']]]))
                secondElement = np.dot(RT, np.array([[pivotPose['x']],
                                                     [pivotPose['y']],
                                                     [pivotPose['z']]]))

                relativeTranslation = np.subtract(firstElement, secondElement)
                print(relativeTranslation)

                print('\nRotation related to pivot position')
                markerRoll, markerPitch, markerYaw = rotationMatrixToEulerAngles(getRMatrixFromVector(markerRVec))
                print('Roll:  {:5.2f}'.format(180/np.pi * (markerRoll - pivotPose['roll'])))
                print('Pitch: {:5.2f}'.format(180/np.pi * (markerPitch - pivotPose['pitch'])))
                print('Yaw:   {:5.2f}'.format(180/np.pi * (markerYaw - pivotPose['yaw'])))

                # markerPose = calculateCameraCoordinates(np.reshape(relativeRotation, (3,1)), np.reshape(relativeTranslation, (3,1)))

                # poses[MARKER_TO_OBJECT[str(markerId)]] = {'found': True,
                #                                           'pose':  {'roll':  markerPose['roll'],
                #                                                     'pitch': markerPose['pitch'],
                #                                                     'yaw':   markerPose['yaw'],
                #                                                     'x':     markerPose['x'],
                #                                                     'y':     markerPose['y'],
                #                                                     'z':     markerPose['z']}}
        
        print('\n\n\n')
        # return poses

    return {'hmd':          {'found': False},
            'marker_pivot': {'found': False}}