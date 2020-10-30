import cv2.cv2 as cv2
import math
import numpy as np

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates
from src.server.objects import OBJECT_DESCRIPTION

# Font parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
INITIAL_POSITION = (10,100)
POSITION_INCREMENT = 50
SCALE = 1.0
COLOR = (0,128,255)
THICKNESS = 2

def livePoseEstimation(markerId, cameraMatrix, distCoeffs, cam, camType):
    coords = {}

    while True:
        image = cam.read()
        
        while (camType == 'USB') and (cam.grabbed == False):
            image = cam.read()
        
        corners, idsForCorners, _ = arucoMarkers.getCorners(image)
        ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

        indexes = np.where(ids == markerId)[0]

        if indexes.size > 0:
            i = indexes[0]
            coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=OBJECT_DESCRIPTION[str(markerId)]['length'])

            # Converting radians to degrees
            for c in ['roll', 'pitch', 'yaw']:
                coords[c] = math.degrees(coords[c])

            # Highlighting the found marker
            cv2.aruco.drawDetectedMarkers(image, corners, idsForCorners)

            # Drawing the marker's axis
            cv2.aruco.drawAxis(image, cameraMatrix, distCoeffs, rVecs[i], tVecs[i], 0.5)

            # Writing coordinates to image
            cv2.putText(image, 'x: ' + '%.2f' % coords['x'], INITIAL_POSITION, FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, 'y: ' + '%.2f' % coords['y'], (INITIAL_POSITION[0], INITIAL_POSITION[1] + 1 * POSITION_INCREMENT), FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, 'z: ' + '%.2f' % coords['z'], (INITIAL_POSITION[0], INITIAL_POSITION[1] + 2 * POSITION_INCREMENT), FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, 'roll: ' + '%.2f' % coords['roll'], (INITIAL_POSITION[0], INITIAL_POSITION[1] + 3 * POSITION_INCREMENT), FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, 'pitch: ' + '%.2f' % coords['pitch'], (INITIAL_POSITION[0], INITIAL_POSITION[1] + 4 * POSITION_INCREMENT), FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, 'yaw: ' + '%.2f' % coords['yaw'], (INITIAL_POSITION[0], INITIAL_POSITION[1] + 5 * POSITION_INCREMENT), FONT, SCALE, COLOR, THICKNESS, cv2.LINE_AA)

            # Displaying the image
            cv2.imshow('Press Q to quit', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            print('Marker not found')
    
    cv2.destroyAllWindows()
    return coords