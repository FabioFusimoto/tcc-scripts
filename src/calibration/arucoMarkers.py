from timeit import default_timer as timer

import cv2.cv2 as cv2
import math
import numpy as np
import glob
import pprint

from src.calibration.commons import rotationMatrixToEulerAngles

# Aruco common parameters
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters_create()
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Rotation flip correction (to align camera and marker axis)
RFlip = np.zeros((3,3), dtype=np.float32)
RFlip[0,0] = 1.0
RFlip[1,1] = -1.0
RFlip[2,2] = -1.0

def generateMarkerGrid(nx, ny, outputFile):
    # Create gridboard, which is a set of Aruco markers
    # the following call gets a board of markers 5 wide X 7 tall
    gridboard = cv2.aruco.GridBoard_create(
            markersX=nx, 
            markersY=ny, 
            markerLength=0.04, 
            markerSeparation=0.01, 
            dictionary=ARUCO_DICT)
    image = gridboard.draw(outSize=(988,1400))
    cv2.imwrite(outputFile, image)

    cv2.imshow('Aruco grid', image)
    cv2.waitKey(0)

def getImageAndResize(sourceFile, scale):
    sourceImage = cv2.imread(sourceFile)
    return cv2.resize(sourceImage, None, fx=scale, fy=scale)

def getCorners(sourceImage):
    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedPoints = cv2.aruco.detectMarkers(grayImage, ARUCO_DICT, parameters=ARUCO_PARAMETERS)# aruco.detectMarkers(grayImage, ARUCO_DICT, ARUCO_PARAMETERS)
    return corners, ids, rejectedPoints

def highlightDetectedMarkers(sourceFile, outputFile, shouldSave, scale):
    sourceImage = getImageAndResize(sourceFile, scale)

    start = timer()
    corners, ids, _ = getCorners(sourceImage)

    cv2.aruco.drawDetectedMarkers(sourceImage, corners, ids)
    end = timer()
    
    print('Time elapsed to find markers: ' + str(end - start))
    cv2.imshow('detected markers', cv2.resize(sourceImage, None, fx=0.3, fy=0.3))
    cv2.waitKey(0)

    if shouldSave:
        cv2.imwrite(outputFile, sourceImage)

def estimatePose(sourceFile, outputFile, scale, markerId, markerLength, cameraMatrix, distCoeffs):
    sourceImage = getImageAndResize(sourceFile, scale)
    
    start = timer()
    
    corners, ids, _ = getCorners(sourceImage)
    rVecs, tVecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

    end = timer()

    print('Time elapsed to estimate pose: ' + '%3fms' % ((end - start) * 1000))

    if ids is not None:
        i = np.where(ids == markerId)[0][0]

        cv2.aruco.drawAxis(sourceImage, cameraMatrix, distCoeffs, rVecs[i], tVecs[i], markerLength/2)

        # Converting the rVector into Euler angles
        rMatrix = np.matrix(cv2.Rodrigues(rVecs[i])[0])
        roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix) # Flipping before converting

        traX, traY, traZ = tVecs[i][0]

        coordNames = ['Roll: ', 'Pitch: ', 'Yaw: ', 'Tra X: ', 'Tra Y: ', 'Tra Z: ']
        coords = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw), traX, traY, traZ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        initialPosition = (10,100)
        positionIncrement = 50
        scale = 1.5
        color = (0,0,0)
        thickness = 2

        for j in range(len(coords) + 3):
            position = (initialPosition[0], initialPosition[1] + j * positionIncrement)
            if j < len(coords):
                cv2.putText(sourceImage, coordNames[j] + '%.3f' % coords[j], position, font, scale, color, thickness, cv2.LINE_AA)
            elif(j == len(coords)):
                cv2.putText(sourceImage, '+X axis: Red', position, font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
            elif(j == (len(coords) + 1)):
                cv2.putText(sourceImage, '+Y axis: Green', position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
            elif(j == (len(coords) + 2)):
                cv2.putText(sourceImage, '+Z axis: Blue', position, font, scale, (255, 0, 0), thickness, cv2.LINE_AA)

        #cv2.imshow('Pose', cv2.resize(sourceImage, None, fx=0.3, fy=0.3))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        cv2.imwrite(outputFile, sourceImage)
    else:
        print('No markers were found')