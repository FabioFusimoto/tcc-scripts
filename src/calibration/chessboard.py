from timeit import default_timer as timer

import cv2.cv2 as cv2
import math
import numpy as np
import glob
import pprint

from src.calibration.commons import rotationMatrixToEulerAngles, getImageAndResize, calculateCoordinates 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Rotation flip correction (to align camera and board axis)
RFlip = np.zeros((3,3), dtype=np.float32)
RFlip[0,0] = -1.0
RFlip[1,1] = -1.0
RFlip[2,2] = -1.0

def getCorners(sourceImage, width, height):
    flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCorners(grayImage, (width, height), flags)

def calibrate(sourcePath, outputPath, prefix, imageFormat, squareSize, width, height, scale):
    """Calibrates camera parameters for chessboard images on the given path"""

    # Creating a matrix of points to map
    objectPoints = np.zeros((height*width, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Using the actual size to convert the coordinates to metric
    objectPoints = objectPoints * squareSize

    # Arrays to store 3D points from the space and 2D points from all images
    pointsInSpace = []
    pointsInPlane = []

    # Fetching all calibration photos which match the given path, prefix and format
    photoFiles = glob.glob(sourcePath + '/' + prefix + '*.' + imageFormat)

    chessboardCornersFound = 0
    for f in photoFiles:
        start = timer()

        photo = getImageAndResize(f, scale)
        grayPhoto = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        print('\nReading image ' + str(f).split('\\')[1])

        # Find the chessboards corners
        patternWasFound, corners = getCorners(photo, width, height)

        # If the corners were found, add them to 3D and 2D points arrays
        if patternWasFound:
            chessboardCornersFound += 1
            pointsInSpace.append(objectPoints)

            refinedCorners = cv2.cornerSubPix(grayPhoto, corners, (11, 11), (-1, -1), criteria)
            pointsInPlane.append(refinedCorners)

            # Save the photos with the corners drawn, if found
            cv2.drawChessboardCorners(photo, (width, height), refinedCorners, patternWasFound)
            scaleAsString = str(scale * 100).split('.')[0]
            outputFilename = outputPath + '/' + str(f).split('\\')[1].split('.')[0] + '-corners-' + scaleAsString + '.' + imageFormat
            cv2.imwrite(outputFilename, photo)

        end = timer()
        print('Time elapsed on processsing: ' + str(end - start) + 's')
    
    print('Chessboard pattern found in ' + str(chessboardCornersFound) + ' out of ' + str(len(photoFiles)) + ' photos')

    # Calibrating the camera using the planar and spacial points found
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(pointsInSpace, pointsInPlane, grayPhoto.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def undistortImage(sourceImage, outputFile, cameraMatrix, distortionCoeffs, scale=1.0):
    h, w = sourceImage.shape[:2] # scaled image resolution - height x width
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs, (w, h), 1, (w, h))

    # Correcting distortions and cropping the result
    undistortedImage = cv2.undistort(sourceImage, cameraMatrix, distortionCoeffs, None, newCameraMatrix)
    x, y, w, h = roi
    undistortedImage = undistortedImage[y:y+h, x:x+w]
    cv2.imwrite(outputFile, undistortedImage)

def drawAxis(image, corners, imagePoints):
    corner = tuple(corners[0].ravel())

    image = cv2.line(image, corner, tuple(imagePoints[0].ravel()), (255,0,0), 5)
    image = cv2.line(image, corner, tuple(imagePoints[1].ravel()), (0,255,0), 5)
    image = cv2.line(image, corner, tuple(imagePoints[2].ravel()), (0,0,255), 5)
    
    return image

def drawOnChessboard(sourceImage, outputFile, axisLength, cameraMatrix, distortionCoeffs, rotVectors, traVectors, corners, width=9, height=6, scale=1.0):
    objectPoints = np.zeros((height*width, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    axis = np.array([[axisLength,0,0], [0,axisLength,0], [0,0,-axisLength]], dtype='f').reshape(-1, 3)

    # Project 3D points to the image plane
    imagePoints, _ = cv2.projectPoints(axis, rotVectors, traVectors, cameraMatrix, distortionCoeffs)
    imageWithAxis = drawAxis(sourceImage, corners, imagePoints)
  
    return imageWithAxis

def getPositionVectors(image, corners, cameraMatrix, distortionCoeffs, width=9, height=6):
    objectPoints = np.zeros((height*width, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    refinedCorners = cv2.cornerSubPix(grayImage, corners, (11,11), (-1, -1), criteria)

    # Find the rotation and translation vectors
    _, rotVectors, traVectors, error = cv2.solvePnPRansac(objectPoints, refinedCorners, cameraMatrix, distortionCoeffs)

    return rotVectors , traVectors, error

def drawPositionVectors(sourceFile, outputFile, cameraMatrix, distortionCoeffs, squareSize=1, width=9, height=6, scale=1.0):
    sourceImage = getImageAndResize(sourceFile, scale)
    found, corners = getCorners(sourceImage, width, height)

    if found:
        rotVectors, traVectors, _ = getPositionVectors(sourceImage, corners, cameraMatrix, distortionCoeffs, width, height)

        coordNames = ['Roll: ', 'Pitch: ', 'Yaw: ', 'Tra X: ', 'Tra Y: ', 'Tra Z: ']
        coords = calculateCoordinates(rotVectors, traVectors, RFlip, scale=squareSize)

        font = cv2.FONT_HERSHEY_SIMPLEX
        initialPosition = (10,100)
        positionIncrement = 50
        scale = 1.0
        color = (0,128,255)
        thickness = 1

        for i in range(len(coords) + 3):
            position = (initialPosition[0], initialPosition[1] + i * positionIncrement)
            if(i < len(coords)):
                cv2.putText(sourceImage, coordNames[i] + '%.3f' % coords[i], position, font, scale, color, thickness, cv2.LINE_AA)
            elif(i == len(coords)):
                cv2.putText(sourceImage, '+X axis: Blue', position, font, scale, (255, 0, 0), thickness, cv2.LINE_AA)
            elif(i == (len(coords) + 1)):
                cv2.putText(sourceImage, '+Y axis: Green', position, font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
            elif(i == (len(coords) + 2)):
                cv2.putText(sourceImage, '-Z axis: Red', position, font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

        imgWithCoords = drawOnChessboard(sourceImage, outputFile, 2, cameraMatrix, distortionCoeffs, rotVectors, traVectors, corners, width=9, height=6, scale=1.0)

        # print('Press a key to close the image and continue')
        # cv2.imshow(outputFile, cv2.resize(imgWithCoords, None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(outputFile, imgWithCoords)
    else:
        print('Chessboard not found on image')