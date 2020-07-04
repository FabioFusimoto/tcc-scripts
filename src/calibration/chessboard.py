from timeit import default_timer as timer

import cv2.cv2 as cv2
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(sourcePath, outputPath, prefix, imageFormat, squareSize, width=9, height=6, shouldDownsize=True, scale=0.5):
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

        photo = cv2.imread(f)

        print('\nReading image ' + str(f).split('\\')[1]) 

        # Downsizing large photos may improve accuracy and reduce procesing time
        if(shouldDownsize):
            photo = cv2.resize(photo, None, fx=scale, fy=scale)

        grayPhoto = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        # Find the chessboards corners
        flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        patternWasFound, corners = cv2.findChessboardCorners(grayPhoto, (width, height), flags)

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

def saveCalibrationCoeficients(mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to a given file"""
    cvFileHandler = cv2.FileStorage(path, flags=1)
    cvFileHandler.write("K", mtx)
    cvFileHandler.write("D", dist)
    cvFileHandler.release()

def loadCalibrationCoeficients(path):
    """Loads camera matrix and distortion coefficients from path"""
    cvFileHandler = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    cameraMatrix = cvFileHandler.getNode("K").mat()
    distortionCoefficients = cvFileHandler.getNode("D").mat()

    cvFileHandler.release()
    return [cameraMatrix, distortionCoefficients]

def undistortImage(sourceFile, outputFile, cameraMatrix, distortionCoeffs, scale=1.0):
    originalImage = cv2.imread(sourceFile)
    scaledImage = cv2.resize(originalImage, None, fx=scale, fy=scale)
    h, w = scaledImage.shape[:2] # scaled image resolution - height x width
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs, (w, h), 1, (w, h))

    # Correcting distortions and cropping the result
    undistortedImage = cv2.undistort(scaledImage, cameraMatrix, distortionCoeffs, None, newCameraMatrix)
    x, y, w, h = roi
    undistortedImage = undistortedImage[y:y+h, x:x+w]
    cv2.imwrite(outputFile, undistortedImage)

def drawAxis(image, corners, imagePoints):
    corner = tuple(corners[0].ravel())

    image = cv2.line(image, corner, tuple(imagePoints[0].ravel()), (255,0,0), 5)
    image = cv2.line(image, corner, tuple(imagePoints[1].ravel()), (0,255,0), 5)
    image = cv2.line(image, corner, tuple(imagePoints[2].ravel()), (0,0,255), 5)
    
    return image

def drawOnChessboard(sourceFile, outputFile, axisLength, cameraMatrix, distortionCoeffs, width=9, height=6, scale=1.0):
    objectPoints = np.zeros((height*width, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    axis = np.array([[axisLength,0,0], [0,axisLength,0], [0,0,-axisLength]], dtype='f').reshape(-1, 3)

    sourceImage = cv2.imread(sourceFile)
    sourceImage = cv2.resize(sourceImage, None, fx=scale, fy=scale)

    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(grayImage, (width, height), flags)

    if found:
        refinedCorners = cv2.cornerSubPix(grayImage, corners, (11,11), (-1, -1), criteria)

        # Find the rotation and translation vectors
        _, rotVectors, traVectors, _ = cv2.solvePnPRansac(objectPoints, refinedCorners, cameraMatrix, distortionCoeffs)

        # Project 3D points to the image plane
        imagePoints, _ = cv2.projectPoints(axis, rotVectors, traVectors, cameraMatrix, distortionCoeffs)

        imageWithAxis = drawAxis(sourceImage, refinedCorners, imagePoints)
        cv2.imshow('Drawn axis', cv2.resize(imageWithAxis, None, fx=0.3, fy=0.3)) # displaying the resized image so it fits

        print('Press a key to exit and save the image')

        cv2.waitKey(0)
        cv2.imwrite(outputFile, imageWithAxis)
    
    cv2.destroyAllWindows()

def getPositionVectors(sourceFile, cameraMatrix, distortionCoeffs, squareSize=1, width=9, height=6, scale=1.0):
    objectPoints = np.zeros((height*width, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    sourceImage = cv2.imread(sourceFile)
    sourceImage = cv2.resize(sourceImage, None, fx=scale, fy=scale)

    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(grayImage, (width, height), flags)

    if found:
        refinedCorners = cv2.cornerSubPix(grayImage, corners, (11,11), (-1, -1), criteria)

        # Find the rotation and translation vectors
        _, rotVectors, traVectors, _ = cv2.solvePnPRansac(objectPoints, refinedCorners, cameraMatrix, distortionCoeffs)

        return rotVectors, traVectors  * squareSize
    else:
        return [], []

def drawPositionVectors(sourceFile, outputFile, cameraMatrix, distortionCoeffs, squareSize=1, width=9, height=6, scale=1.0):
    rotVectors, traVectors = getPositionVectors(sourceFile, cameraMatrix, distortionCoeffs, squareSize, width, height, scale)
    
    rotX, rotY, rotZ = rotVectors
    traX, traY, traZ = traVectors

    sourceImage = cv2.imread(sourceFile)
    sourceImage = cv2.resize(sourceImage, None, fx=scale, fy=scale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    initialPosition = (10,100)
    positionIncrement = 50
    scale = 1
    color = (255,0,0)
    thickness = 1

    coordNames = ['Rotation X: ', 'Rotation Y: ', 'Rotation Z: ',
                  'Translation X: ', 'Translation Y: ', 'Translation Z: ']
    coords = [rotX[0], rotY[0], rotZ[0], traX[0], traY[0], traZ[0]]

    for i in range(len(coords)):
        position = (initialPosition[0], initialPosition[1] + i * positionIncrement)
        cv2.putText(sourceImage, coordNames[i] + str(coords[i]), position, font, scale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Position vectors', cv2.resize(sourceImage, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)

    cv2.imwrite(outputFile, sourceImage)