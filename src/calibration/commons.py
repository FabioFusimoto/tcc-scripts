import cv2.cv2 as cv2
import math
import numpy as np

# Checks if matrix is a valid rotation matrix
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def getImageAndResize(sourceFile, scale):
    sourceImage = cv2.imread(sourceFile)
    return cv2.resize(sourceImage, None, fx=scale, fy=scale)

def calculateCoordinates(rVecs, tVecs, RFlip, scale=None):
    # Converting the rVector into Euler angles
    rMatrix = np.matrix(cv2.Rodrigues(rVecs)[0])
    roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix) # Flipping before converting
    traX, traY, traZ = tVecs

    if scale is None:
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw), traX[0], traY[0], traZ[0]]
    else:
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw), 
                traX[0] * scale, traY[0] * scale, traZ[0] * scale]

def saveCalibrationCoefficients(mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to a given file"""
    cvFileHandler = cv2.FileStorage(path, flags=1)
    cvFileHandler.write("K", mtx)
    cvFileHandler.write("D", dist)
    cvFileHandler.release()

def loadCalibrationCoefficients(path):
    """Loads camera matrix and distortion coefficients from path"""
    cvFileHandler = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    cameraMatrix = cvFileHandler.getNode("K").mat()
    distortionCoefficients = cvFileHandler.getNode("D").mat()

    cvFileHandler.release()
    return [cameraMatrix, distortionCoefficients]