import cv2.cv2 as cv2
import math
import numpy as np
import pprint

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

def calculateCoordinates(rVec, tVec, RFlip, scale=None, printCameraPosition=False):
    # Converting the rVector into Euler angles
    rMatrix = np.matrix(cv2.Rodrigues(rVec)[0])
    roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix) # Flipping before converting
    traX, traY, traZ = tVec

    if(printCameraPosition):
        cameraTranslation = (-1) * rMatrix.T * np.matrix(tVec)

        print('----------------------------------------------')
        print('Camera translation in relation to marker')
        pprint.pprint(['X: {:.2f}'.format(cameraTranslation.item((0,0))), 
                       'Y: {:.2f}'.format(cameraTranslation.item((1,0))), 
                       'Z: {:.2f}'.format(cameraTranslation.item((2,0)))])
        print('Camera rotation in relation to marker')
        pprint.pprint(['Roll: {:.2f}'.format(math.degrees(roll)), 
                       'Pitch: {:.2f}'.format(math.degrees(pitch)), 
                       'Yaw: {:.2f}'.format(math.degrees(yaw))])
        print('----------------------------------------------')
        print('\n\n\n\n')

    if scale is None:
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw), traX[0], traY[0], traZ[0]]
    else:
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw), 
                traX[0] * scale, traY[0] * scale, traZ[0] * scale]

def calculateCameraCoordinates(rVec, tVec, RFlip):
    # Converting the rVector into Euler angles
    rMatrix = np.matrix(cv2.Rodrigues(rVec)[0])
    roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix)

    cameraTranslation = (-1) * rMatrix.T * np.matrix(tVec)

    return {'roll':  math.degrees(roll),
            'pitch': math.degrees(pitch),
            'yaw':   math.degrees(yaw),
            'x':     cameraTranslation.item((0,0)),
            'y':     cameraTranslation.item((1,0)),
            'z':     cameraTranslation.item((2,0))}

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