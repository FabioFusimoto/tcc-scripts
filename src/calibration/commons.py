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

def getRMatrixFromVector(rVec):
    return np.matrix(cv2.Rodrigues(rVec)[0])

def calculateCoordinates(rVec, tVec, RFlip=None, scale=None, printCameraPosition=False):
    # Converting the rVector into Euler angles
    rMatrix = getRMatrixFromVector(rVec)
    
    if RFlip is None:
        roll, pitch, yaw = rotationMatrixToEulerAngles(rMatrix)
    else:
        roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix)

    traX, traY, traZ = tVec

    if scale is None:
        return {'roll':  roll, 
                'pitch': pitch, 
                'yaw':   yaw, 
                'x':     traX[0], 
                'y':     traY[0], 
                'z':     traZ[0]}
    else:
        return {'roll':  roll, 
                'pitch': pitch, 
                'yaw':   yaw, 
                'x':     traX[0] * scale, 
                'y':     traY[0] * scale, 
                'z':     traZ[0] * scale}

def calculateCameraCoordinates(rVec, tVec, RFlip=None):
    '''The rotation and translation vectors are the pivot's, as seen by 
       the camera perspective'''
    # Converting the rVector into Euler angles
    rMatrix = getRMatrixFromVector(rVec)

    if RFlip is None:
        roll, pitch, yaw = rotationMatrixToEulerAngles(rMatrix)
    else:
        roll, pitch, yaw = rotationMatrixToEulerAngles(RFlip*rMatrix)

    cameraTranslation = (-1) * rMatrix.T * np.matrix(tVec)

    return {'roll':  roll,
            'pitch': pitch,
            'yaw':   yaw,
            'x':     cameraTranslation.item((0,0)),
            'y':     cameraTranslation.item((1,0)),
            'z':     cameraTranslation.item((2,0))}

def calculateRelativePose(referencePose, RT, objectRVec, objectTVec):
    objectPose = calculateCoordinates(np.reshape(objectRVec, (3,1)), np.reshape(objectTVec, (3,1)))

    markerTranslation = np.dot(RT, np.array([[objectPose['x']],
                                             [objectPose['y']],
                                             [objectPose['z']]])) 

    referenceTranslation = np.dot(RT, np.array([[referencePose['x']],
                                                [referencePose['y']],
                                                [referencePose['z']]]))

    relativeTranslation = np.subtract(markerTranslation, referenceTranslation)

    objRoll, objPitch, objYaw = rotationMatrixToEulerAngles(getRMatrixFromVector(objectRVec))

    return {'roll':  objRoll - referencePose['roll'],
            'pitch': objPitch - referencePose['pitch'],
            'yaw':   objYaw - referencePose['yaw'],
            'x':     relativeTranslation.item((0,0)),
            'y':     relativeTranslation.item((1,0)),
            'z':     relativeTranslation.item((2,0))}

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