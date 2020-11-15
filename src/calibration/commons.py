import cv2.cv2 as cv2
import math
import numpy as np
import pprint
from scipy.spatial.transform import Rotation

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

def getRMatrixFromVector(rVec, invert=False):
    # return np.matrix(cv2.Rodrigues(rVec)[0])
    r = Rotation.from_rotvec(rVec.reshape(3,))

    if invert:
        r = r.inv()

    return r.as_matrix()

def composeRotations(rVec1, rVec2, inversions=[False, False]):
    r1 = Rotation.from_rotvec(rVec1.reshape(3,))
    r2 = Rotation.from_rotvec(rVec2.reshape(3,))

    if (inversions[0]):
        r1 = r1.inv()
    if (inversions[1]):
        r2 = r2.inv()

    rComposed = r1 * r2
    rComposed = rComposed.as_euler('xyz')

    return {
        'roll': rComposed[0],
        'pitch': rComposed[1],
        'yaw': rComposed[2]
    }

def getRMatrixFromEulerAngles(roll, pitch, yaw, degrees=False):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_matrix()

def getRVectorFromEulerAngles(roll, pitch, yaw, degrees=False):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_rotvec()

def getEulerAnglesFromRVector(rVec, degrees=False):
    rVec = rVec.reshape(3,)
    r = Rotation.from_rotvec(rVec)
    return r.as_euler('xyz', degrees=degrees)

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

def calculateRelativePoseFromVectors(objRVec, objTVec, tgtPose, RTSrcToTgt, scale=1.0, objOffset=None):
    np.set_printoptions(precision=4, suppress=True)
    
    objPose = calculateCoordinates(np.reshape(objRVec, (3,1)), np.reshape(objTVec, (3,1)), scale=scale)

    return calculateRelativePoseFromPose(objPose, tgtPose, RTSrcToTgt, objOffset=objOffset)

def calculateRelativePoseFromPose(objPose, tgtPose, RTSrcToTgt, objOffset=None):
    np.set_printoptions(precision=4, suppress=True)

    objTranslation = np.array([[objPose['x']],
                               [objPose['y']],
                               [objPose['z']]])
    if objOffset is not None:
        for k in ['x', 'y', 'z']:
            objTranslation[k] += objOffset[k]
    
    print('\nCALCULATE RELATIVE POSE -> OBJECT TRANSLATION RELATIVE TO SOURCE [SOURCE COORDINATES]')
    pprint.pprint(objTranslation)

    tgtTranslation = np.array([[tgtPose['x']],
                               [tgtPose['y']],
                               [tgtPose['z']]])

    print('\nCALCULATE RELATIVE POSE -> TARGET TRANSLATION RELATIVE TO SOURCE [SOURCE COORDINATES]')
    pprint.pprint(tgtTranslation)

    objTranslationRelativeToTgt = np.subtract(objTranslation, tgtTranslation)

    print('\nCALCULATE RELATIVE POSE -> OBJECT TRANSLATION RELATIVE TO TARGET [SOURCE COORDINATES]')
    pprint.pprint(objTranslationRelativeToTgt)

    objTranslationOnTgtCoordinates = np.dot(RTSrcToTgt, objTranslationRelativeToTgt)

    print('\nCALCULATE RELATIVE POSE -> OBJECT TRANSLATION RELATIVE TO SOURCE [TARGET COORDINATES]')
    pprint.pprint(objTranslationOnTgtCoordinates)

    print('\n----------------------------------------\n')

    objRoll, objPitch, objYaw = objPose['roll'], objPose['pitch'], objPose['yaw']

    return {'roll':  objRoll - tgtPose['roll'],
            'pitch': objPitch - tgtPose['pitch'],
            'yaw':   objYaw - tgtPose['yaw'],
            'x':     objTranslationOnTgtCoordinates.item(0),
            'y':     objTranslationOnTgtCoordinates.item(1),
            'z':     objTranslationOnTgtCoordinates.item(2)}

def getTransformationMatrix(p, rotationOnly=False):
    '''Returns a 4x4 transformation matrix which converts coordinates between references'''
    RT = getRMatrixFromEulerAngles(p['roll'], p['pitch'], p['yaw']).T # Rotation matrix

    tra = np.dot(-RT, np.array([p['x'], p['y'], p['z']]))

    if rotationOnly:
        M = np.array([[RT.item((0,0)), RT.item((0,1)), RT.item((0,2)), 0],
                      [RT.item((1,0)), RT.item((1,1)), RT.item((1,2)), 0],
                      [RT.item((2,0)), RT.item((2,1)), RT.item((2,2)), 0],
                      [             0,              0,              0, 1]])
    else:
        M = np.array([[RT.item((0,0)), RT.item((0,1)), RT.item((0,2)), tra.item(0)],
                      [RT.item((1,0)), RT.item((1,1)), RT.item((1,2)), tra.item(1)],
                      [RT.item((2,0)), RT.item((2,1)), RT.item((2,2)), tra.item(2)],
                      [             0,              0,              0,           1]])
 
    return M

def transformCoordinates(tVec, M):
    '''Returns the translation vector transformed into the target coordinate system'''
    if isinstance(tVec, dict): # converting from dict to np.array if necessary
        tVec = np.array([
            tVec['x'],
            tVec['y'],
            tVec['z'],
            1
        ])

    tVecT = tVec.T
    product = np.dot(M, tVecT)

    return {'x': product.item(0),
            'y': product.item(1),
            'z': product.item(2)}

def inversePerspective(rVec, tVec, scale=1.0):
    if all(vec.shape != (3,1) for vec in [rVec, tVec]):
        rVec, tVec = rVec.reshape((3, 1)), tVec.reshape((3, 1))
    R, _ = cv2.Rodrigues(rVec)
    R = np.matrix(R).T
    invRVec , _ = cv2.Rodrigues(R)
    invTVec = np.dot(R, np.matrix(-tVec))

    if scale != 1.0:
        invTVec = np.dot(scale, invTVec)

    return invRVec, invTVec

def relativePosition(rVec1, tVec1, rVec2, tVec2, scale=1.0, asEuler=False):
    if all(vec.shape != (3,1) for vec in [rVec1, tVec1]):
        rVec1, tVec1 = rVec1.reshape((3, 1)), tVec1.reshape((3, 1))
        rVec2, tVec2 = rVec2.reshape((3, 1)), tVec2.reshape((3, 1))

    # Invert the second marker
    invRVec, invTVec = inversePerspective(rVec2, tVec2)
    vecComposition = cv2.composeRT(rVec1, tVec1, invRVec, invTVec)
    relativeRotation, relativeTranslation = vecComposition[0], vecComposition[1]

    relativeTranslation = np.dot(scale, relativeTranslation)
    if asEuler:
        relativeRotation = getEulerAnglesFromRVector(relativeRotation)

    return relativeRotation, np.reshape(relativeTranslation, (3,))

def compensateForOffset(tVec, offset):
    tVec[0, 0] -= offset['x']
    tVec[0, 1] -= offset['y']
    tVec[0, 2] -= offset['z']

    return tVec

def compensateForRotation(previousPose, newPose):
    if all (k in previousPose.keys() for k in ['roll', 'pitch', 'yaw']):
        for k in ['roll', 'pitch', 'yaw']:
            while abs(newPose[k] - previousPose[k]) > np.pi:
                if (newPose[k] - previousPose[k]) < np.pi:
                    newPose[k] += 2 * np.pi
                elif (newPose[k] - previousPose[k]) > np.pi:
                    newPose[k] -= 2 * np.pi

    return newPose

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