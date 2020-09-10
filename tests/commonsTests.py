import numpy as np
import pprint

import src.calibration.commons as commons

def testRotationEquivalences(rVec):
    RTFromVector = commons.getRMatrixFromVector(rVec).T

    print('RTFromVector:')
    pprint.pprint(RTFromVector)

    eulerAngles = commons.getEulerAnglesFromRVector(rVec, degrees=False)

    print('\nEuler angles:')
    pprint.pprint(eulerAngles)

    RTFromEulerAngles = commons.getRMatrixFromEulerAngles(eulerAngles[0][0], eulerAngles[0][1], eulerAngles[0][2]).T

    print('\nRTFromEulerAngles:')
    pprint.pprint(RTFromEulerAngles)

def testTransformationMatrix(tVec, M):
    newCoordinates = commons.transformCoordinates(tVec, M)

    print('New coordinates')
    pprint.pprint(newCoordinates)

# rotationVector = np.array([[-2.9646, 0.0495, -0.7202]])
# testRotationEquivalences(rotationVector)

# tVec = np.array([-17.03519408939539, -0.2473057468419321, 42.2326447616194, 1])
# M = np.array([[ 0.834 , -0.0285,  0.551 , -9.0694],
#               [-0.0265, -0.9996, -0.0117, -0.2052],
#               [ 0.5511, -0.0049, -0.8344, 44.6269],
#               [ 0.    ,  0.    ,  0.    ,  1.    ]])

# Refererence pose given in pivot coords
tVec = np.array([22.71440954827702, 0.7692142726460192, 2.5459999449895108, 1])

M = np.array([[  0.6207,   0.0187,  -0.7838, -12.1179],
              [ -0.0792,   0.9961,  -0.0389,   1.1319],
              [  0.78  ,   0.0862,   0.6198, -19.3621],
              [  0.    ,   0.    ,   0.    ,   1.    ]])

print('\nExpected output: [0.0, 0.0, 0.0]')
testTransformationMatrix(tVec, M)

tVec = np.array([0, 0, 10, 1])

M = np.array([[  0.8856,  -0.026 ,   0.4637, -16.5803],
              [ -0.0415,  -0.9989,   0.0233,  -2.0686],
              [  0.4626,  -0.0399,  -0.8857,  47.758 ],
              [  0.    ,   0.    ,   0.    ,   1.    ]])

print('\nExpected output: [-16.6, -2.1, 47.9]')
testTransformationMatrix(tVec, M)