import math
import pprint
from src.server.objects import OBJECT_DESCRIPTION

def testContextTransformations(pose, objectType):
    return {
        'reference': {'0_roll':  0,
                      '1_pitch': 0,
                      '2_yaw':   0,
                      '3_x':     0,
                      '4_y':     0,
                      '5_z':     0},
        'pivot':     {'0_roll':  +math.degrees(pose['roll']),
                      '1_pitch': -math.degrees(pose['pitch']),
                      '2_yaw':   -math.degrees(pose['yaw']),
                      '3_x':     +pose['x'],
                      '4_y':     +pose['z'],
                      '5_z':     +pose['y']},
        'hmd':       {'0_roll':  +math.degrees(pose['yaw']),        # ok
                      '1_pitch': +math.degrees(pose['roll']) + 180, # ok
                      '2_yaw':   -math.degrees(pose['pitch']),
                      '3_x':     +pose['x'],
                      '4_y':     -pose['y'],
                      '5_z':     +pose['z']},
        'syringe':   {'0_roll':  -math.degrees(pose['roll']) + 180,
                      '1_pitch': +math.degrees(pose['yaw']),
                      '2_yaw':   +math.degrees(pose['pitch']) + 180,
                      '3_x':     +pose['x'],
                      '4_y':     -pose['z'],
                      '5_z':     -pose['y']},
        'arm':       {'0_roll':  +90,
                      '1_pitch': 0,
                      '2_yaw':   +180,
                      '3_x':     0,
                      '4_y':     0,
                      '5_z':     0}
    }.get(objectType, {'0_roll':  +math.degrees(pose['roll']) - 90,
                       '1_pitch': -math.degrees(pose['yaw']),
                       '2_yaw':   +math.degrees(pose['pitch']),
                       '3_x':     +pose['x'],
                       '4_y':     -pose['z'],
                       '5_z':     -pose['y']})

def vidaEContextTransformations(pose, objectType):
    return {
        'hmd': {'0_roll':  -math.degrees(pose['yaw']),
                '1_pitch': +math.degrees(pose['roll']) + 180,
                '2_yaw':   -math.degrees(pose['pitch']),
                '3_x':     +875 - pose['z'],
                '4_y':      -10 + pose['x'],
                '5_z':     +120 + pose['y']}
    }.get(objectType, {'0_roll':  +math.degrees(pose['roll']),
                       '1_pitch': +math.degrees(pose['pitch']),
                       '2_yaw':   +math.degrees(pose['yaw']),
                       '3_x':     +pose['x'],
                       '4_y':     +pose['z'],
                       '5_z':     +pose['y']})

def transformCoordinates(pose, objectType, context):
    if context == 'vida-enfermagem':
        return vidaEContextTransformations(pose, objectType)
    else:
        return testContextTransformations(pose, objectType)

def posesToUnrealCoordinates(poses, context):
    unrealCoordinates = {}
    for objectId, pose in poses.items():
        objectName = OBJECT_DESCRIPTION[str(objectId)]['objectName']
        objectType = OBJECT_DESCRIPTION[str(objectId)]['objectType']
        unrealCoordinates[objectName] = transformCoordinates(pose, objectType, context)

    return unrealCoordinates

def posesToUnrealCoordinatesFromPivot(poses, context):
    unrealCoordinates = {}

    # print('\nRaw HMD pose')
    # rawHmdPose = poses['hmd']

    # for k in ['roll', 'pitch', 'yaw']:
    #     rawHmdPose[k] *= 180/3.14

    # pprint.pprint(rawHmdPose)

    for objectId, objectPose in poses.items():
        objectName = OBJECT_DESCRIPTION[str(objectId)]['objectName']
        objectType = OBJECT_DESCRIPTION[str(objectId)]['objectType']
        unrealCoordinates[objectName] = transformCoordinates(objectPose, objectType, context)

    return unrealCoordinates