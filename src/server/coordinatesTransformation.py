def transformationDictionary(pose, objectType):
    return {
        'marker': {'0_roll':  +pose['roll'] + 90,
                   '1_pitch': +pose['yaw'],
                   '2_yaw':   -pose['pitch'] + 90,
                   '3_x':     +pose['z'],
                   '4_y':     +pose['x'],
                   '5_z':     -pose['y']},
        'hmd':    {'0_roll':  +pose['yaw'],
                   '1_pitch': -pose['roll'],
                   '2_yaw':   -pose['pitch'] +180,
                   '3_x':     +pose['z'],
                   '4_y':     +pose['x'],
                   '5_z':     -pose['y']}
    }.get(objectType, {})

def transformationDictionaryFromPivot(pose, objectType):
    return {
        'marker_pivot': {'0_roll':  0,
                         '1_pitch': 0,
                         '2_yaw':   90,
                         '3_x':     0,
                         '4_y':     0,
                         '5_z':     0},
        'hmd':          {'0_roll':  +pose['yaw'], 
                         '1_pitch': -pose['roll'], 
                         '2_yaw':   +pose['pitch'],
                         '3_x':     +pose['y'],
                         '4_y':     +pose['x'],
                         '5_z':     +pose['z']}
    }.get(objectType, {'0_roll':  -pose['roll'], # ok
                       '1_pitch': +pose['pitch'], 
                       '2_yaw':   +pose['yaw'],  # ok
                       '3_x':     -pose['y'],  # ok
                       '4_y':     +pose['x'],  # ok
                       '5_z':     -pose['z']}) # ok

def posesToUnrealCoordinates(poses):
    unrealCoordinates = {}
    for objectId, data in poses.items():
        if data.get('found', False):
            objectType = objectId.split('_')[0]
            unrealCoordinates[objectId] = {'found': True,
                                           'pose':  transformationDictionary(data['pose'], objectType)}
        else:
            unrealCoordinates[objectId] = {'found': False}

    return unrealCoordinates

def posesToUnrealCoordinatesFromPivot(poses):
    unrealCoordinates = {}
    for objectId, data in poses.items():
        if data.get('found', False):
            unrealCoordinates[objectId] = {'found': True,
                                           'pose':  transformationDictionaryFromPivot(data['pose'], objectId)}
        else:
            unrealCoordinates[objectId] = {'found': False}

    return unrealCoordinates