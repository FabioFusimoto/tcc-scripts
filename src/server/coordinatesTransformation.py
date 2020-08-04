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

def posesToUnrealCoordinates(poses):
    unrealCoordinates = {}
    for objectId, data in poses.items():
        if data.get('found', False):
            objectType = objectId.split('_')[0]
            unrealCoordinates[objectId] = {'found':       True,
                                           'coordinates': transformationDictionary(data['pose'], objectType)}
        else:
            unrealCoordinates[objectId] = {'found': False}

    return unrealCoordinates