def saveCoordinates(db, coords, Marker):
    coords.pop('hmd')
    for key, data in coords.items():
        if data['found']:
            pose = data['pose']
            poseAsDict = {'markerId': key,
                          'x':        pose['x'],
                          'y':        pose['y'],
                          'z':        pose['z'],
                          'roll':     pose['roll'],
                          'pitch':    pose['pitch'],
                          'yaw':      pose['yaw']}
            isInDatabase = len(db.search(Marker.markerId == key)) > 0
            if isInDatabase:
                db.update(poseAsDict, Marker.markerId == key)
            else:
                db.insert(poseAsDict)