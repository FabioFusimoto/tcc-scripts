def saveCoordinates(db, markerId, pose, relation, Marker):
    '''Save the reference coordinates relative to the marker[markerId] coordinates (or
       the other way around)'''
    poseAsDict = {'markerId': markerId,
                  'relation': relation,
                  'x':        pose['x'],
                  'y':        pose['y'],
                  'z':        pose['z'],
                  'roll':     pose['roll'],
                  'pitch':    pose['pitch'],
                  'yaw':      pose['yaw']}
    db.upsert(poseAsDict, ((Marker.markerId == markerId) & (Marker.relation == relation)))