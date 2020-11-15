def saveCoordinates(db, markerId, pose, relation, Marker):
    '''Save the reference coordinates relative to the marker[markerId] coordinates (or
       the other way around)'''
    poseAsDict = {
      'type':     'pivotReferenceRelation',
      'markerId': markerId,
      'relation': relation,
      'x':        pose['x'],
      'y':        pose['y'],
      'z':        pose['z'],
      'roll':     pose['roll'],
      'pitch':    pose['pitch'],
      'yaw':      pose['yaw']}
    db.upsert(poseAsDict, ((Marker.markerId == markerId) & (Marker.relation == relation)))

def fetchPivotPoses(db, Marker):
   return db.search((Marker.type == 'pivotReferenceRelation'))

def clearPivotsFromDatabase(db, Marker):
   db.remove((Marker.type == 'pivotReferenceRelation'))

def fetchPoseWithTimestamp(db, markerId, Marker):
   return db.search((Marker.type == 'poseWithTimestamp') & (Marker.markerId == str(markerId)))

def clearGraphicsPointsFromDatabase(db, Marker):
   db.remove((Marker.type == 'poseWithTimestamp'))