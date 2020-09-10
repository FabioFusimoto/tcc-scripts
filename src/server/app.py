from flask import Flask, jsonify, Response, request, session
import logging
import math
import numpy as np
import pprint
import signal
import sys
from time import sleep
from tinydb import TinyDB, Query

from src.calibration.commons import loadCalibrationCoefficients
import src.webcamUtilities.video as webVideo
import src.USBCam.video as USBVideo
from src.server.coordinatesEstimation import estimatePoses, estimatePosesFromPivot, estimatePosesFromMultiplePivots, getCoordinateTransformationMatrixes, estimatePosesFromMultiplePivots2
from src.server.coordinatesTransformation import posesToUnrealCoordinates, posesToUnrealCoordinatesFromPivot
from src.server.databaseFunctions import saveCoordinates
from src.server.objects import OBJECT_DESCRIPTION
from src.server.utils import livePoseEstimation

# General App setup
app = Flask(__name__)
app.config['SECRET_KEY'] = b'SECRET_KEY'

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Camera parameters and configuration
calibrationFile = 'src/server/files/J7-pro.yml'
cameraMatrix, distCoeffs = loadCalibrationCoefficients(calibrationFile)

camType = 'USB'

cam = {}

if camType == 'USB':
    cam = USBVideo.USBCamVideoStream(camIndex=1).start()
else:
    cam = webVideo.ThreadedWebCam().start()

# Database setup
db = TinyDB('db.json')
Marker = Query()

# Helper functions - Session is used to store data between requests
def updateSession(newCoordinates):
    session.permanent = True
    for markerId, markerPose in newCoordinates.items():
        if 'markerPosesRelativeToReference' in session.keys():
            session['markerPosesRelativeToReference'].update({markerId: markerPose})
        else:
            session['markerPosesRelativeToReference'] = {markerId: markerPose}

def updateSessionFromDatabase():
    session.permanent = True
    for relativePose in db:
        markerId = relativePose['markerId']
        relation = relativePose['relation']
        relativePose.pop('markerId')
        relativePose.pop('relation')
        if relation in session.keys():
            session[relation].update({markerId: relativePose})
        else:
            session[relation] = {markerId: relativePose}

@app.route('/')
def home():
    return 'Homepage'

# Pivot related routes
@app.route('/discover-pivots')
def getReferencePoseRelativeToPivot():
    session.permanent = True

    referencePivotId = request.args.get('reference-pivot', default=3, type=int)
    targetPivotId = request.args.get('target-pivot', default=5, type=int)

    # Initiliazing the database, knowing that the reference is on the origin
    origin = {'x':     0,
              'y':     0,
              'z':     0,
              'roll':  0,
              'pitch': 0,
              'yaw':   0}
    saveCoordinates(db, str(referencePivotId), origin, 'referencePosesRelativeToPivot', Marker)
    saveCoordinates(db, str(referencePivotId), origin, 'pivotPosesRelativeToReference', Marker)

    # Getting the reference marker pose, written in pivot coordinates
    poses = estimatePosesFromPivot([referencePivotId], targetPivotId, cameraMatrix, distCoeffs, cam)

    referencePoseRelativeToTargetPivot = poses[str(referencePivotId)]
    saveCoordinates(db, str(targetPivotId), referencePoseRelativeToTargetPivot, 'referencePosesRelativeToPivot', Marker)

    # Getting the pivot marker poses, written in reference coordinates
    poses = estimatePosesFromPivot([targetPivotId], referencePivotId, cameraMatrix, distCoeffs, cam)

    targetPivotPoseRelativeToReference = poses[str(targetPivotId)]
    saveCoordinates(db, str(targetPivotId), targetPivotPoseRelativeToReference, 'pivotPosesRelativeToReference', Marker)

    updatedPivotCoords = db.all()

    for c in updatedPivotCoords:
        c['roll'] *= 180/np.pi
        c['pitch'] *= 180/np.pi
        c['yaw'] *= 180/np.pi

    return jsonify(updatedPivotCoords)

@app.route('/clear-pivots')
def clearPivots():
    session.permanent = True

    db.truncate()
    return 'Database: ' + str(db.all())

@app.route('/load-pivots')
def loadPivots():
    session.permanent = True

    updateSessionFromDatabase()

    return jsonify(session._get_current_object())

# Pose estimation related routes
@app.route('/pose')
def getCoordinates():
    session.permanent = True
    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))

    poses = estimatePoses(markerIds, cameraMatrix, distCoeffs, cam, camType)
    unrealCoordinates = posesToUnrealCoordinates(poses)

    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('markerPosesRelativeToReference', {}))

@app.route('/pose-from-pivot')
def getCoordinatesFromPivotPerspective():
    session.permanent = True
    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))
    pivotMarkerId = request.args.get('pivot', default=3, type=int)
    
    poses = estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam, camType)

    # posesToPrint = {'marker_0': {'roll': 0,
    #                              'pitch': 0,
    #                              'yaw': 0,
    #                              'x': 0,
    #                              'y': 0,
    #                              'z': 0}}

    # marker = poses.get('marker_0', {'found': False})
    # if marker['found']:
    #     pose = marker['pose']
    #     posesToPrint['marker_0']['roll'] = math.degrees(pose['roll'])
    #     posesToPrint['marker_0']['pitch'] = math.degrees(pose['pitch'])
    #     posesToPrint['marker_0']['yaw'] = math.degrees(pose['yaw'])
    #     posesToPrint['marker_0']['x'] = pose['x']
    #     posesToPrint['marker_0']['y'] = pose['y']
    #     posesToPrint['marker_0']['z'] = pose['z']

    # pprint.pprint(posesToPrint)

    unrealCoordinates = posesToUnrealCoordinatesFromPivot(poses)
    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('markerPosesRelativeToReference', {}))

@app.route('/pose-from-multiple-pivots')
def getPoseFromMultiplePivots():
    session.permanent = True
    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if ((k != 'hmd') and (OBJECT_DESCRIPTION[k]['objectType'] not in ['pivot', 'reference']))]))

    referencePosesRelativeToPivot = session._get_current_object().get('referencePosesRelativeToPivot', {})
    pivotPosesRelativeToReference = session._get_current_object().get('pivotPosesRelativeToReference', {})
    markerPosesRelativeToReference = estimatePosesFromMultiplePivots(markerIds, referencePosesRelativeToPivot, pivotPosesRelativeToReference, cameraMatrix, distCoeffs, cam, camType)

    updateSession(markerPosesRelativeToReference)

    markerPosesFromSession = session._get_current_object().get('markerPosesRelativeToReference', {})
    knownPoses = {**pivotPosesRelativeToReference, **markerPosesFromSession} # merging the pivot poses and marker poses

    unrealCoordinates = posesToUnrealCoordinatesFromPivot(knownPoses)

    return jsonify(unrealCoordinates)

@app.route('/pose-from-multiple-pivots-2')
def getPoseFromMultiplePivots2():
    referenceId = request.args.get('reference', default=3, type=int)
    pivotMarkerId = request.args.get('pivot', default=7, type=int)

    transformationParameters = getCoordinateTransformationMatrixes(pivotMarkerId, referenceId, cameraMatrix, distCoeffs, cam)

    poses = estimatePosesFromMultiplePivots2(['0'], [str(pivotMarkerId)], referenceId, transformationParameters, cameraMatrix, distCoeffs, cam=cam)

    return jsonify(poses)

@app.route('/pose-sequence')
def getCoordinateSequence():
    session.permanent = True
    markerIds = list(map(int, OBJECT_DESCRIPTION.keys()))
    pivotMarkerId = request.args.get('pivot', default=3, type=int)
    framesToCapture = request.args.get('count', default=60, type=int)

    poseSequence = {'roll':  [],
                    'pitch': [],
                    'yaw':   [],
                    'x':     [],
                    'y':     [],
                    'z':     []}

    i = 0
    while i < framesToCapture:
        poses = estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam, camType)
        if poses['hmd']['found']:
            hmdPose = poses['hmd']['pose']
            poseSequence['roll'].append(math.degrees(hmdPose['roll']))
            poseSequence['pitch'].append(math.degrees(hmdPose['pitch']))
            poseSequence['yaw'].append(math.degrees(hmdPose['yaw']))
            poseSequence['x'].append(hmdPose['x'])
            poseSequence['y'].append(hmdPose['y'])
            poseSequence['z'].append(hmdPose['z'])
        i += 1
        sleep(1/10)

    return jsonify(poseSequence)

@app.route('/live-pose')
def estimateLivePose():
    markerId = request.args.get('marker', default=3, type=int)
    lastKnownCoords = livePoseEstimation(markerId, cameraMatrix, distCoeffs, cam, camType)

    return jsonify(lastKnownCoords)

@app.before_first_request
def clearPreviousSession():
    session.permanent = True
    session.clear()
    _ = loadPivots()

# On exit (Ctrl + C), we should stop the camera thread so the script stops as expected
def exitHandler(signal, frame):
    cam.stop()
    print('Cam stopped')
    sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exitHandler)
    app.run('127.0.0.1',port=5000)