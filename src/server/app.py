from flask import Flask, jsonify, Response, request, session
import logging
import math
import pprint
import signal
import sys
from time import sleep
from tinydb import TinyDB, Query

from src.calibration.commons import loadCalibrationCoefficients
import src.webcamUtilities.video as webVideo
import src.USBCam.video as USBVideo
from src.server.coordinatesEstimation import estimatePoses, estimatePosesFromPivot
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
    cam = USBVideo.USBCamVideoStream(camIndex=2).start()
else:
    cam = webVideo.ThreadedWebCam().start()

# Database setup
db = TinyDB('db.json')
Marker = Query()

# Helper function - Session is used to store data between requests
def updateSession(newCoordinates):
    session.permanent = True
    for markerId, data in newCoordinates.items():
        if data['found'] == True:
            if 'poses' in session.keys():
                session['poses'].update({markerId: data['pose']})
            else:
                session['poses'] = {markerId: data['pose']}

@app.route('/')
def home():
    return 'Homepage'

@app.route('/pose')
def getCoordinates():
    session.permanent = True
    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))

    poses = estimatePoses(markerIds, cameraMatrix, distCoeffs, cam, camType)
    unrealCoordinates = posesToUnrealCoordinates(poses)

    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('poses', {}))

@app.route('/discover-pivots')
def savePivotRelativePositions():
    session.permanent = True
    pivotMarkerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if OBJECT_DESCRIPTION[k]['objectType'] == 'pivot']))

    # The reference is the coordinate system origin
    referencePivotId = request.args.get('reference-pivot', default=3, type=int)

    poses = estimatePosesFromPivot(pivotMarkerIds, referencePivotId, cameraMatrix, distCoeffs, cam)
    saveCoordinates(db, poses, Marker)

    updatedPivotCoords = db.all()

    return jsonify(updatedPivotCoords)

@app.route('/clear-pivots')
def clearPivots():
    session.permanent = True

    db.truncate()
    return 'Database: ' + str(db.all())

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

    return jsonify(session._get_current_object().get('poses', {}))

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

# On exit (Ctrl + C), we should stop the camera thread so the script stops as expected
def exitHandler(signal, frame):
    cam.stop()
    print('Cam stopped')
    sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exitHandler)
    app.run('127.0.0.1',port=5000)