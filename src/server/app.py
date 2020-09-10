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
from src.server.coordinatesEstimation import estimatePoses, estimatePosesFromPivot, estimatePosesFromMultiplePivots, discoverReferencePoseRelativeToPivot
from src.server.coordinatesTransformation import posesToUnrealCoordinates, posesToUnrealCoordinatesFromPivot
from src.server.databaseFunctions import saveCoordinates
from src.server.objects import OBJECT_DESCRIPTION
from src.server.utils import livePoseEstimation

#####################
# GENERAL APP SETUP #
#####################

app = Flask(__name__)
app.config['SECRET_KEY'] = b'SECRET_KEY'

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#######################################
# CAMERA PARAMETERS AND CONFIGURATION #
#######################################

calibrationFile = 'src/server/files/J7-pro.yml'
cameraMatrix, distCoeffs = loadCalibrationCoefficients(calibrationFile)

camType = 'USB'

cam = {}

if camType == 'USB':
    cam = USBVideo.USBCamVideoStream(camIndex=1).start()
else:
    cam = webVideo.ThreadedWebCam().start()

##################
# DATABASE SETUP #
##################

db = TinyDB('db.json')
Marker = Query()

####################
# HELPER FUNCTIONS #
####################

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

# Session is used to store data between requests
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

##############
# HOME ROUTE #
##############
@app.route('/')
def home():
    return 'Homepage'

########################
# PIVOT RELATED ROUTES #
########################
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

##################################
# POSE ESTIMATION RELATED ROUTES #
##################################

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

    unrealCoordinates = posesToUnrealCoordinatesFromPivot(poses)
    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('markerPosesRelativeToReference', {}))

@app.route('/pose-from-multiple-pivots')
def getPoseFromMultiplePivots():
    referenceId = request.args.get('reference', default=3, type=int)
    pivotMarkerId = request.args.get('pivot', default=7, type=int)

    referencePoseRelativeToPivots = discoverReferencePoseRelativeToPivot(pivotMarkerId, referenceId, cameraMatrix, distCoeffs, cam)

    poses = estimatePosesFromMultiplePivots([], [str(pivotMarkerId)], referenceId, referencePoseRelativeToPivots, cameraMatrix, distCoeffs, cam=cam)

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

####################################
# PARAMETERS FOR EXECUTING THE APP #
####################################

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exitHandler)
    app.run('127.0.0.1',port=5000)