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
from src.server.coordinatesEstimation import estimatePoses, estimatePosesFromPivot, estimatePosesFromMultiplePivots, discoverPivot
from src.server.coordinatesTransformation import posesToUnrealCoordinates, posesToUnrealCoordinatesFromPivot
from src.server.databaseFunctions import saveCoordinates
from src.server.kalmanFilter import KalmanFilter
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
    cam = USBVideo.USBCamVideoStream(camIndex=2).start()
else:
    cam = webVideo.ThreadedWebCam().start()

#######################
# KALMAN FILTER SETUP #
#######################

processNoiseCov = 0.005
measurementNoiseCov = 0.1
frameRate = 30
cameraKalmanFilter = KalmanFilter(processNoiseCov, measurementNoiseCov, (1/frameRate))
syringeKalmanFilter = KalmanFilter(processNoiseCov, measurementNoiseCov, (1/frameRate))

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

    print('First request done. Session:')
    print(session._get_current_object())

# On exit (Ctrl + C), we should stop the camera thread so the script stops as expected
def exitHandler(signal, frame):
    cam.stop()
    print('\nCam stopped\n')
    sys.exit(1)

# Session is used to store data between requests
def updateSession(newCoordinates):
    session.permanent = True
    for markerId, markerPose in newCoordinates.items():
        if 'markerPoseRelativeToReference' in session.keys():
            session['markerPoseRelativeToReference'].update({markerId: markerPose})
        else:
            session['markerPoseRelativeToReference'] = {markerId: markerPose}

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

    referencePivotId = request.args.get('reference-pivot', default=9, type=int)
    targetPivotId = request.args.get('target-pivot', default=5, type=int)
    delay = request.args.get('delay', default=0, type=int)

    sleep(delay)

    # Initiliazing the database, knowing that the reference is on the origin
    # origin = {'x':     0,
    #           'y':     0,
    #           'z':     0,
    #           'roll':  0,
    #           'pitch': 0,
    #           'yaw':   0}
    # saveCoordinates(db, str(referencePivotId), origin, 'referencePoseRelativeToPivot', Marker)
    # saveCoordinates(db, str(referencePivotId), origin, 'pivotPoseRelativeToReference', Marker)

    # Getting the reference marker pose, written in pivot coordinates
    # and the pivot pose, written in reference coordinates
    referencePoseRelativeToPivot, pivotPoseRelativeToReference = discoverPivot(targetPivotId, referencePivotId, cameraMatrix, distCoeffs, cam)

    saveCoordinates(db, str(targetPivotId), referencePoseRelativeToPivot, 'referencePoseRelativeToPivot', Marker)
    saveCoordinates(db, str(targetPivotId), pivotPoseRelativeToReference, 'pivotPoseRelativeToReference', Marker)

    # Update the session acordingly
    updateSessionFromDatabase()

    # Return the whole database
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

    context = request.args.get('context', default='test', type=str)

    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))

    poses = estimatePoses(markerIds, cameraMatrix, distCoeffs, cam, camType)

    if '102' in poses.keys():
        syringeKalmanFilter.correct(poses['102'])
        poses['102'] = syringeKalmanFilter.predict()
    else:
        poses['102'] = syringeKalmanFilter.predictForMissingMeasurement()

    unrealCoordinates = posesToUnrealCoordinates(poses, context)

    return jsonify(unrealCoordinates)

@app.route('/pose-from-pivot')
def getCoordinatesFromPivotPerspective():
    session.permanent = True
    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))

    context = request.args.get('context', default='test', type=str)
    pivotMarkerId = request.args.get('pivot', default=9, type=int)
    
    posesFound = estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam, camType)

    if 'hmd' in posesFound.keys():
        cameraKalmanFilter.correct(posesFound['hmd'])
        posesFound['hmd'] = cameraKalmanFilter.predict()
    else:
        posesFound['hmd'] = cameraKalmanFilter.predictForMissingMeasurement()

    # Syringe pose
    if '102' in posesFound.keys():
        syringeKalmanFilter.correct(posesFound['102'])
        posesFound['102'] = syringeKalmanFilter.predict()
    else:
        posesFound['102'] = syringeKalmanFilter.predictForMissingMeasurement()

    unrealCoordinates = posesToUnrealCoordinatesFromPivot(posesFound, context)
    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('markerPoseRelativeToReference', {}))

@app.route('/pose-from-multiple-pivots')
def getPoseFromMultiplePivots():
    session.permanent = True

    referenceId = request.args.get('reference', default=9, type=int)
    context = request.args.get('context', default='test', type=str)

    referencePoseRelativeToPivots = session._get_current_object().get('referencePoseRelativeToPivot')

    pivotIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if OBJECT_DESCRIPTION[k]['objectType'] in ['arm', 'pivot']]))
    markerIds =  list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if OBJECT_DESCRIPTION[k]['objectType'] not in ['arm', 'hmd', 'pivot']]))

    posesFound = estimatePosesFromMultiplePivots(markerIds, pivotIds, referenceId, referencePoseRelativeToPivots, 
                                                 cameraMatrix, distCoeffs, cam=cam)

    # HMD pose -> Filter
    if 'hmd' in posesFound.keys():
        cameraKalmanFilter.correct(posesFound['hmd'])
        posesFound['hmd'] = cameraKalmanFilter.predict()
    else:
        posesFound['hmd'] = cameraKalmanFilter.predictForMissingMeasurement()

    # Syringe pose -> Filter
    if '102' in posesFound.keys():
        syringeKalmanFilter.correct(posesFound['102'])
        posesFound['102'] = syringeKalmanFilter.predict()
    else:
        posesFound['102'] = syringeKalmanFilter.predictForMissingMeasurement()

    updateSession(posesFound)

    knownMarkerPoses = session._get_current_object().get('markerPoseRelativeToReference', {})
    knownPivotPoses = session._get_current_object().get('pivotPoseRelativeToReference', {})

    allPoses = {**knownMarkerPoses, **knownPivotPoses}

    unrealPoses = posesToUnrealCoordinatesFromPivot(allPoses, context)

    return jsonify(unrealPoses)

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