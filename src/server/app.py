from flask import Flask, jsonify, Response, request, session
import logging
import math
import numpy as np
import pprint
import signal
import sys
from time import sleep, time
from tinydb import TinyDB, Query

from src.calibration.commons import loadCalibrationCoefficients
import src.webcamUtilities.video as webVideo
import src.USBCam.video as USBVideo
from src.server.coordinatesEstimation import estimatePoses, estimatePosesFromPivot, estimatePosesFromMultiplePivots, discoverPivot
from src.server.coordinatesTransformation import posesToUnrealCoordinates, posesToUnrealCoordinatesFromPivot
from src.server.databaseFunctions import saveCoordinates, fetchPivotPoses, clearPivotsFromDatabase, fetchPoseWithTimestamp, clearGraphicsPointsFromDatabase
from src.server.kalmanFilter import KalmanFilter
from src.server.objects import OBJECT_DESCRIPTION
from src.server.utils import livePoseEstimation, plotPoints

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

# On exit (Ctrl + C), we should stop the camera thread so the script stops as expected
def exitHandler(signal, frame):
    cam.stop()
    print('\nCam stopped\n')
    sys.exit(1)

#####################
# SESSION FUNCTIONS #
#####################
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
    posesOnDatabase = fetchPivotPoses(db, Marker)
    for relativePose in posesOnDatabase:
        if ('markerId' in relativePose.keys()) and ('relation' in relativePose.keys()): 
            markerId = relativePose['markerId']
            relation = relativePose['relation']
            relativePose.pop('markerId')
            relativePose.pop('relation')
            if relation in session.keys():
                session[relation].update({markerId: relativePose})
            else:
                session[relation] = {markerId: relativePose}

def saveGraphicsPointsToSession(pivotId, hmdPose):
    if 'poseWithTimestamp' not in session.keys():
        session['poseWithTimestamp'] = []
    session['poseWithTimestamp'].append({
        'type': 'poseWithTimestamp',
        'timestamp': round(time() * 1000),
        'markerId': pivotId,
        'pose': hmdPose
    })

######################
# DATABASE FUNCTIONS #
######################
def saveGraphicsPointsToDatabase():
    session.permanent = True
    clearGraphicsPointsFromDatabase(db, Marker)

    for sample in session._get_current_object().get('poseWithTimestamp', []):
        db.insert(sample)

    return len(session._get_current_object().get('poseWithTimestamp', []))

def clearGraphicsPoints():
    session.permanent = True
    clearGraphicsPointsFromDatabase(db, Marker)

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
    targetPivotIds = request.args.get('target-pivots', default=5, type=str)
    delay = request.args.get('delay', default=0, type=int)

    targetPivotIds = targetPivotIds.split(',')

    sleep(delay)

    # Getting the reference marker pose, written in pivot coordinates
    # and the pivot pose, written in reference coordinates
    for targetPivotId in targetPivotIds:
        referencePoseRelativeToPivot, pivotPoseRelativeToReference = discoverPivot(targetPivotId, referencePivotId, cameraMatrix, distCoeffs, cam)

        saveCoordinates(db, str(targetPivotId), referencePoseRelativeToPivot, 'referencePoseRelativeToPivot', Marker)
        saveCoordinates(db, str(targetPivotId), pivotPoseRelativeToReference, 'pivotPoseRelativeToReference', Marker)

    # Update the session acordingly
    updateSessionFromDatabase()

    # Return all known marker poses relative to reference
    updatedPivotCoords = fetchPivotPoses(db, Marker)

    for c in updatedPivotCoords:
        c['roll'] *= 180/np.pi
        c['pitch'] *= 180/np.pi
        c['yaw'] *= 180/np.pi

    return jsonify(updatedPivotCoords)

@app.route('/clear-pivots')
def clearPivots():
    session.permanent = True
    clearPivotsFromDatabase(db, Marker)
    return 'Pivots cleared'

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
    frameNotFoundThreshold = 10

    markerIds = list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if k != 'hmd']))

    context = request.args.get('context', default='test', type=str)
    pivotMarkerId = request.args.get('pivot', default=9, type=int)
    
    posesFound = estimatePosesFromPivot(markerIds, pivotMarkerId, cameraMatrix, distCoeffs, cam, camType)

    # HMD pose -> filter
    if 'hmd' in posesFound.keys():
        cameraKalmanFilter.correct(posesFound['hmd'])
        posesFound['hmd'] = cameraKalmanFilter.predict()
    else:
        if cameraKalmanFilter.frameNotFoundCount <= frameNotFoundThreshold:
            posesFound['hmd'] = cameraKalmanFilter.predictForMissingMeasurement()

    # Syringe pose -> filter
    if '102' in posesFound.keys():
        syringeKalmanFilter.correct(posesFound['102'])
        posesFound['102'] = syringeKalmanFilter.predict()
    else:
        if syringeKalmanFilter.frameNotFoundCount <= frameNotFoundThreshold:
            posesFound['102'] = syringeKalmanFilter.predictForMissingMeasurement()

    unrealCoordinates = posesToUnrealCoordinatesFromPivot(posesFound, context)
    updateSession(unrealCoordinates)

    return jsonify(session._get_current_object().get('markerPoseRelativeToReference', {}))

@app.route('/pose-from-multiple-pivots')
def getPoseFromMultiplePivots():
    session.permanent = True
    frameNotFoundThreshold = 10

    referenceId = request.args.get('reference', default=9, type=int)
    context = request.args.get('context', default='test', type=str)
    pivotToSave = request.args.get('pivot_to_save', default='', type=str)

    referencePoseRelativeToPivots = session._get_current_object().get('referencePoseRelativeToPivot')

    pivotIds = [9, 101, 103, 104, 105, 106, 107]
    markerIds =  list(map(int, [k for k in OBJECT_DESCRIPTION.keys() if OBJECT_DESCRIPTION[k]['objectType'] not in ['arm', 'hmd', 'pivot']]))

    knownMarkerPoses = session._get_current_object().get('markerPoseRelativeToReference', {})
    knownPivotPoses = session._get_current_object().get('pivotPoseRelativeToReference', {})
    lastKnownPoses = {**knownMarkerPoses, **knownPivotPoses}
    targetPivotId, posesFound = estimatePosesFromMultiplePivots(markerIds, pivotIds, referenceId, referencePoseRelativeToPivots,
                                                                cameraMatrix, distCoeffs, lastKnownPoses=lastKnownPoses, cam=cam)

    if str(targetPivotId) == pivotToSave:
        saveGraphicsPointsToSession(pivotToSave, posesFound['hmd'])

    # HMD pose -> filter
    if 'hmd' in posesFound.keys():
        cameraKalmanFilter.correct(posesFound['hmd'])
        posesFound['hmd'] = cameraKalmanFilter.predict()
    else:
        if cameraKalmanFilter.frameNotFoundCount <= frameNotFoundThreshold:
            posesFound['hmd'] = cameraKalmanFilter.predictForMissingMeasurement()

    # Syringe pose -> filter
    if '102' in posesFound.keys():
        syringeKalmanFilter.correct(posesFound['102'])
        posesFound['102'] = syringeKalmanFilter.predict()
    else:
        if syringeKalmanFilter.frameNotFoundCount <= frameNotFoundThreshold:
            posesFound['102'] = syringeKalmanFilter.predictForMissingMeasurement()

    updateSession(posesFound)

    knownMarkerPoses = session._get_current_object().get('markerPoseRelativeToReference', {})
    lastKnownPoses = {**knownMarkerPoses, **knownPivotPoses}

    unrealPoses = posesToUnrealCoordinatesFromPivot(lastKnownPoses, context)

    return jsonify({**unrealPoses, **{'target_pivot_id': str(targetPivotId)}})

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

###########################
# GRAPHICS RELATED ROUTES #
###########################
@app.route('/save-graphics-points-from-session-to-database')
def saveGraphicsPointsFromSessionToDatabase():
    sampleCount = saveGraphicsPointsToDatabase()

    return 'Session saved to database. Sample count: ' + str(sampleCount)

@app.route('/generate-graphic')
def generateGraphic():
    markerId = request.args.get('marker', default=9, type=int)
    coordinate = request.args.get('coordinate', default='x', type=str)

    poseWithTimestamps = fetchPoseWithTimestamp(db, markerId, Marker)

    timestamps = [p['timestamp'] for p in poseWithTimestamps]
    samples = [p['pose'][coordinate] for p in poseWithTimestamps]

    plotPoints(timestamps, samples, 'Tempo (ms)', coordinate)

    return 'OK'

@app.route('/clear-graphics-points')
def deleteGraphicsPoints():
    clearGraphicsPoints()
    return 'Graphics points cleared'

####################################
# PARAMETERS FOR EXECUTING THE APP #
####################################
if __name__ == "__main__":
    signal.signal(signal.SIGINT, exitHandler)
    app.run('127.0.0.1',port=5000)