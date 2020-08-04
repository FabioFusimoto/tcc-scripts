from flask import Flask, jsonify, Response, session
import pprint
import signal
import sys
from time import sleep

from src.calibration.commons import loadCalibrationCoefficients
import src.webcamUtilities.video as webVideo
import src.USBCam.video as USBVideo
from src.server.coordinatesEstimation import estimatePoses
from src.server.coordinatesTransformation import posesToUnrealCoordinates
from src.server.objects import MARKER_LENGTH, MARKER_TO_OBJECT

app = Flask(__name__)
app.config['SECRET_KEY'] = b'SECRET_KEY'

## Camera parameters and configuration
calibrationFile = 'src/server/files/g7-play-1280x720.yml'
cameraMatrix, distCoeffs = loadCalibrationCoefficients(calibrationFile)

camType = 'USB'

cam = {}

if camType == 'USB':
    cam = USBVideo.USBCamVideoStream().start()
else:
    cam = webVideo.ThreadedWebCam().start()

@app.route('/')
def home():
    return 'Homepage'

@app.route('/pose')
def getCoordinates():
    session.permanent = True
    markerIds = list(map(int, MARKER_TO_OBJECT.keys()))

    poses = estimatePoses(markerIds, MARKER_LENGTH, cameraMatrix, distCoeffs, cam, camType)
    unrealCoordinates = posesToUnrealCoordinates(poses)

    for markerId, data in unrealCoordinates.items():
        if data['found'] == True:
            if 'poses' in session.keys():
                session['poses'].update({markerId: data['coordinates']})
            else:
                session['poses'] = {markerId: data['coordinates']}

    return jsonify(session._get_current_object().get('poses', {}))

@app.route('/counter')
def counter():
    def streamer():
        count = 0
        while True:
            count += 1
            yield "Count: {}\n".format(count)
            sleep(1/60)
    return Response(streamer())

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