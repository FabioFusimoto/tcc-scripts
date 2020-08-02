from flask import Flask, Response, jsonify
from time import sleep

from src.calibration.commons import loadCalibrationCoefficients
import src.webcamUtilities.video as webVideo
import src.USBCam.video as USBVideo
from src.server.coordinatesEstimation import estimatePoses

app = Flask(__name__)

calibrationFile = 'files/g7-play-1280x720-landscape.yml'
markerIds = [0, 1, 3]
markerLength = 3.78
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
    poses = estimatePoses(markerIds, markerLength, cameraMatrix, distCoeffs, cam, camType)
    return jsonify(poses)

@app.route('/pose-stream')
def getCoordinatesStream():
    pose = estimatePoses(markerIds, markerLength, cameraMatrix, distCoeffs, cam, camType)

    def streamer():
        while True:
            yield str(pose)

    return Response(streamer())

@app.route('/counter')
def counter():
    def streamer():
        count = 0
        while True:
            count += 1
            yield "Count: {}\n".format(count)
            sleep(1/60)
    return Response(streamer())