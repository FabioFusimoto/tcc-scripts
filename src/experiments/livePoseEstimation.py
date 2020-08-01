import cv2.cv2 as cv2
import numpy as np
from timeit import default_timer as timer
from threading import Thread

import src.calibration.arucoMarkers as arucoMarkers
import src.calibration.chessboard as chess
import src.webcamUtilities.video as video
from src.calibration.commons import calculateCoordinates

class USBCamVideoStream:
    def __init__(self, camIndex=0):
        self.stream = cv2.VideoCapture(camIndex)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

def videoPoseEstimation(markerIds, markerLength, calibrationFile, frameCount):
    cameraMatrix, distCoeffs = chess.loadCalibrationCoeficients(calibrationFile)
    cam = USBCamVideoStream(camIndex=0).start()

    timeToReadFrames = 0
    timeToEstimatePoses = 0

    for _ in range(frameCount):
        timerBeforeFrame = timer()

        image = cam.read()

        timerAfterFrame = timer()
        
        ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, markerLength, cameraMatrix, distCoeffs)

        estimatedPoses = {}
        for markerId in markerIds:
            indexes = np.where(ids == markerId)[0]
            if indexes.size > 0:
                i = indexes[0]
                coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), arucoMarkers.getRFlip())
                estimatedPoses[markerId] = coords
        
        timerAfterPoseEstimation = timer()

        timeToReadFrames += (timerAfterFrame - timerBeforeFrame)
        timeToEstimatePoses += (timerAfterPoseEstimation - timerAfterFrame)

    print('\nTime elapsed on reading frames: ' + str(timeToReadFrames) + 's')
    print('Time elapsed on estimating pose: ' + str(timeToEstimatePoses) + 's')
    
    cam.stop()

calibrationFile = 'tests/calibration-coefficients/g7-play-X-75-percent-resolution.yml'
frameCount = 100

start = timer()

videoPoseEstimation([0,1,3], 3.78, calibrationFile, frameCount)

end = timer()

timeElapsed = end - start
print('\nTotal time elapsed: ' + str(timeElapsed) + 's')
print('Equivalent to: ' + str(frameCount/timeElapsed) + ' frames per second')