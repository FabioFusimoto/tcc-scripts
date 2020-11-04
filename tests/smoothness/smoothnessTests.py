import numpy as np
import pprint
import traceback
import time

from src.calibration.commons import loadCalibrationCoefficients
import src.USBCam.video as USBVideo

from tests.smoothness.helpers import getCameraPosition, plotPoints, plotMultiple
from tests.smoothness.kalmanFilter import KalmanFilter

calibrationFile = 'tests/calibration-coefficients/J7-pro-720p.yml'
cameraMatrix, distCoeffs = loadCalibrationCoefficients(calibrationFile)

cam = USBVideo.USBCamVideoStream(camIndex=1).start()

frameRate = 30
invertedFrameRate = 1/frameRate

def smoothnessGraph(markerId, sampleCount, processNoiseCovs, measurementNoiseCovs, kalmanArray):
    startedAt = time.perf_counter()
    cameraPositions = np.empty((1, 6), dtype=np.float32)
    filteredCameraPositions = {}

    for processNoise, measurementNoise in zip(processNoiseCovs, measurementNoiseCovs):
        filteredCameraPositions[(processNoise, measurementNoise)] = np.array([], dtype=np.float32)

    intervals = []
    for _ in range(sampleCount):
        processStart = time.perf_counter()
        newPosition = getCameraPosition(cam, markerId, cameraMatrix, distCoeffs)

        cameraPositions = np.append(cameraPositions, newPosition, axis=0)

        for processNoise, measurementNoise, kalman in zip(processNoiseCovs, measurementNoiseCovs, kalmanArray):
            if (not np.isnan(newPosition[0]).all()):
                kalman.correct(newPosition[0])                
            prediction = kalman.predict()
            filteredCameraPositions[(processNoise, measurementNoise)] = np.append(filteredCameraPositions[(processNoise, measurementNoise)], 
                                                                                  prediction[0])
            

        intervals.append(1000*(time.perf_counter() - startedAt))

        processEnd = time.perf_counter()
        timeElapsed = processEnd - processStart

        time.sleep(invertedFrameRate - timeElapsed)

    cameraPositions = np.delete(cameraPositions, 0, 0)

    for processNoise, measurementNoise in zip(processNoiseCovs, measurementNoiseCovs):
        plotMultiple(intervals,
                     cameraPositions.T[0],
                     'Estimativa sem filtro',
                     filteredCameraPositions[(processNoise, measurementNoise)],
                     'Estimativa com filtro - PNC = ' + str(processNoise) + '| MNC = ' + str(measurementNoise),
                     'Tempo (ms)',
                     'x (cm)')

processNoiseCovMultipliers = [0.005]
measurementNoiseCovMultipliers = [0.1]
kalmanFilterArray = [KalmanFilter(processNoise, measurementNoise, invertedFrameRate) 
                     for processNoise, measurementNoise
                     in zip(processNoiseCovMultipliers, measurementNoiseCovMultipliers)]

try:
    smoothnessGraph(3, 200, processNoiseCovMultipliers, measurementNoiseCovMultipliers, kalmanFilterArray)
except Exception as e:
    print('\nThe following exception occurred while sampling')
    print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
finally:
    cam.stop()