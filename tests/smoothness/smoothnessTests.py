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

def smoothnessGraph(markerId, sampleCount, noiseMultipliers, kalmanArray):
    startedAt = time.perf_counter()
    cameraPositions = []
    filteredCameraPositions = {}

    for n in noiseMultipliers:
        filteredCameraPositions[n] = np.array([], dtype=np.float32)

    intervals = []
    for _ in range(sampleCount):
        newPosition = getCameraPosition(cam, markerId, cameraMatrix, distCoeffs)
        cameraPositions.append(newPosition)

        for n, k in zip(noiseMultipliers,kalmanArray):
            k.correct(np.array([np.float32(newPosition[1]['x']), np.float32(newPosition[1]['y'])], np.float32))
            prediction = k.predict()
            filteredCameraPositions[n] = np.append(filteredCameraPositions[n], prediction.item((0, 0)))

        intervals.append(1000*(time.perf_counter() - startedAt))
        time.sleep(1/60)

    for n in noiseMultipliers:
        plotMultiple(intervals,
                     [cp[1]['x'] if cp[0] else 0.0 for cp in cameraPositions],
                     'Estimativa sem filtro',
                     filteredCameraPositions[n],
                     'Estimativa com filtro - PNC = ' + str(n),
                     'Tempo (ms)',
                     'x (cm)')

processNoiseCovMultipliers = [0.0001]
kalmanFilterArray = [KalmanFilter(noise) for noise in processNoiseCovMultipliers]

try:
    smoothnessGraph(3, 400, processNoiseCovMultipliers, kalmanFilterArray)
except Exception as e:
    print('\nThe following exception occurred while sampling')
    print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
finally:
    cam.stop()