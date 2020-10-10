import csv
import cv2.cv2 as cv2
import numpy as np

import src.calibration.arucoMarkers as arucoMarkers
from src.calibration.commons import calculateCoordinates

def markerPose(markerId, cameraMatrix, distCoeffs, imageFile):
    image = cv2.imread(imageFile)

    ids, rVecs, tVecs = arucoMarkers.getPositionVectors(image, 1, cameraMatrix, distCoeffs)

    indexes = np.where(ids == markerId)[0]

    if indexes.size > 0:
        i = indexes[0]
        coords = calculateCoordinates(np.reshape(rVecs[i], (3,1)), np.reshape(tVecs[i], (3,1)), scale=5.3)
        return coords
    else:
        return {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}

def exportResultsToFile(outputFile, results):
    header = ['Valor esperado', 'Amostra 1', 'Amostra 2', 'Amostra 3', 'Erro medio absoluto', 'Erro medio relativo (%)']

    rows = [header]

    for expectedValue, samples in results.items():
        sampleAverage = sum(samples) / len(samples)
        averageError = abs(sampleAverage - expectedValue)
        relativeErrorPercentage = abs(averageError/expectedValue) * 100

        rows.append(['{:.2f}'.format(expectedValue), '{:.2f}'.format(samples[0]), '{:.2f}'.format(samples[1]), 
                     '{:.2f}'.format(samples[2]), '{:.2f}'.format(averageError), '{:.2f}'.format(relativeErrorPercentage)])

    with open(outputFile, mode='w+', newline='') as resultsCSV:
        writer = csv.writer(resultsCSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)