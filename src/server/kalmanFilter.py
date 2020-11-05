import cv2.cv2 as cv2
import numpy as np
import pprint

class KalmanFilter():
    def __init__(self, processNoiseCovMultiplier = 0.005, measurementNoiseCovMultiplier = 0.1, dT = 1):
        self.kalman = cv2.KalmanFilter(18, 6, 0)
        dT2 = (1/2) * (dT**2)

        self.kalman.transitionMatrix = np.array([
            [1,   0,   0,  dT,   0,   0, dT2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   1,   0,   0,  dT,   0,   0, dT2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   1,   0,   0,  dT,   0,   0, dT2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   1,   0,   0,  dT,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   1,   0,   0,  dT,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   0,   1,   0,   0,  dT,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT,   0,   0, dT2,   0,   0], 
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT,   0,   0, dT2,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT,   0,   0, dT2],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT,   0,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,  dT],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0],
            [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1]
        ], dtype=np.float32)

        # Process Noise Covariance = Q
        self.kalman.processNoiseCov = cv2.setIdentity(self.kalman.processNoiseCov, processNoiseCovMultiplier)

        # Measurement Matrix = H
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        # Measurement Noise Covariance = R
        self.kalman.measurementNoiseCov = cv2.setIdentity(self.kalman.measurementNoiseCov, measurementNoiseCovMultiplier)
    
    def printMatrixes(self):
        print('\nTransition Matrix (A)')
        print(self.kalman.transitionMatrix)

        print('\nProcess Noise Covariance (Q)')
        print(self.kalman.processNoiseCov)

        print('\nMeasurement Matrix (H)')
        print(self.kalman.measurementMatrix)

        print('\nMeasurement Noise Covariance (R)')
        print(self.kalman.measurementNoiseCov)

    def correct(self, dictOfPoints):
        pointsAsArray = np.array([
            dictOfPoints['x'],
            dictOfPoints['y'],
            dictOfPoints['z'],
            dictOfPoints['roll'],
            dictOfPoints['pitch'],
            dictOfPoints['yaw']
        ], dtype=np.float32)
        self.kalman.correct(pointsAsArray)

    def predict(self):
        predictionAsArray = self.kalman.predict()
        return {
            'x': predictionAsArray.item((0,0)),
            'y': predictionAsArray.item((1,0)),
            'z': predictionAsArray.item((2,0)),
            'roll': predictionAsArray.item((9,0)),
            'pitch': predictionAsArray.item((10,0)),
            'yaw': predictionAsArray.item((11,0))
        }

    def predictForMissingMeasurement(self):
        measurementNoiseCov = self.kalman.measurementNoiseCov.copy()

        highNoiseCov = cv2.setIdentity(self.kalman.measurementNoiseCov, 1e10)
        setattr(self.kalman, 'measurementNoiseCov', highNoiseCov)

        predictionAsArray = self.kalman.predict()
        predictionAsDict = {
            'x': predictionAsArray.item((0,0)),  
            'y': predictionAsArray.item((1,0)),  
            'z': predictionAsArray.item((2,0)),  
            'roll': predictionAsArray.item((9,0)),  
            'pitch': predictionAsArray.item((10,0)), 
            'yaw': predictionAsArray.item((11,0))  
        }

        setattr(self.kalman, 'measurementNoiseCov', measurementNoiseCov)

        return predictionAsDict