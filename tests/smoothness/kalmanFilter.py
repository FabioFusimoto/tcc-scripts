import cv2.cv2 as cv2
import numpy as np

class KalmanFilter():
    def __init__(self, processNoiseCovMultiplier = 0.03):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * processNoiseCovMultiplier
    
    def correct(self, points):
        self.kalman.correct(points)

    def predict(self):
        return self.kalman.predict()