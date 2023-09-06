
import numpy as np
import cv2 as cv

class Setup:
    def __init__(self, window_name, panel=False):
        self._whitenest = 185
        self._rho = 6
        self._theta_div=90
        self._threshold=50
        self._minLineLength=50
        self._maxLineGap=15
        cv.namedWindow(window_name)
        if panel:
            cv.createTrackbar('Threshold', window_name, self._whitenest, 255, self.set_whitenest)
            cv.createTrackbar('minLineLength', window_name, self._minLineLength, 100, self.set_minLineLength)
            cv.createTrackbar('maxLineGap', window_name, self._maxLineGap, 100, self.set_maxLineGap)
    
    # GETTER

    @property
    def whitenest(self):
        return self._whitenest

    @property
    def rho(self):
        return self._rho

    @property
    def theta_div(self):
        return self._theta_div
    
    @property
    def threshold(self):
        return self._threshold
    
    @property
    def minLineLength(self):
        return self._minLineLength
    
    @property
    def maxLineGap(self):
        return self._maxLineGap
    
    # SETTER 

    @whitenest.setter
    def set_whitenest(self, val):
        if val < 0:
            raise ValueError("whitenest must be positive")
        self.whitenest = val

    @minLineLength.setter
    def set_minLineLength(self, val):
        if val < 0:
            raise ValueError("min_line must be positive")
        self._minLineLength = val
    
    @maxLineGap.setter
    def set_maxLineGap(self, val):
        if val < 0:
            raise ValueError("max_gap must be positive")
        self._maxLineGap = val
