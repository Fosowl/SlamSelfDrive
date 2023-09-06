
import cv2 as cv
import numpy as np
import sys

class area:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        self.top_right = (int(top_right[0]), int(top_right[1]))

    def debug_lines(self, line_image):
        cv.line(line_image, (self.bottom_left[0], self.bottom_left[1]), (self.top_right[0], self.bottom_left[1]), (255, 0, 255), 1)
        cv.line(line_image, (self.bottom_left[0], self.bottom_left[1]), (self.bottom_left[0], self.top_right[1]), (255, 0, 255), 1)
        cv.line(line_image, (self.top_right[0], self.top_right[1]), (self.top_right[0], self.bottom_left[1]), (255, 0, 255), 1)
        cv.line(line_image, (self.top_right[0], self.top_right[1]), (self.bottom_left[0], self.top_right[1]), (255, 0, 255), 1)
    
    def check_within_bounds(self, point):
        if point[0] >= self.bottom_left[0] and point[0] <= self.top_right[0] and point[1] <= self.bottom_left[1] and point[1] >= self.top_right[1]:
            return True
        return False

class Line:
    def __init__(self, height, width, settings, white_value=255, padding=50) -> None:
        self.lines = None
        self.vision = area((padding, height-padding), (width-padding, height*0.5+75))
        self._white_value = white_value
        self._settings = settings

    def detect_lines(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(gray, self._settings.whitenest, self._white_value, cv.THRESH_BINARY)
        edges = cv.Canny(threshold, 100, 200)
        self.lines = cv.HoughLinesP(edges, rho=self._settings.rho, theta=np.pi/self._settings.theta_div, threshold=self._settings.threshold, minLineLength=self._settings.minLineLength, maxLineGap=self._settings.maxLineGap)
        return self.lines

    def overlay(self, frame):
        line_image = np.zeros_like(frame)
        if self.lines is None:
            return
        self.vision.debug_lines(line_image)
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            if self.vision.check_within_bounds((x1, y1)) == False or self.vision.check_within_bounds((x2, y2)) == False:
                continue
            cv.line(line_image, (x1, y1), (x2, y2), (0, self._white_value, 0), 5)
        frame = cv.addWeighted(frame, 0.9, line_image, 0.7, 0)
        return frame