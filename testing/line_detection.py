
import cv2 as cv
import numpy as np
from setup import setup
import sys
import cv2 as cv
import numpy as np
import sys

from setup import setup

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

def show_line(frame, height, width, setting):
    setting.show()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray, setting.whitenest, 255, cv.THRESH_BINARY)
    edges = cv.Canny(threshold, 100, 200)
    lines = cv.HoughLinesP(edges, rho=setting.rho, theta=np.pi/setting.theta_div, threshold=setting.threshold, minLineLength=setting.minLineLength, maxLineGap=setting.maxLineGap)
    line_image = np.zeros_like(frame)
    if lines is None:
        return frame
    vision = area((50, height-50), (width-50, height*0.5+75))
    vision.debug_lines(line_image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if vision.check_within_bounds((x1, y1)) == False or vision.check_within_bounds((x2, y2)) == False:
            continue
        cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    final_image = cv.addWeighted(frame, 0.9, line_image, 0.7, 0)
    frame = final_image
    return final_image

video = cv.VideoCapture("./videos/test_nyc.mp4")
frame_delay = int(1000 / video.get(cv.CAP_PROP_FPS))
width = video.get(cv.CAP_PROP_FRAME_WIDTH)
height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

setting = setup('Settings')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    frame = show_line(frame, height, width, setting)
    cv.imshow("Driving", frame)
    k = cv.waitKey(frame_delay)
    # quit
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()