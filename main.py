import cv2 as cv
import numpy as np
import sys

from sources.yolo import Yolo
from sources.slam import Slam
from sources.render import Renderer3D
from sources.lines import Line
from setup import Setup

video = cv.VideoCapture("./videos/test_nyc.mp4")
frame_delay = int(1000 / video.get(cv.CAP_PROP_FPS))
width = video.get(cv.CAP_PROP_FRAME_WIDTH)
height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

setting = Setup('Settings')
line = Line(height, width, setting)
slam = Slam(width, height)
detector = Yolo(width, height)
renderer = Renderer3D()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    video_dim = (int(frame.shape[1]), int(frame.shape[0]))
    frame = cv.resize(frame, video_dim, interpolation = cv.INTER_AREA)
    #line.detect_lines(frame)
    #line.overlay(frame)
    #detector.detect_objects(frame)
    #detector.draw_detection_box(frame)
    slam.view_points(frame)
    # draw points in 3D space
    points = None
    renderer.render3dSpace(points)
    renderer.render()
    cv.imshow("Driving", frame)
    k = cv.waitKey(frame_delay)
    # quit
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()