import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

from sources.yolo import Yolo
from sources.slam import Slam
from sources.render import Renderer3D
from sources.lines import Line
from setup import Setup
import random
import time

video = cv.VideoCapture("./videos/test_nyc.mp4")
frame_delay = int(1000 / video.get(cv.CAP_PROP_FPS))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_dim = (width, height)

setting = Setup('Settings')
#line = Line(height, width, setting)
#detector = Yolo(width, height)
slam = Slam(width, height)
renderer = Renderer3D()
# get K
camera_matrix = slam.get_camera_intrinsics()

points3d = None
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.resize(frame, video_dim, interpolation = cv.INTER_AREA)
    if renderer.is_freeze() == False:
        matches = slam.match_frame(frame, visualize=True)
    if matches is None:
        continue
    E, cam_pose = slam.get_camera_pose(matches)
    points3d = slam.triangulation(matches)
    if points3d is None:
        continue
    # TODO identical point problem solved but still not working but still not working
    renderer.handle_camera()
    renderer.draw_axes()
    renderer.render3dSpace(points3d, slam.get_position())
    renderer.render(10)
    cv.imshow("Driving", frame)
video.release()
cv.destroyAllWindows()