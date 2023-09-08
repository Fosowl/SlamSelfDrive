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

video = cv.VideoCapture("./videos/forest.mp4")
frame_delay = int(1000 / video.get(cv.CAP_PROP_FPS))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_dim = (width, height)

setting = Setup('Settings')
line = Line(height, width, setting)
slam = Slam(width, height)
detector = Yolo(width, height)
renderer = Renderer3D()
# get K
camera_matrix = slam.get_camera_intrinsics()

points3d = None
x_speed = 0
y_speed = 0
z_speed = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.resize(frame, video_dim, interpolation = cv.INTER_AREA)
    #line.detect_lines(frame)
    #line.overlay(frame)
    #detector.detect_objects(frame)
    #detector.draw_detection_box(frame)
    if renderer.is_freeze() == False:
        matches = slam.match_frame(frame, visualize=True)
    E, cam_pose = slam.get_camera_pose(matches)
    x_speed += cam_pose['t'][0]
    y_speed += cam_pose['t'][1]
    z_speed += cam_pose['t'][2]
    points3d = slam.triangulation(matches)
    if points3d is None:
        continue
    #print("random points :", points3d[random.randint(0, len(points3d))])
    renderer.handle_events()
    renderer.render3dSpace(points3d, cam_pose, camera_matrix)
    renderer.render()
    cv.imshow("Driving", frame)
video.release()
cv.destroyAllWindows()