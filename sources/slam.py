"""
 implementation of Simultaneous Localization and Mapping (SLAM)
"""

import numpy as np
import cv2 as cv
import copy

# matrices terminology
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Camera:
    def __init__(self, video_dim) -> None:
        self.calibration_frames = [] # not used
        self.cx = video_dim[0] / 2
        self.cy = video_dim[1] / 2
        # camera intrinsics (focal length, principal point)
        self.K = None
        # essential matrix (relative position of two cameras)
        self.E = None
        self.focal = 1.0
        fx = self.focal
        fy = self.focal
        self.K = np.array([[fx, 0, self.cx],
                           [0, fy, self.cy],
                           [0, 0, 1]])
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))

    # estimate camera pose and return E and pose (R, t)
    def estimate_pose(self, maches_pair):
        assert maches_pair is not None
        c1 = []
        c2 = []
        for pt1, pt2 in maches_pair:
            c1.append(pt1)
            c2.append(pt2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        focal = 1.0
        pp = (self.cx, self.cy) # principal point
        self.E, _ = cv.findEssentialMat(c1, c2, focal, pp, cv.RANSAC, 0.99, 1)
        _, R, t, _ = cv.recoverPose(self.E, c1, c2, self.K, pp)
        self.pose['R'] = R
        self.pose['t'] = t
        return self.E, self.pose
    
    def get_camera_intrinsics(self):
        return self.K

class Vision:
    def __init__(self, video_dim) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # pair of points from current and previous frame
        self.maches_pair = None
        # last frame used to find matching points
        self.last_frame = None
        # camera class with essential matrix and camera matrix
        self.camera = Camera(video_dim)
        self.frame_delay = 0
    
    # distance between two points
    def distance_between_points(self, pt1, pt2):
        assert pt1 is not None and pt2 is not None
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    # find matching points between current and previous frame
    def find_matching_points(self, frame):
        assert frame is not None
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=1000, qualityLevel=0.01, minDistance=3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame, kps)

        self.frame_delay += 1
        if self.frame_delay > 0:
            self.last_frame = {'kps': kps, 'des': des}
            self.frame_delay = 0
        matches = self.matcher.match(des, self.last_frame['des'])
        self.maches_pair = []
        for m in matches:
            kp1 = kps[m.queryIdx].pt
            kp2 = self.last_frame['kps'][m.trainIdx].pt
            if self.distance_between_points(kp1, kp2) < 50:
                self.maches_pair.append((kp1, kp2))
        return self.maches_pair

    # draw points and lines between current and previous frame
    def view_interest_points(self, frame):
        if self.maches_pair is None:
            print("no matches")
            return
        for pt1, pt2 in self.maches_pair:
            # from current frame
            cv.circle(frame, (int(pt1[0]), int(pt1[1])), color=(57,204,172), radius=3)
            # from previous frame
            cv.circle(frame, (int(pt2[0]), int(pt2[1])), color=(246,218,8), radius=3)
            # draw line
            cv.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=1)
    
    # return camera pose
    def get_camera_pose(self, points):
        if points is None:
            return None, None
        return self.camera.estimate_pose(points)
    
    def get_matches(self):
        return self.sorted_match

class Slam:
    def __init__(self, width, height) -> None:
        self.vision = Vision((width, height))
        self.past_matrices = dict()
        self.past_matrices['E'] = None
        # pose is a matrix R and a vector t
        self.past_matrices['pose'] = None
        self.speeds_history = [[0, 0, 0]]
    
    def get_camera_intrinsics(self):
        return self.vision.camera.get_camera_intrinsics()
    
    # get camera pose
    def get_camera_pose(self, points):
        assert points is not None
        return self.vision.get_camera_pose(points)
    
    # get speed of camera
    def get_speed(self):
        if self.past_matrices['E'] is None or self.past_matrices['pose'] is None:
            return [0, 0, 0]
        pose = self.past_matrices['pose']
        t = pose['t']
        self.speeds_history.append([t[0], t[1], t[2]])
        x_sum = sum([x[0] for x in self.speeds_history])
        y_sum = sum([x[1] for x in self.speeds_history])
        z_sum = sum([x[2] for x in self.speeds_history])
        x_avg = x_sum / len(self.speeds_history)
        y_avg = y_sum / len(self.speeds_history)
        z_avg = z_sum / len(self.speeds_history)
        return (round(x_avg[0], 2), round(y_avg[0], 2), round(z_avg[0], 2))

    
    # get 3D points in space from 2D points in images
    def triangulation(self, points):
        assert points is not None
        E, pose = self.vision.get_camera_pose(points)
        K = self.vision.camera.get_camera_intrinsics()
        if E is None or pose is None:
            return None
        if self.past_matrices['E'] is None or self.past_matrices['pose'] is None:
            self.past_matrices['E'] = copy.deepcopy(E)
            self.past_matrices['pose'] = copy.deepcopy(pose) 
            return None
        P1 = np.dot(K, np.hstack((self.past_matrices['pose']['R'], self.past_matrices['pose']['t'])))
        P2 = np.dot(K, np.hstack((pose['R'], pose['t'])))
        frame_points1 = []
        frame_points2 = []
        for pt1, pt2 in points:
            frame_points1.append([pt1[0], pt1[1]])
            frame_points2.append([pt2[0], pt2[1]])
        frame_points1 = np.array(frame_points1).T  # T cause shape (2, N)
        frame_points2 = np.array(frame_points2).T  
        points4D = cv.triangulatePoints(P1, P2, frame_points1, frame_points2)
        points3D = (points4D[:3] / points4D[3]).T
        self.past_matrices['E'] = None
        self.past_matrices['pose'] = None
        return points3D
    
    def match_frame(self, frame, visualize=False):
        assert frame is not None
        matches_pair = self.vision.find_matching_points(frame)
        if visualize:
            self.vision.view_interest_points(frame)
        return matches_pair
