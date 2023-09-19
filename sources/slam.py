"""
 implementation of Simultaneous Localization and Mapping (SLAM)
"""

import numpy as np
import math
import cv2 as cv
import copy
import copyreg

# matrices terminology
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Camera:
    def __init__(self, video_dim) -> None:
        self.calibration_frames = [] # not used
        self.cx = video_dim[0] / 2
        self.cy = video_dim[1] / 2
        self.focal = 1.0
        fx = self.focal
        fy = self.focal
        # camera intrinsics (focal length, principal point)
        self.K = np.array([[fx, 0, self.cx],
                           [0, fy, self.cy],
                           [0, 0, 1]])
        # essential matrix (relative position of two cameras)
        self.E = None
        # pose R, t
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))

    def estimate_pose(self, maches_pair):
        """
        estimate camera pose frm matching points and return matrices E, R and vector t
        """
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
        """
        Return camera intrisics matrix K
        """
        return self.K

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)

class Vision:
    def __init__(self, video_dim) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # pair of points from current and previous frame
        self.maches_pair = None
        # last frame used to find matching points
        self.last_tracked_points = None
        # camera class with essential matrix and camera matrix
        self.camera = Camera(video_dim)
        self.feature_limit = 3000
        self.frame_delay = 2
    
    def get_angle_R(self, R) -> (float, float, float):
        """
        Return angle of rotation matrix as angles on axis x,y,z
        """
        assert R is not None
        theta_x = math.atan2(R[2][1], R[2][2]) * (180 / math.pi)
        theta_y = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2)) * (180 / math.pi)
        theta_z = math.atan2(R[1][0], R[0][0]) * (180 / math.pi)
        return (round(theta_x, 2), round(theta_y, 2), round(theta_z, 2))

    def distance_between_points(self, pt1, pt2):
        """
        Return the distance between pt1 and pt2
        """
        assert pt1 is not None and pt2 is not None
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def find_matching_points(self, frame):
        """
        Find and return matching points between the current frame and the previous frame
        """
        assert frame is not None
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=self.feature_limit, qualityLevel=0.01, minDistance=3)
        tracked_points = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        tracked_points, des = self.orb.compute(frame, tracked_points)

        if self.last_tracked_points is None:
            self.last_tracked_points = {'kps': copy.deepcopy(tracked_points), 'des': des.copy()}
        matches = self.matcher.match(des, self.last_tracked_points['des'])
        self.maches_pair = []
        for m in matches:
            kp1 = tracked_points[m.queryIdx].pt
            kp2 = self.last_tracked_points['kps'][m.trainIdx].pt
            if self.distance_between_points(kp1, kp2) < 50:
                self.maches_pair.append((kp1, kp2))
        if self.frame_delay >= 2:
            self.last_tracked_points = {'kps': copy.deepcopy(tracked_points), 'des': des.copy()}
            assert self.last_tracked_points['kps'] != tracked_points
            self.frame_delay = 0
        else:
            self.frame_delay += 1
        return self.maches_pair

    def view_interest_points(self, frame):
        """
        Draw matching points between current and previous frame
        """
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
    
    def get_camera_pose(self, points):
        """
        Estimate the camera pose based on matching points
        """
        if points is None:
            return None, None
        return self.camera.estimate_pose(points)
    
    def get_matches(self):
        """
        Get matches sorted
        """
        return self.sorted_match

class Slam:
    def __init__(self, width, height) -> None:
        self.vision = Vision((width, height))
        self.past_matrices = dict()
        self.past_matrices['E'] = None
        # pose is a matrix R and a vector t
        self.past_matrices['pose'] = None
        self.speeds_history = [[0, 0, 0]]
        self.position = [0, 0, 0]
    
    def get_camera_intrinsics(self):
        return self.vision.camera.get_camera_intrinsics()
    
    def get_camera_pose(self, points):
        assert points is not None
        return self.vision.get_camera_pose(points)
    
    def get_position(self):
        """
        estimate position in x,y,z
        """
        return self.position
    
    def update_speed(self):
        """
        estimate average speed on axis x,y,z using vector t
        """
        if self.past_matrices['E'] is None or self.past_matrices['pose'] is None:
            return (0, 0, 0)
        pose = self.past_matrices['pose']
        t = pose['t']
        self.speeds_history.append([-t[0], t[1], t[2]])
        x_sum = sum([x[0] for x in self.speeds_history])
        y_sum = sum([x[1] for x in self.speeds_history])
        z_sum = sum([x[2] for x in self.speeds_history])
        self.position = np.array([round(x_sum[0], 2), round(y_sum[0], 2), round(z_sum[0], 2)])

    def triangulation(self, points):
        """
        get 3D points in space from 2D points in images
        """
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
        points4D = cv.triangulatePoints(P1, P2, frame_points1, frame_points2).T
        points4D /= points4D[:, 3:]
        goods = (np.abs(points4D[:, 3]) > 0.005) & (points4D[:, 2] > 0)
        points3D = []
        for i, p in enumerate(points4D):
            if goods[i]:
                points3D.append([p[0], p[1], p[2]])
        self.update_speed()
        self.past_matrices['E'] = None
        self.past_matrices['pose'] = None
        return points3D
    
    def match_frame(self, frame, visualize=False):
        assert frame is not None
        matches_pair = self.vision.find_matching_points(frame)
        if visualize:
            self.vision.view_interest_points(frame)
        return matches_pair
