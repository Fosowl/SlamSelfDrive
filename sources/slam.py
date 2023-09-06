import numpy as np
import cv2 as cv

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

    def estimate_pose(self, twin_points):
        c1 = []
        c2 = []
        for pt1, pt2 in twin_points:
            c1.append(pt1)
            c2.append(pt2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        focal = 1.0
        pp = (self.cx, self.cy) # principal point
        self.E, _ = cv.findEssentialMat(c1, c2, focal, pp, cv.RANSAC, 0.999, 1)
        _, R, t, _ = cv.recoverPose(self.E, c1, c2, self.K, pp)
        self.pose['R'] = R
        self.pose['t'] = t
        return self.E, self.pose

class Vision:
    def __init__(self, video_dim) -> None:
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # pair of points from current and previous frame
        self.twin_points = None
        # last frame used to find matching points
        self.last_frame = None
        # camera class with essential matrix and camera matrix
        self.camera = Camera(video_dim)
        self.frame_delay = 3
    
    def distance_between_points(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def find_matching_points(self, frame):
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame, kps)

        self.frame_delay += 1
        if self.frame_delay >= 3:
            self.last_frame = {'kps': kps, 'des': des}
            self.frame_delay = 0
        matches = self.matcher.match(des, self.last_frame['des'])
        self.twin_points = []
        for m in matches:
            kp1 = kps[m.queryIdx].pt
            kp2 = self.last_frame['kps'][m.trainIdx].pt
            if self.distance_between_points(kp1, kp2) < 50:
                self.twin_points.append((kp1, kp2))
        return self.twin_points

    def view_interest_points(self, frame):
        if self.twin_points is None:
            print("no matches")
            return
        for pt1, pt2 in self.twin_points:
            # from current frame
            cv.circle(frame, (int(pt1[0]), int(pt1[1])), color=(57,204,172), radius=3)
            # from previous frame
            cv.circle(frame, (int(pt2[0]), int(pt2[1])), color=(246,218,8), radius=3)
            # draw line
            cv.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=1)
    
    def get_camera_pose(self):
        if self.twin_points is None:
            return None, None
        return self.camera.estimate_pose(self.twin_points)
    
    def get_matches(self):
        return self.sorted_match

class Slam:
    def __init__(self, width, height) -> None:
        self.vision = Vision((width, height))
        self.past_matrices = dict()
        self.past_matrices['E'] = None
        # pose is a matrix and a vector
        self.past_matrices['pose'] = None
    
    def triangulation(self, points):
        E, pose = self.vision.get_camera_pose()
        if E is None or pose is None:
            return
        if self.past_matrices['E'] is None or self.past_matrices['pose'] is None:
            self.past_matrices['E'] = E
            self.past_matrices['pose'] = pose
            return
        E_diff = self.past_matrices['E'] - E
        print("E_diff : ", E_diff)
        projection_matrix = np.hstack((pose['R'], pose['t']))
        past_projection_matrix = np.hstack((self.past_matrices['pose']['R'], self.past_matrices['pose']['t']))
        projPoints1 = []
        projPoints2 = []
        # Convert matching keypoints into the required format
        for kp1, kp2 in points:
            projPoints1.append([kp1[0], kp1[1]])
            projPoints2.append([kp2[0], kp2[1]])
        # Convert to NumPy arrays
        projPoints1 = np.array(projPoints1).T  # Shape: (2, N)
        projPoints2 = np.array(projPoints2).T  
        points4D = cv.triangulatePoints(past_projection_matrix, projection_matrix, projPoints1, projPoints2)
        points3D = (points4D[:3] / points4D[3]).T
        self.past_matrices['E'] = None
        self.past_matrices['pose'] = None
        return points3D
    
    def view_points(self, frame):
        points = self.vision.find_matching_points(frame)
        if points is not None:
            print("extracted : ", len(points))
        self.vision.view_interest_points(frame)
        self.triangulation(points)
