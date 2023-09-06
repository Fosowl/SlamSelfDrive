import numpy as np
import cv2 as cv

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

# matrices terminology
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Camera:
    def __init__(self, video_dim) -> None:
        self.calibration_frames = []
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
    
    def find_matching_points(self, frame):
        match = np.mean(frame, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame, kps)
        self.last_frame = {'kps': kps, 'des': des}
        if self.last_frame is None:
            return None
        matches = self.matcher.match(des, self.last_frame['des'])
        self.twin_points = []
        for m in matches:
            kp1 = kps[m.queryIdx].pt
            kp2 = self.last_frame['kps'][m.trainIdx].pt
            self.twin_points.append((kp1, kp2))
        return self.twin_points

    def view_interest_points(self, frame):
        if self.twin_points is None:
            print("no matches")
            return
        for pt1, pt2 in self.twin_points:
            # from current frame
            cv.circle(frame, (int(pt1[0]), int(pt1[1])), color=(0, 255, 0), radius=2)
            # from previous frame
            cv.circle(frame, (int(pt2[0]), int(pt2[1])), color=(0, 0, 255), radius=2)
        self.camera.estimate_pose(self.twin_points)
    
    def get_camera_pose(self):
        assert self.twin_points is not None
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
    
    def triangulation(self):
        E, pose = self.vision.get_camera_pose()
        self.past_matrices['E'] = E
        self.past_matrices['pose'] = pose
        if self.past_matrices['E'] is None or self.past_matrices['pose'] is None:
            return
        E_diff = self.past_matrices['E'] - E
        R_diff = self.past_matrices['pose']['R'] - pose['R']
        t_diff = self.past_matrices['pose']['t'] - pose['t']
        print("E_diff : ", E_diff)
        print("R_diff : ", R_diff)
        print("t_diff : ", t_diff)
        #cv.triangulatePoints()
    
    def view_points(self, frame):
        points = self.vision.find_matching_points(frame)
        print("extracted : ", len(points))
        self.vision.view_interest_points(frame)
        self.triangulation()
