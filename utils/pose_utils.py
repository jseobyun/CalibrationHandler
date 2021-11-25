import cv2
import numpy as np

def make_mtx(fx, fy, cx, cy):
    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)
    return mtx

def build_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t.reshape(3)
    return T

def apply_RT(point, R, T):
    pc = point.copy() # N 3
    world_coord = np.dot(R, pc.transpose(1,0)).transpose(1,0) + T.reshape(1,3)
    return world_coord

def apply_RT_inv(point, R, T):
    pc = point.copy()
    cam_coord = pc - T.reshape(1,3)
    cam_coord= np.dot(R.transpose(1,0), cam_coord.transpose(1,0)).transpose(1,0)
    return cam_coord


def pixel2cam(point, f, c):
    pc = point.copy()
    x = (pc[:, 0] - c[0]) / f[0] * pc[:, 2]
    y = (pc[:, 1] - c[1]) / f[1] * pc[:, 2]
    z = pc[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

def get_pose(pt3d, pt2d, mtx, dist):
    ret, rvec, tvec = cv2.solvePnP(pt3d, pt2d, mtx, dist, cv2.SOLVEPNP_ITERATIVE)
    return ret, rvec, tvec
