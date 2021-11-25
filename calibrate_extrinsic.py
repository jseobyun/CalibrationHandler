import os
import cv2
import json
import numpy as np
import argparse
import warnings

from utils.io_utils import *
from utils.pose_utils import *
from utils.tag_utils import *
from utils.vis_utils import *

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_size', default=0.005, help='length of one cell in april tag. apriltag consists of 11 x 11 cells')
    parser.add_argument('--grid_size', default=(4, 3))  # (h, w)
    parser.add_argument('--h_dist', default=0.005)
    parser.add_argument('--v_dist', default=0.005)
    parser.add_argument('--subpix_window_size', default=(5, 5))
    parser.add_argument('--root', default='/home/jseob/Desktop/yjs/codes/Test_2nDC/multi_calib1', help='root directory that contains images and depths directories')

    args = parser.parse_args()
    return args


class ExtrinsicCalibrator():
    def __init__(self, root, pattern_spec):
        self.root = root
        self.cell_size = pattern_spec['cell_size']
        self.grid_size = pattern_spec['grid_size']
        self.h_dist = pattern_spec['h_dist']
        self.v_dist = pattern_spec['v_dist']
        self.start_marker_idx = pattern_spec['start_marker_idx']
        self.end_makrer_idx= pattern_spec['end_marker_idx']

        self.subpix_window_size = (5, 5)
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        dirs = os.listdir(self.root)
        dirs = [dir for dir in dirs if dir.startswith('cam')]
        self.num_views = len(dirs)

        if self.num_views == 0:
            raise FileNotFoundError(f"No view folders in {self.root}.")


        mtxs, dists, fs, cs = [], [], [], []
        for vidx in range(self.num_views):
            intrinsic_path = os.path.join(self.root, f'cam{vidx+1}', 'intrinsic.json')
            intrinsic_json = load_json(intrinsic_path)
            if intrinsic_json is None:
                raise FileNotFoundError(f"No intrinsic parameter in cam{vidx+1} folder.")
            fx, fy, cx, cy, dist = decode_intrinsic_json(intrinsic_json)
            fs.append(np.array([fx, fy]))
            cs.append(np.array([cx, cy]))
            dists.append(dist)
            mtxs.append(make_mtx(fx, fy, cx, cy))

        self.mtxs = mtxs
        self.dists = dists
        self.fs = fs
        self.cs = cs

        self.pattern3d = get_grid_tag_3Dpoints(self.cell_size, self.grid_size, self.h_dist, self.v_dist)
        file_names = os.listdir(os.path.join(self.root, 'cam1'))
        for vidx in range(1, self.num_views):
            num_files = len(os.listdir(os.path.join(self.root, f'cam{vidx+1}')))
            if num_files != len(file_names):
                warnings.warn("The number of files in each view folder does not match")
                break

        self.file_names = [file_name for file_name in file_names if file_name.lower().endswith('.png') or file_name.lower().endswith('.jpg')]

    def calibrate(self):
        row = [[] for _ in range(self.num_views)]
        T_table = [row for _ in range(self.num_views)]

        for file_name in self.file_names:
            print(f"Processing... {file_name}")
            Ts =[]
            for vidx in range(self.num_views):
                file_path = os.path.join(self.root, f'cam{vidx+1}', file_name)
                img = cv2.imread(file_path)
                #img = cv2.undistort(img, self.mtxs[vidx], self.dists[vidx])
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tags, num_tags = detect_tags(img_gray)
                tags, num_tags = filter_tags(tags, self.start_marker_idx, self.end_makrer_idx)
                print(f"cam{vidx+1}: detected marker {num_tags}/{self.grid_size[0]*self.grid_size[1]}")

                if num_tags != self.grid_size[0] * self.grid_size[1]:
                    Ts.append(None)
                else:
                    # canvas = draw_tags(img, tags)
                    # canvas = cv2.resize(canvas, (1200, 800))
                    # cv2.waitKey(1000)

                    pattern2d = tags2array(tags)
                    pattern2d = cv2.cornerSubPix(img_gray, pattern2d, self.subpix_window_size, (-1, -1), self.subpix_criteria)
                    RMSE, rvec, tvec = get_pose(self.pattern3d, pattern2d, self.mtxs[vidx], self.dists[vidx])
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec
                    T = build_T(R, t)
                    Ts.append(T)
            valid_idx = []
            for tidx, T in enumerate(Ts):
                if T is not None:
                    valid_idx.append(tidx)
            if len(valid_idx) <=1:
                print(f"No marker in {file_name} of all views.")
                continue
            for i in range(len(valid_idx)):
                for j in range(i+1, len(valid_idx)):
                    ii, jj = valid_idx[i], valid_idx[j]
                    T_im, T_jm = Ts[ii], Ts[jj]
                    T_ij = np.dot(T_im,  np.linalg.inv(T_jm))
                    T_table[ii][jj].append(T_ij)

        extrinsic = {}
        for i in range(self.num_views):
            for j in range(i+1, self.num_views):
                T_list = np.array(T_table[i][j])
                T = np.mean(T_list, axis=0)
                key = f'T{i}{j}'
                extrinsic[key] = T.tolist()
        self.extrinsics = extrinsic
        print("Calibration is done.")
        save_json(os.path.join(self.root, 'extrinsic_cams.json'), extrinsic)

    def visualize(self):
        visualize_cameras(self.extrinsics)

    def load(self):
        self.extrinsics = load_json(os.path.join(self.root, 'extrinsic_cams.json'))


if __name__=='__main__':

    root = '/home/jseob/Desktop/yjs/images/Gopro/calib_test'
    pattern_spec = {
        'cell_size' : 0.005,
        'grid_size' : (4, 3),
        'h_dist' : 0.005,
        'v_dist' : 0.005,
        'start_marker_idx': 20,
        'end_marker_idx' : 31,

    }
    calibrator = ExtrinsicCalibrator(root, pattern_spec)
    #calibrator.calibrate()
    calibrator.load()
    calibrator.visualize()


