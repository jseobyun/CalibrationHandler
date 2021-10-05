import os
import cv2
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.tag_utils import *
from utils.vis_utils import *
from utils.dir_utils import *

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/home/user/Desktop/yjs/codes/MultiBasler/oneshot_calib/view2')
    parser.add_argument('--cell_size', default=0.005)
    parser.add_argument('--grid_size', default=(4,3)) # (h, w)
    parser.add_argument('--h_dist', default=0.005)
    parser.add_argument('--v_dist', default=0.005)
    parser.add_argument('--subpix_window_size', default=(5,5))
    parser.add_argument('--vis', action='store_true', default=True)
    args = parser.parse_args()
    args_dict = vars(args)
    print("====Current arguments====")
    for key in args_dict.keys():
        print(key, " : ", args_dict[key])
    print("=========================")
    return args

def get_reproj_error(obj_point, img_point, mtx, dist, rvec, tvec):
    reproj_point, _ = cv2.projectPoints(obj_point, rvec, tvec, mtx, dist)
    reproj_point = reproj_point.reshape(-1 ,2)

    error = img_point - reproj_point
    rmse = np.mean(np.sqrt(np.sum(error**2, axis=1)))
    return error, rmse

def display_errors(obj_points, img_points, mtx, dist, rvecs, tvecs):
    assert len(obj_points) == len(img_points)
    assert len(img_points) == len(rvecs)
    assert len(img_points) == len(tvecs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    limit_circle = plt.Circle((0, 0), 1.0, color='r', linestyle='--', fill=False, alpha=0.5)
    ax.add_artist(limit_circle)
    print("RMSE values for individual images:")

    for i, (obj_point, img_point, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        error, rmse = get_reproj_error(obj_point, img_point, mtx, dist, rvec, tvec)

        ax.scatter(error[:, 0], error[:, 1], marker='+')
        print(i, '--', rmse)

    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_aspect('equal')

    fig.suptitle("reprojection errors")
    plt.show()

def undistort_images(image, mtx, dist):
    image_undistorted = cv2.undistort(image, mtx, dist)
    return image_undistorted

if __name__ == '__main__':
    args = parse_config()
    img_dir = os.path.join(args.root_dir, 'images')
    if not os.path.exists(img_dir):
        raise Exception(f"No [images] directory in the root {args.root_dir}")
    vis_dir = os.path.join(args.root_dir, 'vis')
    mkdir_safe(vis_dir)

    num_all_tags = args.grid_size[0] * args.grid_size[1]
    h_dist = args.h_dist
    v_dist = args.v_dist
    c_size = args.cell_size
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    img_points = []
    obj_points = []
    obj_point = get_multi_tag_3Dpoints(cell_size=c_size, dim=args.grid_size, h_dist=args.h_dist, v_dist=args.v_dist)

    file_names = os.listdir(img_dir)
    file_paths = [os.path.join(img_dir, file_name) for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
    file_paths_used = []
    for i, file_path in enumerate(file_paths):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error : {file_path} is not loaded well")
            continue
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags, num_tags = detect_tags(image_gray)
        if num_tags != num_all_tags:
            print(f"Check : {num_tags}/{num_all_tags} markers detected in {file_path}")
            continue
        tag_points = tags2array(tags)
        tag_points = cv2.cornerSubPix(image_gray, tag_points, args.subpix_window_size, (-1, -1), subpix_criteria)
        if args.vis:
            canvas = draw_tags(image, tags)
            cv2.imwrite(os.path.join(vis_dir, 'tag_' + file_names[i]), canvas)

        print(len(obj_points),'===', file_path)
        obj_points.append(obj_point)
        img_points.append(tag_points)
        file_paths_used.append(file_path)

    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    RMSE, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (2560, 2048), mtx, dist, rvecs, tvecs)

    display_errors(obj_points, img_points, mtx, dist, rvecs, tvecs)
    print("###########################")
    print("RMSE", RMSE)
    print("K", mtx)
    print("distortion coeffs", dist)

    intrinsic = {
        'fx' : mtx[0, 0],
        'fy' : mtx[1, 1],
        'cx' : mtx[0, 2],
        'cy' : mtx[1, 2],
        'height': 2048,
        'width' : 2560,
        'coeffs': dist.reshape(-1).tolist()
    }
    with open(os.path.join(args.root_dir, 'intrinsic.json'), 'w') as json_file:
        json.dump(intrinsic, json_file)
        print(f"Intrinsic param is saved at {os.path.join(args.root_dir, 'intrinsic.json')}")

    for file_path_used in file_paths_used:
        image = cv2.imread(file_path_used)
        image_ud = cv2.undistort(image, mtx, dist)
        cv2.imwrite(os.path.join(vis_dir, 'ud_' + file_path_used.split('/')[-1]), image_ud)



