import os
import cv2
import json
import numpy as np
import open3d as o3d

from pupil_apriltags import Detector
#
def apply_RT(point, R, T):
    pc = point.copy()
    world_coord = np.dot(R, pc.transpose(1,0)).transpose(1,0) + T.reshape(1,3)
    return world_coord

def apply_RT_inv(point, R, T):
    pc = point.copy()
    cam_coord = pc - T.reshape(1,3)
    cam_coord= np.dot(R.transpose(1,0), cam_coord.transpose(1,0)).transpose(1,0)
    return cam_coord

def pixel2cam(point, f, c):
    x = (point[:, 0] - c[0]) / f[0] * point[:, 2]
    y = (point[:, 1] - c[1]) / f[1] * point[:, 2]
    z = point[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord

def get_pose(canvas, tags_i, mtx, dist, save_path):
    img_points = []
    for tag in tags_i:
        for idx in range(len(tag.corners)):
            cv2.line(
                canvas,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 200, 0), 2
            )
            cv2.putText(
                canvas,
                str(idx),
                tuple(tag.corners[idx, :].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            img_points.append(tag.corners[idx, :])

        cv2.putText(
            canvas,
            str(tag.tag_id),
            (tag.center[0].astype(int) - 20,
             tag.center[1].astype(int) + 20,),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    img_points = np.array(img_points, dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(world_points, img_points, mtx, dist)

    reproj_points, _ = cv2.projectPoints(world_points, rvec, tvec, mtx, dist)
    reproj_points = reproj_points[:, 0, :]
    for idx in range(np.shape(reproj_points)[0]):
        cv2.line(
            canvas,
            tuple(reproj_points[idx - 1, :].astype(int)),
            tuple(reproj_points[idx, :].astype(int)),
            (0, 0, 255), 1
        )

    cv2.imwrite(os.path.join(save_path), canvas)
    return ret, rvec, tvec, img_points

if __name__=='__main__':
    root = '/home/user/Desktop/yjs/codes/MultiRealSense/calib'
    img_dir = os.path.join(root, 'images')
    depth_dir = os.path.join(root, 'depths')

    view0_dir = os.path.join(img_dir, 'view0')
    view1_dir = os.path.join(img_dir, 'view1')
    view2_dir = os.path.join(img_dir, 'view2')

    depth0_dir = os.path.join(depth_dir, 'view0')
    depth1_dir = os.path.join(depth_dir, 'view1')
    depth2_dir = os.path.join(depth_dir, 'view2')

    result_dir = os.path.join(root, 'results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    save0_dir = os.path.join(root, 'results', 'view0')
    save1_dir = os.path.join(root, 'results', 'view1')
    save2_dir = os.path.join(root, 'results', 'view2')
    if not os.path.exists(save0_dir):
        os.mkdir(save0_dir)
    if not os.path.exists(save1_dir):
        os.mkdir(save1_dir)
    if not os.path.exists(save2_dir):
        os.mkdir(save2_dir)

    intrinsic_path = os.path.join(root, 'intrinsic.json')
    extrinsic_path = os.path.join(root, 'extrinsic.json')

    with open(intrinsic_path, 'r') as json_file:
        json_data = json.load(json_file)
        view0_info = json_data['0']['stream.color']
        view1_info = json_data['1']['stream.color']
        view2_info = json_data['2']['stream.color']
        mtx0 = np.array([[view0_info['fx'], 0, view0_info['cx']],
                         [0, view0_info['fy'], view0_info['cy']],
                         [0, 0, 1]], dtype=np.float32)
        f0 = np.array([view0_info['fx'], view0_info['fy']])
        c0 = np.array([view0_info['cx'], view0_info['cy']])
        mtx1 = np.array([[view1_info['fx'], 0, view1_info['cx']],
                         [0, view1_info['fy'], view1_info['cy']],
                         [0, 0, 1]], dtype=np.float32)
        f1 = np.array([view1_info['fx'], view1_info['fy']])
        c1 = np.array([view1_info['cx'], view1_info['cy']])
        mtx2 = np.array([[view2_info['fx'], 0, view2_info['cx']],
                         [0, view2_info['fy'], view2_info['cy']],
                         [0, 0, 1]], dtype=np.float32)
        f2 = np.array([view2_info['fx'], view2_info['fy']])
        c2 = np.array([view2_info['cx'], view2_info['cy']])

        dist0 = np.array(view0_info['coeffs'], dtype=np.float32)
        dist1 = np.array(view1_info['coeffs'], dtype=np.float32)
        dist2 = np.array(view1_info['coeffs'], dtype=np.float32)

    inner_dist = 0.1715  # m
    outer_dist = 0.2200  # m

    world_points = np.array([[0, 0, 0],
                             [inner_dist, 0, 0],
                             [inner_dist, inner_dist, 0],
                             [0, inner_dist, 0],  # 0
                             [inner_dist + outer_dist, 0, 0],
                             [inner_dist * 2 + outer_dist, 0, 0],
                             [inner_dist * 2 + outer_dist, inner_dist, 0],
                             [inner_dist + outer_dist, inner_dist, 0],  # 1
                             [0, -outer_dist - inner_dist, 0],
                             [inner_dist, -outer_dist - inner_dist, 0],
                             [inner_dist, -outer_dist, 0],
                             [0, -outer_dist, 0],  # 2
                             [inner_dist + outer_dist, -outer_dist - inner_dist, 0],
                             [inner_dist * 2 + outer_dist, -outer_dist - inner_dist, 0],
                             [inner_dist * 2 + outer_dist, -outer_dist, 0],
                             [inner_dist + outer_dist, -outer_dist, 0]], dtype=np.float32)

    ### do R, T calculation
    file_names = os.listdir(view0_dir)
    for file_name in file_names:
        print(file_name)
        if file_name != '3_img.png':
            continue
        img0_path = os.path.join(view0_dir, file_name)
        img1_path = os.path.join(view1_dir, file_name)
        img2_path = os.path.join(view2_dir, file_name)
        depth_name = file_name.split('_')[0] + '_depth.png'
        depth0_path = os.path.join(depth0_dir, depth_name)
        depth1_path = os.path.join(depth1_dir, depth_name)
        depth2_path = os.path.join(depth2_dir, depth_name)

        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        depth0 = cv2.imread(depth0_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth2_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        img_h, img_w, img_c = np.shape(img0)

        at_detector = Detector(families='tagStandard41h12',
                               nthreads=1,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0)

        canvas0 = img0.copy()
        canvas1 = img1.copy()
        canvas2 = img2.copy()

        if (img_c == 3):
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        tags0 = at_detector.detect(img0, estimate_tag_pose=False, camera_params=None, tag_size=None)
        tags1 = at_detector.detect(img1, estimate_tag_pose=False, camera_params=None, tag_size=None)
        tags2 = at_detector.detect(img2, estimate_tag_pose=False, camera_params=None, tag_size=None)
        num_tags0 = np.shape(tags0)[0]
        num_tags1 = np.shape(tags1)[0]
        num_tags2 = np.shape(tags2)[0]

        if num_tags0 != 4 or num_tags1 != 4 or num_tags2 != 4:
            continue
        ret0, rvec0, T0, x0 = get_pose(canvas0, tags0, mtx0, dist0, os.path.join(save0_dir, file_name))
        ret1, rvec1, T1, x1 = get_pose(canvas1, tags1, mtx1, dist1, os.path.join(save1_dir, file_name))
        ret2, rvec2, T2, x2 = get_pose(canvas2, tags2, mtx2, dist2, os.path.join(save2_dir, file_name))

        if not ret0 or not ret1 or not ret2:
            continue
        R0, _ = cv2.Rodrigues(rvec0)
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)

        # view 0 to view 1
        R01 = np.matmul(R0, R1.transpose(1, 0))
        T01 = -np.dot(np.matmul(R0, R1.transpose(1, 0)), T1) + T0
        R01 = R01.transpose(1, 0)
        T01 = -np.dot(R01, T01)

        R02 = np.matmul(R0, R2.transpose(1, 0))
        T02 = -np.dot(np.matmul(R0, R2.transpose(1, 0)), T2) + T0
        R02 = R02.transpose(1, 0)
        T02 = -np.dot(R02, T02)

        print('R01', R01)
        print('T01', T01)
        print('R02', R02)
        print('T02', T02)
        ####
        def RT2EssentialMat(R, T):
            tx = np.array([[0, -T[2], T[1]],
                           [T[2], 0, -T[0]],
                           [-T[1], T[0], 0]], dtype=np.float32)

            E = np.dot(tx, R)
            return E


        def drawlines(img1_ori, img2_ori, lines, pts1, pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r, c, _ = img1_ori.shape
            img1 = img1_ori.copy()
            img2 = img2_ori.copy()
            for i, r in enumerate(lines):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                x0, y0 = map(int, [0, -r[2] / r[1]])
                x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
                img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

            # for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
            #     color = tuple(np.random.randint(0, 255, 3).tolist())
            #     x0, y0 = map(int, [0, -r[2] / r[1]])
            #     x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            #     img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            #
            #     img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
            #     img1 = cv2.putText(img1, str(i), (int(pt1[0]), int(pt1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
            #                        2)
            #
            #     img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, -1)
            #     img2 = cv2.putText(img2, str(i), (int(pt2[0]), int(pt2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
            #                        2)
            return img1, img2


        test0 = cv2.imread('/home/user/Desktop/three_person/images/view0/15_img.png')
        test1 = cv2.imread('/home/user/Desktop/three_person/images/view1/15_img.png')
        test2 = cv2.imread('/home/user/Desktop/three_person/images/view2/15_img.png')

        R01_inv = R01.transpose(1, 0)
        T01_inv = -T01

        R02_inv = R02.transpose(1, 0)
        T02_inv = -T02

        E01 = RT2EssentialMat(R01, T01)
        E02 = RT2EssentialMat(R02, T02)
        E01 = E01 / E01[2, 2]
        E02 = E02 / E02[2, 2]
        F01 = np.dot(np.dot(np.linalg.inv(mtx1.transpose(1,0)), E01), np.linalg.inv(mtx0))
        F02 = np.dot(np.dot(np.linalg.inv(mtx2.transpose(1, 0)), E02), np.linalg.inv(mtx0))

        lines01_left = cv2.computeCorrespondEpilines(x1.reshape(-1, 1, 2), 2, F01)
        lines01_right = cv2.computeCorrespondEpilines(x0.reshape(-1, 1, 2), 1, F01)
        lines01_left = lines01_left.reshape(-1,3)
        lines01_right = lines01_right.reshape(-1, 3)

        img_left_result0, _ = drawlines(canvas0, canvas1, lines01_left, x0, x1)
        img_right_result0, _ = drawlines(canvas1, canvas0, lines01_right, x1, x0)

        lines02_left = cv2.computeCorrespondEpilines(x2.reshape(-1, 1, 2), 2, F02)
        lines02_right = cv2.computeCorrespondEpilines(x0.reshape(-1, 1, 2), 1, F02)
        lines02_left = lines02_left.reshape(-1, 3)
        lines02_right = lines02_right.reshape(-1, 3)

        img_left_result1, _ = drawlines(canvas0, canvas2, lines02_left, x0, x2)
        img_right_result1, _ = drawlines(canvas2, canvas0, lines02_right, x2, x0)

        # cv2.imshow('01-0', img_left_result0)
        # cv2.imshow('01-1', img_right_result0)
        # cv2.imshow('02-0', img_left_result1)
        # cv2.imshow('02-2', img_right_result1)
        # #cv2.waitKey(0)
        ######
        valid = np.argwhere(depth0 != 0)
        r_valid = valid[:, 0]
        c_valid = valid[:, 1]
        rgb_valid = canvas0[r_valid, c_valid][:, ::-1]
        color0 = np.array(rgb_valid, dtype=np.float64) / 255.0
        d_valid = depth0[r_valid, c_valid]

        pcd0 = np.concatenate([c_valid[:, None], r_valid[:, None], d_valid[:, None]], axis=1)
        pcd0 = pixel2cam(pcd0, f0, c0)
        #pcd0 = apply_RT(pcd0, R, T * 1000)

        valid = np.argwhere(depth1 != 0)
        r_valid = valid[:, 0]
        c_valid = valid[:, 1]
        rgb_valid = canvas1[r_valid, c_valid][:, ::-1]
        color1 = np.array(rgb_valid, dtype=np.float64) / 255.0
        d_valid = depth1[r_valid, c_valid]

        pcd1 = np.concatenate([c_valid[:, None], r_valid[:, None], d_valid[:, None]], axis=1)
        pcd1 = pixel2cam(pcd1, f1, c1)
        pcd1 = apply_RT_inv(pcd1, R01, T01 * 1000)

        valid = np.argwhere(depth2 != 0)
        r_valid = valid[:, 0]
        c_valid = valid[:, 1]
        rgb_valid = canvas2[r_valid, c_valid][:, ::-1]
        color2 = np.array(rgb_valid, dtype=np.float64) / 255.0
        d_valid = depth2[r_valid, c_valid]

        pcd2 = np.concatenate([c_valid[:, None], r_valid[:, None], d_valid[:, None]], axis=1)
        pcd2 = pixel2cam(pcd2, f2, c2)
        pcd2 = apply_RT_inv(pcd2, R02, T02 * 1000)

        vis0 = o3d.geometry.PointCloud()
        vis0.points = o3d.utility.Vector3dVector(pcd0)
        green = np.zeros_like(color0)
        green[:,1] = 1.0
        vis0.colors = o3d.utility.Vector3dVector(color0)

        vis1 = o3d.geometry.PointCloud()
        vis1.points = o3d.utility.Vector3dVector(pcd1)
        red = np.zeros_like(color1)
        red[:, 0] = 1.0
        vis1.colors = o3d.utility.Vector3dVector(color1)

        vis2 = o3d.geometry.PointCloud()
        vis2.points = o3d.utility.Vector3dVector(pcd2)
        blue = np.zeros_like(color2)
        blue[:, 2] = 1.0
        vis2.colors = o3d.utility.Vector3dVector(color2)

        #o3d.visualization.draw_geometries([vis0, vis1, vis2])
        break


    extrinsic = {
        'R0' : np.identity(3).tolist(),
        'T0' : [0, 0, 0],
        'R1' : R01.tolist(),
        'T1' : T01.tolist(),
        'R2' : R02.tolist(),
        'T2' : T02.tolist(),
    }
    with open(extrinsic_path, 'w') as json_file:
        json.dump(extrinsic, json_file)
    print(f"Extrinsic file is saved at {extrinsic_path}")
