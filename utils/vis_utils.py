import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.pose_utils import apply_RT

def draw_tags(img, tags):
    canvas = img.copy()
    for tag in tags:
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

    return canvas

def draw_points(img, points):
    canvas = img.copy()
    for point in points:
        cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas

def visualize_cameras(extrinsics):
    max_cam_num = 10
    cameras = []
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, max_cam_num)]
    colors = [(c[2], c[1], c[0]) for c in colors]

    k = 0.003
    camera_line = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    camera_points = np.array([[0, 0, 0],
                              [-17 * k, -10 * k, 40 * k],
                              [17 * k, -10 * k, 40 * k],
                              [17 * k, 10 * k, 40 * k],
                              [-17 * k, 10 * k, 40 * k],
                              ])
    camera = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(camera_points),
        lines=o3d.utility.Vector2iVector(camera_line),
    )
    camera_colors = [colors[0] for _ in range(8)]
    camera.colors = o3d.utility.Vector3dVector(camera_colors)
    cameras.append(camera)

    for kidx, key in enumerate(extrinsics.keys()):
        i,j = key[1], key[2]
        if int(i) != 0: # base
            continue
        T_0j = np.array(extrinsics[key]).reshape(4, 4)
        R_0j = T_0j[:3, :3]
        t_0j = T_0j[:3, -1]

        camera_points_kidx= apply_RT(camera_points, R_0j, t_0j)
        camera_kidx = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(camera_points_kidx),
            lines=o3d.utility.Vector2iVector(camera_line),
        )

        camera_colors = [colors[kidx+1] for _ in range(8)]
        camera_kidx.colors = o3d.utility.Vector3dVector(camera_colors)
        cameras.append(camera_kidx)

    o3d.visualization.draw_geometries(cameras)





