from pupil_apriltags import Detector
import numpy as np
at_detector = Detector(families='tagStandard41h12',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)


def get_tag_3Dpoints(cell_size):
    world_points = np.array([[0, 0, 0],
                             [cell_size, 0, 0],
                             [cell_size, cell_size, 0],
                             [0, cell_size, 0]], dtype=np.float32)*5
    return world_points

def get_multi_tag_3Dpoints(cell_size, dim, h_dist, v_dist):
    num_row = dim[0]
    num_col = dim[1]

    hor_intv = cell_size * 11 + h_dist
    ver_intv = cell_size * 11 + v_dist

    total_points = []
    reference = get_tag_3Dpoints(cell_size) # [4,3]

    for r in range(num_row):
        for c in range(num_col):
            points = reference.copy()
            points[:,0] += c * hor_intv
            points[:,1] -= r * ver_intv
            total_points.append(points)
    total_points = np.concatenate(total_points, axis=0)
    return total_points



def tags2array(tags):
    repos = []
    for tag in tags:
        repos.append(tag.corners)
    repos = np.array(repos, dtype=np.float32)
    return repos.reshape(-1,2)


def detect_tags(img_gray):
    tags = at_detector.detect(img_gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
    num_tags = np.shape(tags)[0]
    if num_tags !=0:
        return tags, num_tags
    else:
        return None, 0