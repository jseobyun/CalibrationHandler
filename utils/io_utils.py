import os
import json
import numpy as np

def load_json(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_json(json_path, dict):
    with open(json_path, 'w') as json_file:
        data = json.dump(dict, json_file)
    print(f"JSON file is saved at {json_path}.")

def decode_intrinsic_json(intrinsic_json):
    fx = intrinsic_json['fx']
    fy = intrinsic_json['fy']
    cx = intrinsic_json['cx']
    cy = intrinsic_json['cy']
    dist = np.array(intrinsic_json['coeffs'])
    return fx, fy, cx, cy, dist
