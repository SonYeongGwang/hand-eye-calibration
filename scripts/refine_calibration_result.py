import yaml
import cv2
import os
import argparse
import pprint

import numpy as np
import open3d as o3d

from camera import IntelCamera

from ruamel.yaml import YAML

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calibrate cameras and save calibration data as YAML.')
parser.add_argument('--camera', type=str, required=True, default='right', help='List of cameras to calibrate (default: right left)')
args = parser.parse_args()

yaml = YAML()
yaml.default_flow_style = None
log_file1 = os.path.expanduser('~/catkin_ws/src/hand-eye-calibration/calibration/right.yml')
log_file2 = os.path.expanduser('~/catkin_ws/src/hand-eye-calibration/calibration/left.yml')

# Define constants for file paths
CALIBRATION_DIR = os.path.expanduser('~/catkin_ws/src/hand-eye-calibration/calibration')
TASK_CONFIG_PATH = os.path.expanduser('~/catkin_ws/src/suction_net_ros/config/task_config.yml')

def _transform(x=0, y=0, z=0, rx=0, ry=0, rz=0):
    if x != 0:
        T = np.array([[    1,                         0,                         0,                          x],
                      [    0,                         np.math.cos(rx),           -np.math.sin(rx),           y],
                      [    0,                         np.math.sin(rx),            np.math.cos(rx),           z],
                      [    0,                         0,                         0,                          1]])
    elif y != 0:                 
        T = np.array([[    np.math.cos(ry),           0,                         np.math.sin(ry),            x],
                      [    0,                         1,                         0,                          y],
                      [    -np.math.sin(ry),          0,                         np.math.cos(ry),            z],
                      [    0,                         0,                         0,                          1]])
    elif z != 0:                 
        T = np.array([[    np.math.cos(rz),           -np.math.sin(rz),          0,                          x],
                      [    np.math.sin(rz),           np.math.cos(rz),           0,                          y],
                      [    0,                         0,                         1,                          z],
                      [    0,                         0,                         0,                          1]])

    elif rx != 0:
        T = np.array([[    1,                         0,                         0,                          x],
                      [    0,                         np.math.cos(rx),           -np.math.sin(rx),           y],
                      [    0,                         np.math.sin(rx),            np.math.cos(rx),           z],
                      [    0,                         0,                         0,                          1]])
    elif ry != 0:                 
        T = np.array([[    np.math.cos(ry),           0,                         np.math.sin(ry),            x],
                      [    0,                         1,                         0,                          y],
                      [    -np.math.sin(ry),          0,                         np.math.cos(ry),            z],
                      [    0,                         0,                         0,                          1]])
    elif rz != 0:                 
        T = np.array([[    np.math.cos(rz),           -np.math.sin(rz),          0,                          x],
                      [    np.math.sin(rz),           np.math.cos(rz),           0,                          y],
                      [    0,                         0,                         1,                          z],
                      [    0,                         0,                         0,                          1]])

    return T

def _rotate(rx=0, ry=0, rz=0):

    if rx != 0:
        R = np.array([[    1,                         0,                         0,                ],
                      [    0,                         np.math.cos(rx),           -np.math.sin(rx)  ],
                      [    0,                         np.math.sin(rx),            np.math.cos(rx)  ]])
    elif ry != 0:                 
        R = np.array([[    np.math.cos(ry),           0,                         np.math.sin(ry)   ],
                      [    0,                         1,                         0,                ],
                      [    -np.math.sin(ry),          0,                         np.math.cos(ry)   ]])
    elif rz != 0:                 
        R = np.array([[    np.math.cos(rz),           -np.math.sin(rz),          0,                ],
                      [    np.math.sin(rz),           np.math.cos(rz),           0,                ],
                      [    0,                         0,                         1,                ]])

    return R

def load_yaml(file_path: str) -> dict:
    """
    Load YAML data from a file.
    """
    yaml = YAML()
    yaml.default_flow_style = None
    with open(file_path) as f:
        data = yaml.load(f)
    return data

def load_poses(file_path: str):
    """
    Load poses from a YAML file.
    """
    pose_stack = load_yaml(file_path)
    num_of_poses = len(pose_stack['cam2marker'].keys())
    rot_a = np.empty((num_of_poses, 3, 3))
    rot_b = np.empty((num_of_poses, 3, 3))
    trans_a = np.empty((num_of_poses, 3, 1))
    trans_b = np.empty((num_of_poses, 3, 1))

    for i in range(num_of_poses):
        rot_b[i,:,:] = np.reshape(pose_stack['cam2marker'][str(i+1)], (4, 4))[:3, :3]
        rot_a[i,:,:] = np.reshape(pose_stack['base2end'][str(i+1)], (4, 4))[:3, :3]
        trans_b[i,:,-1] = np.reshape(pose_stack['cam2marker'][str(i+1)], (4, 4))[:3, -1]
        trans_a[i,:,-1] = np.reshape(pose_stack['base2end'][str(i+1)], (4, 4))[:3, -1]

    return rot_a, trans_a, rot_b, trans_b

def calibrate_camera(log_file: str):
    """
    Perform hand-eye calibration for a camera.
    """
    rot_a, trans_a, rot_b, trans_b = load_poses(log_file)
    return cv2.calibrateRobotWorldHandEye(rot_a, trans_a, rot_b, trans_b)

def main():
    """
    Main function to perform calibration and update configuration.
    """

    R = np.eye(3, 3)
    past_cam2base_R = np.eye(3, 3)
    updated_cam2base_R = np.eye(3, 3)

    log_file_right = os.path.join(CALIBRATION_DIR, 'right.yml')
    log_file_left = os.path.join(CALIBRATION_DIR, 'left.yml')

    cfg = load_yaml(TASK_CONFIG_PATH)
    if args.camera == 'right':
        rot_a, trans_a, rot_b, trans_b = load_poses(log_file_right)
        cfg['cam'] = 'right'
        calib_result = calibrate_camera(log_file_right)
    else:
        rot_a, trans_a, rot_b, trans_b = load_poses(log_file_left)
        cfg['cam'] = 'left'
        calib_result = calibrate_camera(log_file_left)

    base2cam = np.vstack((np.hstack((calib_result[2], calib_result[3])), np.array([0, 0, 0, 1])))
    
    base2cam[:3, -1] = np.array([0.16764602, 0.72984984, 0.92537267]).T
    
    cam2base = np.linalg.inv(base2cam)

    cam2base_t = cam2base[:3, -1]

    cam = IntelCamera(cfg)

    for i in range(10):
        rgb, depth = cam.stream()
    
    xyz = cam.generate(depth)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    base_frame.transform(cam2base)

    past_cam2base_R = np.eye(3, 3)

    stored_marker_poses = []
    for i in range(len(rot_b)):
        marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        marker_frame.rotate(rot_b[i], center=[0, 0, 0])
        marker_frame.translate(trans_b[i])
        stored_marker_poses.append(marker_frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window('Calibration Result Display', width=848, height=480)
    added = False
    transform_resolution = 0.01

    # o3d.visualization.draw_geometries([camera_frame, base_frame, pcd])

    while 1:

        win = cv2.namedWindow("key")
        cv2.imshow("rgb", rgb)
        key = cv2.waitKey(1)

        if key == ord('q'):
            R =_rotate(rx=transform_resolution, ry=0, rz=0)

        elif key == ord('w'):
            R =_rotate(rx=0, ry=transform_resolution, rz=0)

        elif key == ord('e'):
            R =_rotate(rx=0, ry=0, rz=transform_resolution)

        elif key == ord('a'):
            R = _rotate(rx=-transform_resolution, ry=0, rz=0)

        elif key == ord('s'):
            R = _rotate(rx=0, ry=-transform_resolution, rz=0)

        elif key == ord('d'):
            R = _rotate(rx=0, ry=0, rz=-transform_resolution)

        elif key == ord('y'):
            
            updated_R = np.dot(cam2base[:3, :3], updated_cam2base_R)
            cam2base[:3, :3] = updated_R

            print("-"*50)
            print('REFINED!\n', repr(np.linalg.inv(cam2base)))

        elif key == ord('r'):
            print("select trasnsform resolution level (currnet:{})".format(transform_resolution))
            transform_resolution = float(input())


        updated_cam2base_R = np.dot(past_cam2base_R, R)

        base_frame.rotate(np.linalg.inv(past_cam2base_R), center=base_frame.get_center())
        base_frame.rotate(updated_cam2base_R, center=base_frame.get_center())

        if added == False:
            vis.add_geometry(camera_frame)
            vis.add_geometry(base_frame)
            vis.add_geometry(pcd)

            for i in range(len(stored_marker_poses)):
                vis.add_geometry(stored_marker_poses[i])

            added = True
        vis.update_geometry(camera_frame)
        vis.update_geometry(base_frame)
        vis.update_geometry(pcd)

        for i in range(len(stored_marker_poses)):
            vis.update_geometry(stored_marker_poses[i])

        vis.poll_events()
        vis.update_renderer()
    
    # pprint.pprint("Calibration Result for Camera 1:")
    # pprint.pprint(result_right)
    # pprint.pprint("Calibration Result for Camera 2:")
    # pprint.pprint(result_left)

        past_cam2base_R = updated_cam2base_R
        R = np.eye(3, 3)

if __name__ == "__main__":
    main()