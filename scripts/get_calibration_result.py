import os
import pprint
from ruamel.yaml import YAML
import numpy as np
import cv2
import argparse

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

def save_yaml(file_path: str, data: dict):
    """
    Save data to a YAML file.
    """
    yaml = YAML()
    yaml.default_flow_style = None
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

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
    log_file_right = os.path.join(CALIBRATION_DIR, 'right.yml')
    log_file_left = os.path.join(CALIBRATION_DIR, 'left.yml')
    
    result_right = calibrate_camera(log_file_right)
    result2 = calibrate_camera(log_file_left)
    
    config_data = load_yaml(TASK_CONFIG_PATH)
    
    homogeneous_matrix_1 = np.vstack((np.hstack((result_right[2], result_right[3])), np.array([0, 0, 0, 1])))
    homogeneous_matrix_2 = np.vstack((np.hstack((result2[2], result2[3])), np.array([0, 0, 0, 1])))
    
    config_data['TF']['base2cam_demo']['base2cam_right'] = np.round(homogeneous_matrix_1, 8).tolist()
    config_data['TF']['base2cam_demo']['base2cam_left'] = np.round(homogeneous_matrix_2, 8).tolist()
    
    save_yaml(TASK_CONFIG_PATH, config_data)
    
    pprint.pprint("Calibration Result for Camera 1:")
    pprint.pprint(result_right)
    pprint.pprint("Calibration Result for Camera 2:")
    pprint.pprint(result2)

if __name__ == "__main__":
    main()
