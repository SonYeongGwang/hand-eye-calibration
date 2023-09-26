import sys
import cv2
import yaml
import numpy as np
import pprint
import os
log_file1 = os.path.expanduser('~/catkin_ws/src/hand-eye-calibration/calibration/first.yml')
log_file2 = os.path.expanduser('~/catkin_ws/src/hand-eye-calibration/calibration/second.yml')

def load_poses(file_path):
    with open(file_path) as f:
        pose_stack = yaml.load(f, Loader=yaml.FullLoader)

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

# Load poses from the two log files
rot_a1, trans_a1, rot_b1, trans_b1 = load_poses(log_file1)
rot_a2, trans_a2, rot_b2, trans_b2 = load_poses(log_file2)

# Perform hand-eye calibration for the two cameras
result1 = cv2.calibrateRobotWorldHandEye(rot_a1, trans_a1, rot_b1, trans_b1)
result2 = cv2.calibrateRobotWorldHandEye(rot_a2, trans_a2, rot_b2, trans_b2)

task_path = os.path.expanduser('~/catkin_ws/src/suction_net_ros/config/task_config.yml')
with open(task_path, 'r') as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)

# Update the matrices in the config data
config_data['TF']['base2cam_demo']['base2cam_1'] = np.hstack((result1[0], result1[1].reshape(-1, 1))).flatten().tolist()
config_data['TF']['base2cam_demo']['base2cam_2'] = np.hstack((result2[0], result2[1].reshape(-1, 1))).flatten().tolist()

# Write the updated config data back to the file
with open(task_path, 'w') as f:
    yaml.dump(config_data, f)

pprint.pprint("Calibration Result for Camera 1:")
pprint.pprint(result1)
pprint.pprint("Calibration Result for Camera 2:")
pprint.pprint(result2)


