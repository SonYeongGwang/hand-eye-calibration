import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import yaml
import numpy as np
import pprint

with open('/home/hand-eye/catkin_ws/src/hand-eye_calibration/calibration/log.yml') as f:   
    pose_stack = yaml.load(f, Loader=yaml.FullLoader)

    # stored_cam2marker = np.reshape(stored_cam2marker, (4, 4))

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

pprint.pprint(cv2.calibrateRobotWorldHandEye(rot_a, trans_a, rot_b, trans_b))