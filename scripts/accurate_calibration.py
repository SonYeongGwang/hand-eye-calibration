import sys
import os
import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose

sys.path.insert(0, '/home/vision/catkin_ws/src/hand-eye-calibration/scripts')
from camera import IntelCamera

import cv2
import yaml
import numpy as np
from camera import IntelCamera, KinectCamera
from scipy.spatial.transform import Rotation as R
import pprint

def tcp_pose_callback(msg):
    global tcp_pose
    global updated_tcp_pose
    
    tcp_pose = msg
    updated_tcp_pose = True
    print("GOT POSE")

ref_path = os.getcwd()

cam = IntelCamera(cfg=[])
rospy.init_node('manipulator', anonymous=True)
rospy.Subscriber('tcp_pose', Pose, tcp_pose_callback)
tcp_pose_publisher = rospy.Publisher('/command', UInt8, queue_size=10)
tcp_pose = Pose()
updated_tcp_pose = False

stack_count = 1
pose_stack = {}
pose_stack['cam2marker'] = {}
pose_stack['base2end'] = {}

pose_stack_yaml = {}
pose_stack_yaml['cam2marker'] = {}
pose_stack_yaml['base2end'] = {}

while not rospy.is_shutdown():
    rgb_img, _ = cam.stream()
    try:
        cam.detectCharuco()
    except:
        print("No charuco board detected")
    cv2.imshow("rgb", rgb_img)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        
        ## send request for the tcp pose
        tcp_pose_publisher.publish(1)
        
        ## wait until the pose is updated
        while updated_tcp_pose == False:
            # print("waiting for the pose update...")
            pass
        
        ## read pose data
        x = tcp_pose.position.x
        y = tcp_pose.position.y
        z = tcp_pose.position.z
        qx = tcp_pose.orientation.x
        qy = tcp_pose.orientation.y
        qz = tcp_pose.orientation.z
        qw = tcp_pose.orientation.w
        
        ## convert into rotation matrix
        T = np.eye(4)
        scipy_quat = R.from_quat(np.array([qx, qy, qz, qw]))
        rotation_mat = scipy_quat.as_matrix()
        
        T[:3, :3] = rotation_mat
        T[:3, -1] = np.array([x, y, z]).T
        
        ## store in the stack
        pose_stack['cam2marker'][str(stack_count)] = cam.cam2marker
        pose_stack['base2end'][str(stack_count)] = T
        pose_stack_yaml['cam2marker'][str(stack_count)] = cam.cam2marker.tolist()
        pose_stack_yaml['base2end'][str(stack_count)] = T.tolist()
        
        print("POSE STORED...{}".format(stack_count))

        stack_count += 1
        updated_tcp_pose = False
        
    elif key == ord('e'):
        break

with open("/home/vision/catkin_ws/src/hand-eye-calibration/calibration/log.yml", 'w') as f:
      
    yaml.dump(pose_stack_yaml, f, default_flow_style=None)

print("DONE!")

num_of_poses = len(pose_stack['cam2marker'].keys())

rot_a = np.empty((num_of_poses, 3, 3))
rot_b = np.empty((num_of_poses, 3, 3))
trans_a = np.empty((num_of_poses, 3, 1))
trans_b = np.empty((num_of_poses, 3, 1))

for i in range(num_of_poses):
    rot_a[i,:,:] = pose_stack['cam2marker'][str(i+1)][:3, :3]
    rot_b[i,:,:] = pose_stack['base2end'][str(i+1)][:3, :3]
    trans_a[i,:,-1] = pose_stack['cam2marker'][str(i+1)][:3, -1]
    trans_b[i,:,-1] = pose_stack['base2end'][str(i+1)][:3, -1]