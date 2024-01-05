"""@file camera_calibration.py
@brief Camera calibration for robotics application.
@details This script provides an interface for calibrating a camera in a robotics application.
"""

import sys
import os
import rospy
import cv2
import yaml
import numpy as np
from camera import IntelCamera, KinectCamera
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose
import argparse
import pprint

class CameraCalibration:
    """@class CameraCalibration
    @brief Handles the calibration processes for a camera in a robotics setup.
    """
    def __init__(self, config_file, camera):
        """@brief Constructor
        @param config_file The path to the configuration file.
        @param camera The camera type (e.g., 'left' or 'right').
        """
        self.load_config(config_file, camera)
        self.tcp_pose = Pose()
        self.updated_tcp_pose = False
        self.stack_count = 1
        self.pose_stack = {"cam2marker": {}, "base2end": {}}
        self.pose_stack_yaml = {"cam2marker": {}, "base2end": {}}

    def load_config(self, config_file, camera):
        """@brief Load the configuration for the specified camera.
        """
        with open(config_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        if camera == 'right':
            self.cfg['cam'] = 'right'
        elif camera == 'left':
            self.cfg['cam'] = 'left'
        else:
            raise ValueError("Invalid camera name")

        self.cam = IntelCamera(self.cfg)

    def init_ros(self):
        """@brief Initialize ROS node and set up subscribers and publishers.
        """
        rospy.init_node('manipulator', anonymous=True)
        rospy.Subscriber('tcp_pose', Pose, self.tcp_pose_callback)
        self.tcp_pose_publisher = rospy.Publisher('/command', UInt8, queue_size=10)

    def tcp_pose_callback(self, msg):
        """@brief Callback for receiving the TCP pose from ROS.
        """
        self.tcp_pose = msg
        self.updated_tcp_pose = True
        print("GOT POSE")

    def stream(self):
        """@brief Stream from the camera and handle user input.
        """
        while not rospy.is_shutdown():
            rgb_img, _ = self.cam.stream()
            try:
                self.cam.detectCharuco()
            except:
                print("No charuco board detected")
            
            cv2.imshow("rgb", rgb_img)
            key = cv2.waitKey(1)

            if key == ord('s'):
                self.capture_pose()
            elif key == ord('e'):
                break

    def capture_pose(self):
        """@brief Capture pose data.
        """
        self.tcp_pose_publisher.publish(1)

        while not self.updated_tcp_pose:
            pass

        T = np.eye(4)
        pose_data = self.tcp_pose_to_transform(self.tcp_pose)
        T[:3, :3] = pose_data['rot']
        T[:3, -1] = pose_data['trans'].T

        self.pose_stack['cam2marker'][str(self.stack_count)] = self.cam.cam2marker
        self.pose_stack['base2end'][str(self.stack_count)] = T
        self.pose_stack_yaml['cam2marker'][str(self.stack_count)] = self.cam.cam2marker.tolist()
        self.pose_stack_yaml['base2end'][str(self.stack_count)] = T.tolist()

        print(f"POSE STORED...{self.stack_count}")
        self.stack_count += 1
        self.updated_tcp_pose = False

    def tcp_pose_to_transform(self, pose):
        """@brief Convert a ROS pose message to a transformation matrix.
        @returns Dictionary with rotation and translation data.
        """
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        scipy_quat = R.from_quat([qx, qy, qz, qw])
        rotation_mat = scipy_quat.as_matrix()

        return {"rot": rotation_mat, "trans": np.array([x, y, z])}

    def save(self, path):
        """
        @brief Save the pose stack to a YAML file.
        @param path The path to save the file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.pose_stack_yaml, f, default_flow_style=None)

def main():
    """@brief Main function. Parses arguments, runs calibration, and saves data.
    """
    parser = argparse.ArgumentParser(description='Calibrate cameras and save calibration data as YAML.')
    parser.add_argument('--camera', type=str, required=True, default='right', help='List of cameras to calibrate (default: right left)')
    args = parser.parse_args()
    config_file = "./src/suction_net_ros/config/task_config.yml"
    calibrator = CameraCalibration(config_file, args.camera)
    calibrator.init_ros()
    calibrator.stream()
    save_path = f"/home/vision/catkin_ws/src/hand-eye-calibration/calibration/{args.camera}.yml"
    calibrator.save(save_path)
    print("DONE!")

if __name__ == "__main__":
    main()
