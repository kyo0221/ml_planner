#!/usr/bin/env python3

import csv
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, UInt8

import cv2

from ml_planner.zed_api_utils import ZED_API_Utils


SAMPLE_INTERVAL = 0.2


class DataCreator(Node):
    def __init__(self):
        super().__init__('create_data')
        self.zed = ZED_API_Utils()
        self.collected_data = []
        self.collect_flag = False

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, qos_profile_system_default)
        self.create_subscription(UInt8, '/command', self.command_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        
    def flag_callback(self, _msg):
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            self.get_logger().info('⚪Create data started')
        else:
            self.get_logger().info('🔴Data collect stopped')

    def vel_callback(self, msg):
        self.latest_vel = msg

    def command_callback(self, msg):
        self.command = msg.data

    def timer_callback(self):
        if not self.zed.grab() or not self.collect_flag:
            return

        image = self.zed.get_image()
        self.collected_data.append((image, self.latest_vel, self.command))
        self.get_logger().info(f'🟢Collected data #{len(self.collected_data)}')

    def save_data(self) -> None:
        if len(self.collected_data) == 0:
            self.get_logger().info('🔴No data to save')
            return

        package_root = Path(__file__).parent.parent
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = data_base_dir / f'{timestamp}_dataset'
        
        images_dir = dataset_dir / 'images'
        action_dir = dataset_dir / 'actions'
        command_dir = dataset_dir / 'commands'

        images_dir.mkdir(parents=True, exist_ok=True)
        action_dir.mkdir(parents=True, exist_ok=True)
        command_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, action, command) in enumerate(self.collected_data, start=1):
            image_path = images_dir / f'{idx:05d}.png'
            action_path = action_dir / f'{idx:05d}.csv'
            command_path = command_dir / f'{idx:05d}.csv'

            cv2.imwrite(str(image_path), image)

            with open(str(action_path), 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([action.linear.x, action.angular.z])

            with open(str(command_path), 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([command])

        self.get_logger().info(f'🔵Saved {len(self.collected_data)} samples to {dataset_dir}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCreator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
