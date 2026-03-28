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
        self.collect_flag = False
        self.latest_vel = None
        self.command = None

        package_root = Path(__file__).parent.parent
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.dataset_dir = data_base_dir / f'{timestamp}_dataset'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.current_episode_index = 0
        self.current_sample_index = 1
        self.total_collected_samples = 0
        self.images_dir = None
        self.action_dir = None
        self.command_dir = None

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, qos_profile_system_default)
        self.create_subscription(UInt8, '/command', self.command_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        
    def flag_callback(self, _msg):
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            self._start_new_episode()
        else:
            self.get_logger().info('🔴Data collect stopped')

    def _start_new_episode(self):
        self.current_episode_index += 1
        self.current_sample_index = 1

        episode_dir = self.dataset_dir / f'episode{self.current_episode_index:02d}'
        self.images_dir = episode_dir / 'images'
        self.action_dir = episode_dir / 'actions'
        self.command_dir = episode_dir / 'commands'

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.action_dir.mkdir(parents=True, exist_ok=True)
        self.command_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'⚪Create data started: {episode_dir.name}')

    def vel_callback(self, msg):
        self.latest_vel = msg

    def command_callback(self, msg):
        self.command = msg.data

    def timer_callback(self):
        if not self.zed.grab() or not self.collect_flag:
            return
        if self.latest_vel is None or self.command is None:
            return

        image = self.zed.get_image()
        sample_id = f'{self.current_sample_index:05d}'
        image_path = self.images_dir / f'{sample_id}.png'
        action_path = self.action_dir / f'{sample_id}.csv'
        command_path = self.command_dir / f'{sample_id}.csv'

        cv2.imwrite(str(image_path), image)

        with open(str(action_path), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([self.latest_vel.linear.x, self.latest_vel.angular.z])

        with open(str(command_path), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([self.command])

        self.get_logger().info(f'🟢Collected episode{self.current_episode_index:02d} #{self.current_sample_index}')
        self.current_sample_index += 1
        self.total_collected_samples += 1

    def save_data(self) -> None:
        if self.current_episode_index == 0:
            self.get_logger().info('🔴No data to save')
            return
        self.get_logger().info(
            f'🔵Saved {self.total_collected_samples} samples in {self.current_episode_index} episodes to {self.dataset_dir}'
        )


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
