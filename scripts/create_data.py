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

        self.current_episode = 0
        self.current_images_dir = None
        self.current_actions_dir = None
        self.current_commands_dir = None
        self.current_sample_idx = 1
        self.total_collected = 0

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, qos_profile_system_default)
        self.create_subscription(UInt8, '/command', self.command_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)

    def _start_new_episode(self) -> None:
        self.current_episode += 1
        episode_dir = self.dataset_dir / f'episode{self.current_episode:02d}'
        self.current_images_dir = episode_dir / 'images'
        self.current_actions_dir = episode_dir / 'actions'
        self.current_commands_dir = episode_dir / 'commands'

        self.current_images_dir.mkdir(parents=True, exist_ok=True)
        self.current_actions_dir.mkdir(parents=True, exist_ok=True)
        self.current_commands_dir.mkdir(parents=True, exist_ok=True)
        self.current_sample_idx = 1
        
    def flag_callback(self, _msg):
        if self.collect_flag:
            self.collect_flag = False
            self.get_logger().info('🔴Data collect stopped')
            return

        self._start_new_episode()
        self.collect_flag = True
        self.get_logger().info(
            f'⚪Create data started (episode{self.current_episode:02d}, idx={self.current_sample_idx:05d})'
        )

    def vel_callback(self, msg):
        self.latest_vel = msg

    def command_callback(self, msg):
        self.command = msg.data

    def timer_callback(self):
        if not self.collect_flag or not self.zed.grab():
            return

        if self.latest_vel is None or self.command is None:
            return

        image = self.zed.get_image()
        image_path = self.current_images_dir / f'{self.current_sample_idx:05d}.png'
        action_path = self.current_actions_dir / f'{self.current_sample_idx:05d}.csv'
        command_path = self.current_commands_dir / f'{self.current_sample_idx:05d}.csv'

        cv2.imwrite(str(image_path), image)

        with open(str(action_path), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([self.latest_vel.linear.x, self.latest_vel.angular.z])

        with open(str(command_path), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([self.command])

        self.total_collected += 1
        self.get_logger().info(
            f'🟢Collected episode{self.current_episode:02d}/#{self.current_sample_idx:05d} '
            f'(total={self.total_collected})'
        )
        self.current_sample_idx += 1

    def save_data(self) -> None:
        if self.total_collected == 0:
            self.get_logger().info('🔴No data to save')
            return

        self.get_logger().info(
            f'🔵Saved {self.total_collected} samples into {self.current_episode} episodes at {self.dataset_dir}'
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
