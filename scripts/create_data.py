#!/usr/bin/env python3

import csv
from pathlib import Path
import time

import cv2
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Empty, UInt8

from ml_planner.zed_api_utils import ZED_API_Utils


SAMPLE_INTERVAL = 0.2


class DataCreator(Node):
    def __init__(self):
        super().__init__('create_data')
        self.zed = ZED_API_Utils()
        self.latest_vel = Twist()
        self.command = 1
        self.collect_flag = False
        self.current_episode = []
        self.session_dir = None
        self.episode_index = 0

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.create_subscription(Twist, '/cmd_vel', self.vel_callback, qos_profile_system_default)
        self.create_subscription(UInt8, '/command', self.command_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)

    def flag_callback(self, _msg):
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            if self.session_dir is None:
                package_root = Path(__file__).parent.parent
                data_base_dir = package_root / 'data'
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                self.session_dir = data_base_dir / f'{timestamp}_dataset'
                self.session_dir.mkdir(parents=True, exist_ok=True)
            self.current_episode = []
            self.get_logger().info('Create data started')
        else:
            self.save_current_episode()
            self.get_logger().info('Data collect stopped')

    def vel_callback(self, msg):
        self.latest_vel = msg

    def command_callback(self, msg):
        self.command = int(msg.data)

    def timer_callback(self):
        if not self.collect_flag or not self.zed.grab():
            return

        image = self.zed.get_image()
        self.current_episode.append((image, self.latest_vel.linear.x, self.latest_vel.angular.z, self.command))
        self.get_logger().info(f'Collected frame #{len(self.current_episode)}')

    def save_current_episode(self) -> None:
        if not self.current_episode or self.session_dir is None:
            return

        self.episode_index += 1
        episode_dir = self.session_dir / f'episode_{self.episode_index:05d}'
        images_dir = episode_dir / 'images'
        actions_dir = episode_dir / 'actions'
        commands_dir = episode_dir / 'commands'

        images_dir.mkdir(parents=True, exist_ok=True)
        actions_dir.mkdir(parents=True, exist_ok=True)
        commands_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, linear_x, angular_z, command) in enumerate(self.current_episode, start=1):
            image_path = images_dir / f'{idx:05d}.png'
            action_path = actions_dir / f'{idx:05d}.csv'
            command_path = commands_dir / f'{idx:05d}.csv'

            cv2.imwrite(str(image_path), image)

            with action_path.open('w', newline='') as csvfile:
                csv.writer(csvfile).writerow([linear_x, angular_z])

            with command_path.open('w', newline='') as csvfile:
                csv.writer(csvfile).writerow([command])

        self.get_logger().info(f'Saved episode with {len(self.current_episode)} frames to {episode_dir}')
        self.current_episode = []


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCreator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.save_current_episode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
