#!/usr/bin/env python3

from collections import deque
import csv
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

import cv2

import pyzed.sl as sl
from ml_planner.zed_api_utils import ZED_API_Utils


SAMPLE_INTERVAL = 0.5
FUTURE_STEPS = 10

class Trajectory:
    def __init__(self):
        self.points = []

    @staticmethod
    def _is_state_ok(state):
        return state == sl.POSITIONAL_TRACKING_STATE.OK

    @classmethod
    def _yaw_from_quaternion(cls, orientation):
        x = orientation['x']
        y = orientation['y']
        z = orientation['z']
        w = orientation['w']
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @classmethod
    def from_odoms(cls, odom_list):
        if len(odom_list) != FUTURE_STEPS:
            return None

        if any(not cls._is_state_ok(odom['state']) for odom in odom_list):
            return None

        trajectory = cls()
        x_sum = 0.0
        y_sum = 0.0
        yaw_sum = 0.0
        for odom in odom_list:
            dx_local = odom['position']['x']
            dy_local = odom['position']['y']
            cos_yaw = math.cos(yaw_sum)
            sin_yaw = math.sin(yaw_sum)
            x_sum += cos_yaw * dx_local - sin_yaw * dy_local
            y_sum += sin_yaw * dx_local + cos_yaw * dy_local
            yaw_sum += cls._yaw_from_quaternion(odom['orientation'])
            trajectory.points.append({
                'x': x_sum,
                'y': y_sum,
            })
        return trajectory


class DataCreator(Node):
    def __init__(self):
        super().__init__('create_data')
        self.zed = ZED_API_Utils()
        self.collected_data = []
        self.window = deque(maxlen=FUTURE_STEPS + 1)

        self.collect_flag = False

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        

    def flag_callback(self, _msg):
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            self.window.clear()
            self.get_logger().info('⚪Create data started')
        else:
            self.window.clear()
            self.get_logger().info('🔴Data collect stopped')

    def timer_callback(self):
        if not self.zed.grab():
            return

        image = self.zed.get_image()
        odom = self.zed.get_odom()
        self.publisher_odom(odom)

        if not self.collect_flag:
            return

        self.window.append({'image': image, 'odom': odom})

        if len(self.window) < FUTURE_STEPS + 1:
            return

        samples = list(self.window)
        image_at_t = samples[0]['image']
        future_odoms = [sample['odom'] for sample in samples[1:]]
        trajectory = Trajectory.from_odoms(future_odoms)
        if trajectory is None:
            self.get_logger().info('🟡Skipped sample: trajectory contains non-OK odom state')
            return

        self.collected_data.append((image_at_t, trajectory))
        self.get_logger().info(f'🟢Collected data #{len(self.collected_data)}')

    def publisher_odom(self, odom):
        msg = Odometry()
        timestamp_ns = int(odom['timestamp_ns'])
        msg.header.stamp.sec = timestamp_ns // 1000000000
        msg.header.stamp.nanosec = timestamp_ns % 1000000000
        msg.header.frame_id = 'base_link'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = odom['position']['x']
        msg.pose.pose.position.y = odom['position']['y']
        msg.pose.pose.position.z = odom['position']['z']
        msg.pose.pose.orientation.x = odom['orientation']['x']
        msg.pose.pose.orientation.y = odom['orientation']['y']
        msg.pose.pose.orientation.z = odom['orientation']['z']
        msg.pose.pose.orientation.w = odom['orientation']['w']

        self.odom_pub.publish(msg)

    def save_data(self) -> None:
        if len(self.collected_data) == 0:
            self.get_logger().info('🔴No data to save')
            return

        package_root = Path(__file__).parent.parent
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = data_base_dir / f'{timestamp}_dataset'
        
        images_dir = dataset_dir / 'images'
        trajectory_dir = dataset_dir / 'paths'

        images_dir.mkdir(parents=True, exist_ok=True)
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, trajectory) in enumerate(self.collected_data, start=1):
            image_path = images_dir / f'{idx:05d}.png'
            csv_path = trajectory_dir / f'{idx:05d}.csv'

            cv2.imwrite(str(image_path), image)

            with open(str(csv_path), 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['x', 'y'])
                for point in trajectory.points:
                    csv_writer.writerow([point['x'], point['y']])

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
