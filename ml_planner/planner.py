import os
import sys
from pathlib import Path

from scipy.interpolate import splprep, splev

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from nav_msgs.msg import Path

import torch

from ml_planner.zed_api_utils import ZED_API_Utils

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.init_ros_parameter()
        self.init_torch_model()
        self.zed = ZED_API_Utils()

        self.autonomous_flag = False

        self.create_subscription(Bool, '/autonomous', self.autonomous_callback, qos_profile_system_default)
        self.path_pub = self.create_publisher(Path, '/path', qos_profile_system_default)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile_system_default)

        self.create_timer(self.interval_ms, self.timer_callback)

    def init_ros_parameter(self):
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('interval_ms', 100)

        self.model_path = self.get_parameter('model_name').value
        self.interval_ms = self.get_parameter('interval_ms').value

    def init_torch_model(self):
        planner_dir = Path(__file__).parent
        package_root = planner_dir.parent
        weight_path = package_root / 'weights' /self.model_path

        self.device = torch.device('cuda')
        self.model = torch.jit.load(weight_path, map_location=self.device)
        self.model.eval()

    def autonomous_callback(self, msg):
        self.autonomous_flag = msg.data
        self.get_logger().info(f'received autonomous flag : {self.autonomous_flag}')

    def timer_callback(self):
        if not self.zed.grab() or not self.autonomous_flag:
            return
        
        image = self.zed.get_image()
        image_tensor = self.preprocess_image(image)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='base_link')

        with torch.no_grad():
            output = self.model(image_tensor)

        path_msgs_smoothing = self.apply_bspline_smoothing(output, header)
        self.path_pub.publish(path_msgs_smoothing)

    def preprocess_image(self, image):
        image = image[..., :3]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
        return image_tensor.to(self.device, dtype=torch.float32)

    def apply_bspline_smoothing(self, output, header):
        waypoints = output.cpu().numpy().reshape(-1, 2)
        x = waypoints[:, 0]
        y = waypoints[:, 1]

        # s: smoothing factor（値が大きいほど滑らか、0だと補間）
        # k: スプラインの次数（3次）
        tck, u = splprep([x, y], s=0.1, k=3)
        u_new = np.linspace(0, 1, 30)
        x_smooth, y_smooth = splev(u_new, tck)

        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'base_link'

        path_msg.poses.append(PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=0.0, y=0.0))))
        path_msg.poses.extend(PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=float(x_smooth[i]), y=float(y_smooth[i])))) for i in range(len(x_smooth)))

        return path_msg


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
