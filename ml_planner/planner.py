from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Bool
import torch

from ml_planner.zed_api_utils import ZED_API_Utils
from ml_planner.placenav.place_recognition import PlaceRecognition


class PlannerNode(Node):
    NUM_BRANCHES = 4

    def __init__(self):
        super().__init__('planner_node', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.init_ros_parameter()
        self.init_torch_model()
        self.zed = ZED_API_Utils()
        self.placenav = PlaceRecognition(
            self.placenav_weight_path,
            self.topomap_path,
            device=self.device,
            delta=self.placenet_delta,
            window_lower=self.placenet_window_lower,
            window_upper=self.placenet_window_upper,
        )

        self.autonomous_flag = True
        self.command = 0

        self.create_subscription(Bool, '/autonomous', self.autonomous_callback, qos_profile_system_default)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile_system_default)
        self.create_timer(self.interval_ms / 1000.0, self.timer_callback)

    def init_ros_parameter(self):
        self.linear_vel = float(self.get_parameter('linear_max.vel').value)
        self.model_path = self.get_parameter('model_name').value
        self.placenet_model_name = self.get_parameter('placenet_model_name').value
        self.topomap_name = self.get_parameter('topomap_dir_name').value
        self.placenet_delta = float(self.get_parameter('placenet_delta').value)
        self.placenet_window_lower = int(self.get_parameter('placenet_window_lower').value)
        self.placenet_window_upper = int(self.get_parameter('placenet_window_upper').value)
        self.interval_ms = int(self.get_parameter('interval_ms').value)

    def init_torch_model(self):
        package_root = Path(get_package_share_directory('ml_planner')).parents[3] / 'src' / 'ml_planner'
        weight_path = package_root / 'weights' / self.model_path
        self.placenav_weight_path = package_root / 'weights' / self.placenet_model_name
        self.topomap_path = package_root / 'config' / self.topomap_name / 'topomap.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(weight_path, map_location=self.device)
        self.model.eval()

    def autonomous_callback(self, msg):
        self.autonomous_flag = msg.data

    def timer_callback(self):
        if not self.autonomous_flag or not self.zed.grab():
            return

        image = self.zed.get_image()
        image_tensor = self.preprocess_image(image)
        placenet_image_tensor = self.preprocess_placenet_image(image)

        command = self.placenav.get_recognition(placenet_image_tensor)
        command_tensor = self.preprocess_command(command)

        with torch.no_grad():
            output = self.model(image_tensor, command_tensor)

        self.publisher_vel(output)

    def preprocess_image(self, image):
        image = image[..., :3]
        image = image[:, 112:400, :]   # 400 - 112 = 288
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
        return image_tensor.to(self.device, dtype=torch.float32)

    def preprocess_placenet_image(self, image):
        image = image[..., :3]
        image = image[:, 112:400, :]
        placenet_image = cv2.resize(image, (85, 85), interpolation=cv2.INTER_AREA)
        placenet_image_tensor = torch.from_numpy(placenet_image).permute(2, 0, 1).unsqueeze(0).contiguous()
        return placenet_image_tensor.to(self.device, dtype=torch.float32)

    def preprocess_command(self, command=None):
        command_tensor = torch.zeros((1, self.NUM_BRANCHES), device=self.device, dtype=torch.float32)
        command_idx = self.command if command is None else int(command)
        command_tensor[0, command_idx] = 1.0
        return command_tensor
    
    def publisher_vel(self, output):
        twist = Twist()
        twist.linear.x = self.linear_vel
        twist.angular.z = float(output.squeeze().item())
        self.vel_pub.publish(twist)


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
