from pathlib import Path

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Bool
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
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile_system_default)
        self.create_timer(self.interval_ms / 1000.0, self.timer_callback)

    def init_ros_parameter(self):
        self.declare_parameter('linear_vel', 1.0)
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('interval_ms', 100)

        self.linear_vel = float(self.get_parameter('linear_vel').value)
        self.model_path = self.get_parameter('model_name').value
        self.interval_ms = int(self.get_parameter('interval_ms').value)

    def init_torch_model(self):
        planner_dir = Path(__file__).parent
        package_root = planner_dir.parent
        weight_path = package_root / 'weights' / self.model_path

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

        with torch.no_grad():
            output = self.model(image_tensor)

        self.publisher_vel(output)

    def preprocess_image(self, image):
        image = image[..., :3]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
        return image_tensor.to(self.device, dtype=torch.float32)
    
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
