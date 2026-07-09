from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import torch
from torchvision import transforms

from ml_planner.zed_api_utils import ZED_API_Utils
from ml_planner.placenav.place_recognition import PlaceRecognition


class PlannerNode(Node):
    COMMAND_LABELS = ['roadside', 'straight', 'left', 'right']
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

        self.autonomous_flag = False
        self.command = 0
        self.cv_bridge = CvBridge()
        self.placenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((85, 85), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.create_subscription(Bool, '/autonomous', self.autonomous_callback, qos_profile_system_default)
        self.path_pub = self.create_publisher(NavPath, '/ml_planner/path', qos_profile_system_default)
        self.debug_image_pub = self.create_publisher(Image, '/ml_planner/place_recognition_debug', qos_profile_system_default)
        self.create_timer(self.interval_ms / 1000.0, self.timer_callback)

    def init_ros_parameter(self):
        self.path_frame_id = self.get_parameter('path_frame_id').value
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

        command, idx = self.placenav.get_recognition(placenet_image_tensor)
        self.get_logger().info(f'place recognition: command={command}, idx={idx}')
        self.publish_place_recognition_debug_image(image, command)
        command_tensor = self.preprocess_command(command)

        with torch.no_grad():
            output = self.model(image_tensor, command_tensor)

        self.publish_path(output)

    def preprocess_image(self, image):
        image = image[..., :3]
        image = image[:, 112:400, :]   # 400 - 112 = 288
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
        return image_tensor.to(self.device, dtype=torch.float32) / 255.0

    def preprocess_placenet_image(self, image):
        image = image[..., :3]
        image = cv2.cvtColor(image[:, 112:400, :], cv2.COLOR_BGR2RGB)
        placenet_image_tensor = self.placenet_transform(image).unsqueeze(0)
        return placenet_image_tensor.to(self.device, dtype=torch.float32)

    def publish_place_recognition_debug_image(self, image, command):
        image = image[..., :3]
        image = cv2.resize(image[:, 112:400, :], (340, 340), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(image, (0, 260), (340, 340), (30, 30, 30), -1)
        for idx, label in enumerate(self.COMMAND_LABELS):
            x = 10 + idx * 82
            color = (0, 255, 0) if idx == int(command) else (100, 100, 100)
            cv2.rectangle(image, (x, 275), (x + 72, 325), color, -1)
            cv2.rectangle(image, (x, 275), (x + 72, 325), (255, 255, 255), 2)
            cv2.putText(image, label, (x + 4, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8'))

    def preprocess_command(self, command=None):
        command_tensor = torch.zeros((1, self.NUM_BRANCHES), device=self.device, dtype=torch.float32)
        command_idx = self.command if command is None else int(command)
        command_tensor[0, command_idx] = 1.0
        return command_tensor
    
    def publish_path(self, output):
        path_msg = NavPath()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.path_frame_id
        for x, y in output.squeeze(0).cpu().numpy():
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)


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
