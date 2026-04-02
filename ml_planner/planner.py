from collections import deque
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import torch
from torchvision import transforms

from ml_planner.image_utils import preprocess_policy_image
from ml_planner.models import ActionNormalizer, DiffusionPolicy
from ml_planner.placenav.place_recognition import PlaceRecognition
from ml_planner.zed_api_utils import ZED_API_Utils


class PlannerNode(Node):
    COMMAND_LABELS = ['roadside', 'straight', 'left', 'right']

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
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.cv_bridge = CvBridge()
        self.placenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((85, 85), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.create_subscription(Bool, '/autonomous', self.autonomous_callback, qos_profile_system_default)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile_system_default)
        self.debug_image_pub = self.create_publisher(Image, '/ml_planner/place_recognition_debug', qos_profile_system_default)
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
        self.num_inference_steps = int(self.get_parameter('num_inference_steps').value)
        self.n_obs_steps = int(self.get_parameter('n_obs_steps').value)
        self.pred_horizon = int(self.get_parameter('pred_horizon').value)
        self.policy_image_size = int(self.get_parameter('policy_image_size').value)

    def init_torch_model(self):
        package_root = Path(get_package_share_directory('ml_planner')).parents[3] / 'src' / 'ml_planner'
        checkpoint_path = package_root / 'weights' / self.model_path
        self.placenav_weight_path = package_root / 'weights' / self.placenet_model_name
        self.topomap_path = package_root / 'config' / self.topomap_name / 'topomap.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_config = checkpoint['model_config']
        self.model = DiffusionPolicy(
            action_dim=model_config['action_dim'],
            pred_horizon=model_config['pred_horizon'],
            diffusion_step_embed_dim=model_config.get('diffusion_step_embed_dim', 256),
            global_cond_dim=model_config.get('global_cond_dim', 512),
            down_dims=tuple(model_config.get('down_dims', (64, 128, 256))),
            kernel_size=model_config.get('kernel_size', 5),
            n_groups=model_config.get('n_groups', 8),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.normalizer = ActionNormalizer()
        self.normalizer.load_state_dict(checkpoint['normalizer'])

        self.scheduler = DDIMScheduler.from_config(checkpoint['scheduler_config'])

    def autonomous_callback(self, msg):
        self.autonomous_flag = msg.data

    def timer_callback(self):
        if not self.autonomous_flag or not self.zed.grab():
            return

        image = self.zed.get_image()
        policy_image = self.preprocess_policy_image(image)
        placenet_image_tensor = self.preprocess_placenet_image(image)

        self.obs_buffer.append(policy_image)
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.appendleft(policy_image.clone())

        command, idx = self.placenav.get_recognition(placenet_image_tensor)
        self.get_logger().info(f'place recognition: command={command}, idx={idx}')
        self.publish_place_recognition_debug_image(image, command)

        obs_images = torch.stack(list(self.obs_buffer), dim=0).unsqueeze(0).to(self.device, dtype=torch.float32)
        command_tensor = self.preprocess_command(command)

        with torch.no_grad():
            normalized_actions = self.model.sample_actions(
                obs_images=obs_images,
                command=command_tensor,
                noise_scheduler=self.scheduler,
                num_inference_steps=self.num_inference_steps,
            )
            actions = self.normalizer.denormalize(normalized_actions)

        self.publish_velocity(actions[0, 0, 0].item())

    def preprocess_policy_image(self, image):
        return preprocess_policy_image(image, self.policy_image_size)

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

    def preprocess_command(self, command):
        command_tensor = torch.zeros((1, len(self.COMMAND_LABELS)), device=self.device, dtype=torch.float32)
        command_tensor[0, int(command)] = 1.0
        return command_tensor

    def publish_velocity(self, angular_z: float):
        twist = Twist()
        twist.linear.x = self.linear_vel
        twist.angular.z = float(angular_z)
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
