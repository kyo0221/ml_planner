import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('ml_planner')
    main_param_path = os.path.join(package_share_dir, 'config', 'main_param.yaml')

    param_file = LaunchConfiguration('param_file')
    declare_param_file = DeclareLaunchArgument(
        'param_file',
        default_value=main_param_path,
        description='Path to parameter file for planner_node',
    )

    planner_node = Node(
        package='ml_planner',
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=[param_file],
    )

    launch_description = LaunchDescription()
    launch_description.add_entity(declare_param_file)
    launch_description.add_entity(planner_node)

    return launch_description
