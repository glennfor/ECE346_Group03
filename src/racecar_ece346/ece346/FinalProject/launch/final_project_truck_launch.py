"""
Final Project — Real Truck Launch

Brings up: visualization + AprilTag obstacle detection, PS4 joy pipeline,
and the student's safety filter. The truck's f1tenth_stack provides
control_gate and the VESC bridge; it consumes AckermannDriveStamped on
/drive when R2 is held (autonomous mode).

Students should not need to modify this file.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ece_346_share = FindPackageShare('racecar_ece346')
    racecar_interface_share = FindPackageShare('racecar_interface')

    default_params_file = PathJoinSubstitution([
        ece_346_share, 'config', 'final_project_truck.yaml',
    ])

    param_file = LaunchConfiguration('param_file')

    declare_params = DeclareLaunchArgument(
        'param_file',
        default_value=default_params_file,
        description='YAML file for all FinalProject truck nodes',
    )

    # Visualization (RViz + routing + viz node — no simulator)
    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                racecar_interface_share, 'launch', 'visualization_launch.py'
            ])
        ),
        launch_arguments={'param_file': param_file}.items()
    )

    # AprilTag -> /Obstacles/Static
    obstacle_detection_node = Node(
        package='racecar_ece346',
        executable='fp_obstacle_detection_node.py',
        name='static_obstacle_detection_node',
        output='screen',
        parameters=[param_file],
    )

    # PS4 joystick driver (runs on the student laptop)
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[param_file],
    )

    # Joy -> AckermannDriveStamped on /teleop
    joy_to_ack_node = Node(
        package='racecar_ece346',
        executable='fp_joy_to_ackermann_node.py',
        name='joy_to_ackermann_node',
        output='screen',
        parameters=[param_file],
    )

    # Student's safety filter: /teleop (+ pose + obstacles) -> /drive
    safety_filter_node = Node(
        package='racecar_ece346',
        executable='fp_safety_filter_node.py',
        name='safety_filter_node',
        output='screen',
        parameters=[param_file],
    )

    return LaunchDescription([
        declare_params,
        visualization_launch,
        obstacle_detection_node,
        joy_node,
        joy_to_ack_node,
        safety_filter_node,
    ])
