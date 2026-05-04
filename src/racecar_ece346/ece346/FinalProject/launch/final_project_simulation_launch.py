"""
Final Project — Simulation Launch

Brings up: simulator + traffic sim + visualization, PS4 joy pipeline, the
student's safety filter, and a /drive -> /control bridge so the student
publishes the same topic/message type in sim and on the truck.

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
        ece_346_share, 'config', 'final_project_simulation.yaml',
    ])

    param_file = LaunchConfiguration('param_file')

    declare_params = DeclareLaunchArgument(
        'param_file',
        default_value=default_params_file,
        description='YAML file for all FinalProject simulation nodes',
    )

    # Upstream: simulator + traffic sim + visualization
    traffic_simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                racecar_interface_share, 'launch',
                'traffic_simulation_launch.py'
            ])
        ),
        launch_arguments={'param_file': param_file}.items()
    )

    # PS4 joystick driver
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

    # Sim bridge: /drive (Ackermann) -> /control (ServoMsg)
    drive_to_servo_node = Node(
        package='racecar_ece346',
        executable='fp_drive_to_servo_node.py',
        name='drive_to_servo_node',
        output='screen',
        parameters=[param_file],
    )

    # Static obstacles: yaml preset + RViz click-to-place
    obstacle_yaml = PathJoinSubstitution([
        ece_346_share, 'config', 'static_obstacles.yaml'
    ])
    static_obs_pub_node = Node(
        package='racecar_ece346',
        executable='fp_static_obstacle_publisher_node.py',
        name='static_obstacle_publisher_node',
        output='screen',
        parameters=[param_file, {'obstacle_yaml': obstacle_yaml}],
    )

    return LaunchDescription([
        declare_params,
        traffic_simulation_launch,
        joy_node,
        joy_to_ack_node,
        safety_filter_node,
        drive_to_servo_node,
        static_obs_pub_node,
    ])
