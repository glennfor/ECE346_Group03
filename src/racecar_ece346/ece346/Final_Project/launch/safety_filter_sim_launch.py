from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ece_share = FindPackageShare("racecar_ece346")
    interface_share = FindPackageShare("racecar_interface")
    routing_share = FindPackageShare("racecar_routing")

    default_params_file = PathJoinSubstitution(
        [ece_share, "config", "safety_filter_sim.yaml"]
    )
    param_file = LaunchConfiguration("param_file")

    declare_params = DeclareLaunchArgument(
        "param_file",
        default_value=default_params_file,
        description="Safety filter simulator parameters",
    )

    simulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([interface_share, "launch", "traffic_simulation_launch.py"])
        ),
        launch_arguments={"param_file": param_file}.items(),
    )

    routing_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([routing_share, "launch", "routing_launch.py"])
        ),
        launch_arguments={"param_file": param_file}.items(),
    )

    safety_filter_node = Node(
        package="racecar_ece346",
        executable="safety_filter_node.py",
        name="safety_filter_node",
        output="screen",
        parameters=[param_file],
    )

    joy_to_servo_node = Node(
        package="racecar_ece346",
        executable="joy_to_servo_node.py",
        name="joy_to_servo_node",
        output="screen",
        parameters=[param_file],
    )

    safety_viz_node = Node(
        package="racecar_ece346",
        executable="safety_viz_node.py",
        name="safety_viz_node",
        output="screen",
        parameters=[param_file],
    )

    return LaunchDescription(
        [
            declare_params,
            simulator_launch,
            routing_launch,
            joy_to_servo_node,
            safety_filter_node,
            safety_viz_node,
        ]
    )

