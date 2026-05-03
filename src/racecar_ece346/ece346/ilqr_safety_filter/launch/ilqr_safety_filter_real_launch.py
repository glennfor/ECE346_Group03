from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ece_share = FindPackageShare("racecar_ece346")

    default_params_file = PathJoinSubstitution(
        [ece_share, "config", "ilqr_safety_filter_real.yaml"]
    )
    param_file = LaunchConfiguration("param_file")

    declare_params = DeclareLaunchArgument(
        "param_file",
        default_value=default_params_file,
        description="ILQR safety filter real-truck parameters",
    )

    safety_filter_node = Node(
        package="racecar_ece346",
        executable="ilqr_safety_filter_node.py",
        name="ilqr_safety_filter_node",
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

    viz_node = Node(
        package="racecar_ece346",
        executable="ilqr_safety_viz_node.py",
        name="ilqr_safety_viz_node",
        output="screen",
        parameters=[param_file],
    )

    return LaunchDescription(
        [
            declare_params,
            joy_to_servo_node,
            safety_filter_node,
            viz_node,
        ]
    )
