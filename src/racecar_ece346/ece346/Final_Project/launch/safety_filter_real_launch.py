from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ece_share = FindPackageShare("racecar_ece346")
    default_params_file = PathJoinSubstitution(
        [ece_share, "config", "safety_filter_real.yaml"]
    )
    param_file = LaunchConfiguration("param_file")

    declare_params = DeclareLaunchArgument(
        "param_file",
        default_value=default_params_file,
        description="Safety filter real-truck parameters",
    )

    safety_filter_node = Node(
        package="racecar_ece346",
        executable="safety_filter_node.py",
        name="safety_filter_node",
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

    return LaunchDescription([declare_params, safety_filter_node, safety_viz_node])

