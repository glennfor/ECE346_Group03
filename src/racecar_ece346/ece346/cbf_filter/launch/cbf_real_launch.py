"""
cbf_real_launch.py — Heuristic safety filter (margins + throttle cap + steer blend) on the real truck.

Usage:
    # Start with passthrough to verify topics are live:
    ros2 launch racecar_ece346 cbf_real_launch.py enable_qp:=false

    # Then enable intervention (no restart needed):
    ros2 param set /safety_filter_qp_node enable_qp true

    # Or launch with intervention enabled from the start:
    ros2 launch racecar_ece346 cbf_real_launch.py

Notes:
    - No simulator here; /slam_pose comes from the real truck's localization.
    - Uses cbf_filter_real.yaml by default (lower speed, larger safety margins).
    - Before the first run: verify wheelbase_m and measure stopping time T_stop.
      Then set horizon_H = ceil(1.5 * T_stop / 0.05) in cbf_filter_real.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ece_share = FindPackageShare("racecar_ece346")
    routing_share = FindPackageShare("racecar_routing")

    default_params = PathJoinSubstitution([ece_share, "config", "cbf_filter_real.yaml"])

    param_file = LaunchConfiguration("param_file")
    enable_qp = LaunchConfiguration("enable_qp")

    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file", default_value=default_params,
            description="YAML parameter file for all CBF nodes",
        ),
        DeclareLaunchArgument(
            "enable_qp", default_value="true",
            description="true=CBF intervention; false=passthrough",
        ),

        # Lane geometry server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([routing_share, "launch", "routing_launch.py"])
            ),
            launch_arguments={"param_file": param_file}.items(),
        ),

        # Joystick → /human_control
        Node(
            package="racecar_ece346",
            executable="joy_to_servo_node.py",
            name="joy_to_servo_node",
            output="screen",
            parameters=[param_file],
        ),

        Node(
            package="racecar_ece346",
            executable="cbf_backup_planner_node.py",
            name="backup_planner_node",
            output="screen",
            parameters=[param_file],
        ),

        Node(
            package="racecar_ece346",
            executable="cbf_safety_monitor_node.py",
            name="safety_monitor_node",
            output="screen",
            parameters=[param_file],
        ),

        Node(
            package="racecar_ece346",
            executable="cbf_safety_filter_qp_node.py",
            name="safety_filter_qp_node",
            output="screen",
            parameters=[param_file, {"enable_qp": ParameterValue(enable_qp, value_type=bool)}],
        ),
    ])
