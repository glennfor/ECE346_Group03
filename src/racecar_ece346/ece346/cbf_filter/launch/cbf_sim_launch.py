"""
cbf_sim_launch.py — heuristic safety filter in simulation.

Nodes:
  simulator + traffic, routing, joy_to_servo
  backup_planner_node      /slam_pose, /control → /safety/backup_u0
  safety_monitor_node      margins + lookahead → /safety/value
  safety_filter_qp_node    human + h + backup → /control (throttle cap + steer blend)
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
    iface_share = FindPackageShare("racecar_interface")
    routing_share = FindPackageShare("racecar_routing")

    default_params = PathJoinSubstitution([ece_share, "config", "cbf_filter_sim.yaml"])

    param_file = LaunchConfiguration("param_file")
    enable_qp = LaunchConfiguration("enable_qp")

    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file", default_value=default_params,
            description="YAML parameter file for all CBF nodes",
        ),
        DeclareLaunchArgument(
            "enable_qp", default_value="true",
            description="true=CBF intervention active; false=passthrough (safety value still computed)",
        ),

        # Simulator + traffic obstacles
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([iface_share, "launch", "traffic_simulation_launch.py"])
            ),
            launch_arguments={"param_file": param_file}.items(),
        ),

        # Lanelet2 routing / lane geometry server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([routing_share, "launch", "routing_launch.py"])
            ),
            launch_arguments={"param_file": param_file}.items(),
        ),

        # Raw joystick driver
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
            parameters=[param_file],
        ),

        # Joystick → /human_control
        Node(
            package="racecar_ece346",
            executable="joy_to_servo_node.py",
            name="joy_to_servo_node",
            output="screen",
            parameters=[param_file],
        ),

        # Node 1 — FALLBACK: roll out backup policy
        Node(
            package="racecar_ece346",
            executable="cbf_backup_planner_node.py",
            name="backup_planner_node",
            output="screen",
            parameters=[param_file],
        ),

        # Node 2 — MONITOR: margins + human lookahead → /safety/value
        Node(
            package="racecar_ece346",
            executable="cbf_safety_monitor_node.py",
            name="safety_monitor_node",
            output="screen",
            parameters=[param_file],
        ),

        # Node 3 — Interventions: throttle cap + steer blend on /control
        Node(
            package="racecar_ece346",
            executable="cbf_safety_filter_qp_node.py",
            name="safety_filter_qp_node",
            output="screen",
            parameters=[param_file, {"enable_qp": ParameterValue(enable_qp, value_type=bool)}],
        ),

        # Safety visualization (RViz markers)
        Node(
            package="racecar_ece346",
            executable="safety_viz_node.py",
            name="safety_viz_node",
            output="screen",
            parameters=[param_file],
        ),
    ])
