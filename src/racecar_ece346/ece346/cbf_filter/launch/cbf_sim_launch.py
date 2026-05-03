"""
cbf_sim_launch.py — full CBF safety filter in simulation.

Usage:
    # Full system with QP intervention enabled:
    ros2 launch racecar_ece346 cbf_sim_launch.py

    # Watch safety value without any intervention (passthrough):
    ros2 launch racecar_ece346 cbf_sim_launch.py enable_qp:=false

    # Custom parameters:
    ros2 launch racecar_ece346 cbf_sim_launch.py param_file:=/path/to/my.yaml

    # Toggle intervention at runtime (no restart needed):
    ros2 param set /safety_filter_qp_node enable_qp true

Nodes launched:
    simulator + traffic_simulation  (racecar_interface)
    routing                         (racecar_routing)
    joy_to_servo_node               joy → /human_control
    backup_planner_node             /slam_pose → /safety/backup_traj + /safety/backup_u0
    safety_monitor_node             trajectory + obstacles → /safety/value + /safety/grad
    safety_filter_qp_node           human + safety value → /control
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

        # Node 2 — MONITOR: evaluate safety margins, compute h_imp + gradient
        Node(
            package="racecar_ece346",
            executable="cbf_safety_monitor_node.py",
            name="safety_monitor_node",
            output="screen",
            parameters=[param_file],
        ),

        # Node 3 — INTERVENTIONER: CBF-QP, outputs /control
        Node(
            package="racecar_ece346",
            executable="cbf_safety_filter_qp_node.py",
            name="safety_filter_qp_node",
            output="screen",
            parameters=[param_file, {"enable_qp": ParameterValue(enable_qp, value_type=bool)}],
        ),
    ])
