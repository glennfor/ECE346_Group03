#!/usr/bin/env python3
"""
Final Project — Safety Filter (STUDENT SKELETON)

You will build a ROS2 node that sits between a human driver (PS4 joystick)
and the vehicle, intervening only when necessary to keep the car safe.

Pipeline:

    /joy --> joy_to_ackermann --> /teleop ┐
                                           │
                             /SLAM/Pose ───┤---> [safety_filter_node] ---> /drive
                                           │
                       /Obstacles/Static ──┘

Topics you will work with:

    /teleop            ackermann_msgs/AckermannDriveStamped  (human command in)
    <odom_topic>       nav_msgs/Odometry                     (vehicle state in)
    /Obstacles/Static  visualization_msgs/MarkerArray        (obstacles in)
    /drive             ackermann_msgs/AckermannDriveStamped  (safe command out)

------------------------------------------------------------------------------
Task 1: Build the node plumbing (subscribers, publisher, timer).
Task 2: Implement the safety filter logic.
------------------------------------------------------------------------------

You may:
  - Add imports, helper methods, and ROS parameters to THIS file.
  - Add your own files anywhere under FinalProject/ — e.g. an `ilqr/`
    subfolder with your ILQR solver, an `ilqr_params.yaml` with cost
    weights, utility modules, etc. Load yaml configs from within this
    node using `open()` + `yaml.safe_load()`, or declare a ROS param
    for the config path and set it in `final_project_*.yaml`.
  - Edit any yaml under `FinalProject/config/` — add parameters, tune
    thresholds, change topic names. All existing yaml values are
    documented and intended to be tunable.

You should NOT need to rewrite the launch files, the plumbing nodes
(joy_to_ackermann, drive_to_servo, etc.), or the CMakeLists top-level
install lists. Only touch those if you are adding a new standalone
executable — in which case ask first.
"""

import math

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray


def yaw_from_quat(qx, qy, qz, qw):
    """Extract yaw (heading, rad) from a quaternion. Useful for Task 2."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SafetyFilterNode(Node):

    # =========================================================================
    # TASK 1 — Node setup (subscribers, publisher, timer)
    # =========================================================================
    #
    # Fill in __init__ below so that the node:
    #   1. Declares ROS parameters for each topic name and for the publish rate.
    #      Hint: use self.declare_parameter('name', default_value). The yaml
    #      file (final_project_*.yaml) will override these at launch time.
    #      Required parameter names (match the yaml):
    #          teleop_topic, drive_topic, odom_topic, static_obs_topic,
    #          publish_rate
    #
    #   2. Creates three subscribers that cache the latest message each:
    #          /teleop            -> AckermannDriveStamped
    #          <odom_topic>       -> Odometry
    #          /Obstacles/Static  -> MarkerArray
    #      Hint: self.create_subscription(MsgType, topic, callback, queue_size)
    #      Each callback can be a one-liner that stores msg into an instance
    #      attribute (e.g. self._latest_teleop = msg).
    #
    #   3. Creates one publisher:
    #          /drive             -> AckermannDriveStamped
    #      Hint: self.create_publisher(MsgType, topic, queue_size)
    #
    #   4. Creates a timer at `publish_rate` Hz that calls a method which
    #      invokes self.safety_filter(...) and publishes the result.
    #      Hint: self.create_timer(period_sec, callback)
    #
    # For reference, open any other node in this repo (e.g.
    # FinalProject/scripts/joy_to_ackermann_node.py) to see the same pattern.
    # =========================================================================

    def __init__(self):
        super().__init__('safety_filter_node')

        # ---- TODO(Task 1.1): declare ROS parameters ----


        # ---- TODO(Task 1.2): read parameter values ----

        self._latest_teleop = None     # AckermannDriveStamped
        self._latest_odom = None       # Odometry
        self._latest_obs = None        # MarkerArray

        # ---- TODO(Task 1.3): create subscribers ----


        # ---- TODO(Task 1.4): create the publisher ----

        # ---- TODO(Task 1.5): create a timer at publish_rate Hz ----

        self.get_logger().info(
            f"safety_filter_node ready: {teleop_topic} + {odom_topic} "
            f"+ {obs_topic} -> {drive_topic}"
        )

    # ---- TODO(Task 1.6): implement callbacks ----


    # ---- TODO(Task 1.7): the timer callback ----
    # Should:
    #   - return early if no teleop has arrived yet (self._latest_teleop is None)
    #   - call self.safety_filter(teleop=..., odom=..., obstacles=...)
    #   - if the returned command is not None, update its header.stamp to now
    #     and publish it on /drive
    #
    # def _publish_filtered(self):
    #   

    # =========================================================================
    # TASK 2 — Safety filter implementation
    # =========================================================================
    #
    # Start as a passthrough, then add real safety logic.
    #
    # You are free to pick any approach (or your own). Add helper methods,
    # extra parameters, even a sub-folder of modules — this skeleton will
    # get out of the way.
    # =========================================================================

    def safety_filter(self, teleop, odom, obstacles):
        """
        Args
        ----
        teleop : ackermann_msgs.msg.AckermannDriveStamped   (or None)
            Human's desired command. Useful fields:
                teleop.drive.speed           float   m/s     target forward speed
                teleop.drive.steering_angle  float   rad     target steering angle
                teleop.drive.acceleration    float   m/s^2   usually 0 from the joy
                teleop.header.stamp          Time            when the command was issued

        odom : nav_msgs.msg.Odometry                        (or None)
            Vehicle state. Useful fields:
                odom.pose.pose.position.x       float   m       map-frame x
                odom.pose.pose.position.y       float   m       map-frame y
                odom.pose.pose.position.z       float   m       usually 0
                odom.pose.pose.orientation      Quaternion (.x .y .z .w)
                    → use yaw_from_quat(..) above to get heading in rad
                odom.twist.twist.linear.x       float   m/s     forward velocity
                odom.twist.twist.angular.z      float   rad/s   yaw rate

        obstacles : visualization_msgs.msg.MarkerArray      (or None)
            Static obstacles (cubes). Useful fields:
                obstacles.markers               list[Marker]
                for m in obstacles.markers:
                    m.pose.position.x / .y / .z float   m       obstacle center
                    m.scale.x / .y / .z         float   m       cube size (x=y=z typically)
                    m.id                        int             obstacle id
                    m.ns                        str             namespace

        Returns
        -------
        ackermann_msgs.msg.AckermannDriveStamped
            The command to publish on /drive. Set `.drive.speed` (m/s) and
            `.drive.steering_angle` (rad). Header.stamp is overwritten for you.
            Return None to skip publishing this tick.
        """
        # ---- TODO(Task 2): replace this passthrough ----
        return teleop


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
