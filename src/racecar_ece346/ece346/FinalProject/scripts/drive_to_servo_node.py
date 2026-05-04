#!/usr/bin/env python3
"""
Simulation-only bridge: ackermann_msgs/AckermannDriveStamped on `/drive` ->
racecar_msgs/ServoMsg on `/control`.
"""

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from racecar_msgs.msg import ServoMsg


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DriveToServo(Node):
    def __init__(self):
        super().__init__('drive_to_servo_node')

        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('odom_topic', '/slam_pose')

        # Defaults target the Bicycle4D sim (which accepts ±5 m/s² natively
        # and has no onboard velocity controller). The truck yaml overrides
        # `a_max` and `speed_kp` back to its physical limits.
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('a_max', 5.0)
        self.declare_parameter('delta_max', 0.31)

        # Velocity-loop gain: accel = Kp * (v_target - v_current)
        self.declare_parameter('speed_kp', 5.0)

        drive_topic = self.get_parameter('drive_topic').value
        control_topic = self.get_parameter('control_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.v_max = self.get_parameter('v_max').value
        self.a_max = self.get_parameter('a_max').value
        self.delta_max = self.get_parameter('delta_max').value
        self.kp = self.get_parameter('speed_kp').value

        self.v_current = 0.0

        self.pub = self.create_publisher(ServoMsg, control_topic, 1)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 1)
        self.create_subscription(AckermannDriveStamped, drive_topic, self._drive_cb, 1)

        self.get_logger().info(
            f"drive_to_servo: {drive_topic} -> {control_topic} "
            f"(v_max={self.v_max}, a_max={self.a_max}, delta_max={self.delta_max})"
        )

    def _odom_cb(self, msg: Odometry):
        self.v_current = msg.twist.twist.linear.x

    def _drive_cb(self, msg: AckermannDriveStamped):
        # Velocity loop (sim has no onboard velocity controller)
        v_target = _clamp(msg.drive.speed, 0.0, self.v_max)
        accel = _clamp(self.kp * (v_target - self.v_current), -self.a_max, self.a_max)

        steer = _clamp(msg.drive.steering_angle, -self.delta_max, self.delta_max)

        servo = ServoMsg()
        servo.header = msg.header
        servo.throttle = float(accel)
        servo.steer = float(steer)
        servo.reverse = False
        self.pub.publish(servo)


def main(args=None):
    rclpy.init(args=args)
    node = DriveToServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
