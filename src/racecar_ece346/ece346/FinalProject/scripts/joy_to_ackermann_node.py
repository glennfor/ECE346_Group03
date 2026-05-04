#!/usr/bin/env python3
"""
Convert sensor_msgs/Joy (from a PS4 controller via `joy_node`) into
ackermann_msgs/AckermannDriveStamped on the `/teleop` topic.

Axis mapping (defaults match a typical PS4 layout reported by ros-foxy-joy):
    steering_axis  = left stick X       (axis 0, left = +1)
    throttle_axis  = left stick Y       (axis 1, up   = +1)

Speed and steering are scaled by `max_speed` and `max_steering_angle` so that
the behavior is identical between simulation and the real truck.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped


class JoyToAckermann(Node):
    def __init__(self):
        super().__init__('joy_to_ackermann_node')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('teleop_topic', '/teleop')

        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('throttle_axis', 1)
        self.declare_parameter('steering_sign', 1.0)
        self.declare_parameter('throttle_sign', 1.0)

        self.declare_parameter('max_speed', 2.0)              # m/s
        self.declare_parameter('max_steering_angle', 0.35)    # rad
        self.declare_parameter('deadzone', 0.05)

        joy_topic = self.get_parameter('joy_topic').value
        teleop_topic = self.get_parameter('teleop_topic').value

        self.steering_axis = self.get_parameter('steering_axis').value
        self.throttle_axis = self.get_parameter('throttle_axis').value
        self.steering_sign = self.get_parameter('steering_sign').value
        self.throttle_sign = self.get_parameter('throttle_sign').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.deadzone = self.get_parameter('deadzone').value

        self.pub = self.create_publisher(AckermannDriveStamped, teleop_topic, 1)
        self.sub = self.create_subscription(Joy, joy_topic, self.joy_cb, 1)

        self.get_logger().info(
            f"joy_to_ackermann: {joy_topic} -> {teleop_topic} "
            f"(max_speed={self.max_speed}, max_steer={self.max_steering_angle})"
        )

    @staticmethod
    def _apply_deadzone(v, dz):
        return 0.0 if abs(v) < dz else v

    def joy_cb(self, msg: Joy):
        if len(msg.axes) <= max(self.steering_axis, self.throttle_axis):
            return

        steer_raw = self._apply_deadzone(msg.axes[self.steering_axis], self.deadzone)
        throt_raw = self._apply_deadzone(msg.axes[self.throttle_axis], self.deadzone)

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.steering_angle = float(
            self.steering_sign * steer_raw * self.max_steering_angle
        )
        out.drive.speed = float(
            self.throttle_sign * throt_raw * self.max_speed
        )
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JoyToAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
