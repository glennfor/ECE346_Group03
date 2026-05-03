#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from racecar_msgs.msg import ServoMsg


class JoyToServoNode(Node):
    def __init__(self):
        super().__init__("joy_to_servo_node")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("human_control_topic", "/human_control")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("throttle_axis", 1)
        self.declare_parameter("steer_axis", 0)
        self.declare_parameter("steer_axis_alt", 3)
        self.declare_parameter("deadman_button", 4)
        self.declare_parameter("throttle_scale", 1.0)
        self.declare_parameter("steer_scale", 0.35)
        self.declare_parameter("throttle_offset", 0.0)
        self.declare_parameter("invert_throttle", False)
        self.declare_parameter("invert_steer", False)

        self.joy_topic = self.get_parameter("joy_topic").value
        self.human_control_topic = self.get_parameter("human_control_topic").value
        self.publish_rate_hz = self.get_parameter("publish_rate_hz").value
        self.throttle_axis = self.get_parameter("throttle_axis").value
        self.steer_axis = self.get_parameter("steer_axis").value
        self.steer_axis_alt = self.get_parameter("steer_axis_alt").value
        self.deadman_button = self.get_parameter("deadman_button").value
        self.throttle_scale = self.get_parameter("throttle_scale").value
        self.steer_scale = self.get_parameter("steer_scale").value
        self.throttle_offset = self.get_parameter("throttle_offset").value
        self.invert_throttle = self.get_parameter("invert_throttle").value
        self.invert_steer = self.get_parameter("invert_steer").value

        self.latest_msg = None
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        self.control_pub = self.create_publisher(ServoMsg, self.human_control_topic, 1)
        self.timer = self.create_timer(1.0 / float(self.publish_rate_hz), self.publish_control)

        self.get_logger().info(
            f"joy_to_servo_node ready: {self.joy_topic} -> {self.human_control_topic}, "
            f"axes throttle={self.throttle_axis}, steer={self.steer_axis}, "
            f"deadman={self.deadman_button}"
        )

    def joy_callback(self, msg: Joy):
        self.latest_msg = msg

    def publish_control(self):
        if self.latest_msg is None:
            return

        if not self._is_enabled(self.latest_msg):
            throttle = 0.0
            steer = 0.0
        else:
            throttle_axis = self._axis(self.latest_msg, self.throttle_axis)
            steer_axis = self._steer_axis(self.latest_msg)
            if self.invert_throttle:
                throttle_axis *= -1.0
            if self.invert_steer:
                steer_axis *= -1.0

            throttle = self.throttle_offset + self.throttle_scale * throttle_axis
            steer = self.steer_scale * steer_axis

        msg = ServoMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = float(max(-1.0, min(1.0, throttle)))
        msg.steer = float(max(-1.0, min(1.0, steer)))
        msg.reverse = msg.throttle < 0.0
        self.control_pub.publish(msg)

    def _is_enabled(self, msg: Joy) -> bool:
        if self.deadman_button < 0:
            return True
        if self.deadman_button >= len(msg.buttons):
            return False
        return bool(msg.buttons[self.deadman_button])

    def _steer_axis(self, msg: Joy) -> float:
        primary = self._axis(msg, self.steer_axis)
        alternate = self._axis(msg, self.steer_axis_alt)
        if abs(alternate) > abs(primary):
            return alternate
        return primary

    @staticmethod
    def _axis(msg: Joy, idx: int) -> float:
        if idx < 0 or idx >= len(msg.axes):
            return 0.0
        return float(msg.axes[idx])


def main(args=None):
    rclpy.init(args=args)
    node = JoyToServoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

