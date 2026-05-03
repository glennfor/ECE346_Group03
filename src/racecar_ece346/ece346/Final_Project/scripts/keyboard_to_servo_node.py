#!/usr/bin/env python3
"""
Minimal keyboard teleop: hold W to go forward, release to stop.
No pynput needed — works in any terminal inside the container.
"""
import sys
import tty
import termios
import select

import rclpy
from rclpy.node import Node
from racecar_msgs.msg import ServoMsg


class KeyboardToServoNode(Node):
    def __init__(self):
        super().__init__('keyboard_to_servo_node')
        self.declare_parameter('human_control_topic', '/human_control')
        self.declare_parameter('throttle_scale', 1.0)
        self.declare_parameter('publish_rate_hz', 30.0)

        topic = self.get_parameter('human_control_topic').value
        self._scale = float(self.get_parameter('throttle_scale').value)
        rate = float(self.get_parameter('publish_rate_hz').value)

        self._pub = self.create_publisher(ServoMsg, topic, 1)
        self._timer = self.create_timer(1.0 / rate, self._tick)

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)

        self._throttle = 0.0
        self.get_logger().info('Hold W to go forward. Ctrl-C or Q to quit.')

    def _tick(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            key = sys.stdin.read(1)
            if key in ('\x03', 'q', 'Q'):
                self._restore()
                rclpy.shutdown()
                return
            self._throttle = 1.0 if key in ('w', 'W') else 0.0
        else:
            self._throttle = 0.0

        msg = ServoMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = float(self._scale * self._throttle)
        msg.steer = 0.0
        msg.reverse = False
        self._pub.publish(msg)

    def _restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def destroy_node(self):
        self._restore()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardToServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
