#!/usr/bin/env python3
"""
safety_filter_qp_node — simple heuristic safety filter.

Uses the safety value h from safety_monitor_node to decide how to
modify (or pass through) the human joystick command:

  h >= throttle_cap_margin  →  passthrough (truck is safe, no intervention)
  0 <= h < throttle_cap_margin  →  throttle capped to zero; human steers freely
  h < 0                     →  full backup control (brake + steer to center)

No QP, no gradients, no finite differences.

Published topics:
  /control           ServoMsg          — filtered command to truck
  /safety/override   Bool              — True when human input was modified
  /safety/u_human    Float64MultiArray — [a, omega] human intent (debug)
  /safety/u_filtered Float64MultiArray — [a, omega] what was sent (debug)
"""
import traceback

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray

from racecar_msgs.msg import ServoMsg

from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.ros_utils import (
    control_to_servo_msg,
    odom_to_state,
    servo_msg_to_control,
)


class SafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_qp_node")
        self.params: CbfParams = declare_and_load(self)

        self.declare_parameter("enable_qp", True)
        self.enable_qp: bool = self.get_parameter("enable_qp").value

        self.delta_estimate = 0.0
        self.last_state: np.ndarray = None
        self.last_odom_time: float = None
        self.last_human_msg = None
        self.last_human_time: float = None
        self.last_h: float = None
        self.last_safety_time: float = None
        self.last_backup_u0: np.ndarray = None

        self._setup_io()
        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info(
            f"safety_filter_node ready  enable_qp={self.enable_qp}"
        )

    def _setup_io(self):
        self.create_subscription(Odometry, self.params.odom_topic, self._odom_cb, 10)
        self.create_subscription(ServoMsg, self.params.human_control_topic, self._human_cb, 10)
        self.create_subscription(Float32, "/safety/value", self._value_cb, 1)
        self.create_subscription(Float64MultiArray, "/safety/backup_u0", self._u0_cb, 1)

        self.control_pub = self.create_publisher(ServoMsg, self.params.filtered_control_topic, 1)
        self.override_pub = self.create_publisher(Bool, "/safety/override", 1)
        self.u_human_pub = self.create_publisher(Float64MultiArray, "/safety/u_human", 1)
        self.u_filtered_pub = self.create_publisher(Float64MultiArray, "/safety/u_filtered", 1)

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9

    def _human_cb(self, msg: ServoMsg):
        self.last_human_msg = msg
        self.last_human_time = self.get_clock().now().nanoseconds * 1e-9

    def _value_cb(self, msg: Float32):
        self.last_h = float(msg.data)
        self.last_safety_time = self.get_clock().now().nanoseconds * 1e-9

    def _u0_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.last_backup_u0 = np.array(msg.data, dtype=float)

    def _step(self):
        if self.last_state is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        state = self.last_state.copy()
        p = self.params

        odom_stale = self.last_odom_time is None or now - self.last_odom_time > p.stale_timeout_s
        human_stale = self.last_human_time is None or now - self.last_human_time > p.stale_timeout_s
        if odom_stale or human_stale:
            self._publish(np.array([p.a_min * 0.5, 0.0]), state, True, None)
            return

        u_human = servo_msg_to_control(self.last_human_msg, state, p)

        self.enable_qp = self.get_parameter("enable_qp").value
        if not self.enable_qp:
            self._publish(u_human, state, False, u_human)
            return

        safety_stale = (
            self.last_h is None
            or self.last_safety_time is None
            or now - self.last_safety_time > p.stale_timeout_s
        )
        if safety_stale:
            self._publish(u_human, state, False, u_human)
            return

        try:
            u_out, override = self._filter(state, u_human)
        except Exception:
            self.get_logger().error(f"filter fault:\n{traceback.format_exc()}")
            u_out = u_human
            override = False

        self._publish(u_out, state, override, u_human)

    def _filter(self, state: np.ndarray, u_human: np.ndarray):
        h = self.last_h
        p = self.params

        if h < 0:
            # Unsafe: commit to backup control (brake + steer toward lane center).
            u_out = (
                self.last_backup_u0.copy()
                if self.last_backup_u0 is not None
                else np.array([p.a_min, 0.0])
            )
            return u_out, True

        if h < p.throttle_cap_margin:
            # Near boundary: suppress forward acceleration, human steers freely.
            u_out = u_human.copy()
            u_out[0] = min(u_out[0], 0.0)
            override = u_out[0] < u_human[0] - 1e-6
            return u_out, override

        # Safe: pass through unchanged.
        return u_human.copy(), False

    def _publish(self, u: np.ndarray, state: np.ndarray, override: bool, u_human):
        stamp = self.get_clock().now().to_msg()
        msg = control_to_servo_msg(ServoMsg, u, state, self.params, stamp)
        self.delta_estimate = float(msg.steer)
        self.control_pub.publish(msg)

        ov = Bool()
        ov.data = bool(override)
        self.override_pub.publish(ov)

        if u_human is not None:
            uh = Float64MultiArray()
            uh.data = [float(u_human[0]), float(u_human[1])]
            self.u_human_pub.publish(uh)

        uf = Float64MultiArray()
        uf.data = [float(u[0]), float(u[1])]
        self.u_filtered_pub.publish(uf)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
