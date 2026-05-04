#!/usr/bin/env python3
"""
safety_filter_qp_node — heuristic safety filter with smooth steering blend.

Uses the safety value h from safety_monitor_node:

  h >= throttle_cap_margin  →  passthrough (human controls freely)
  0 <= h < throttle_cap_margin  →  WARNING BAND
      - full human throttle (still h >= 0 geometrically)
      - steering blended toward backup (lateral nudge back to center)
  h < 0  →  UNSAFE: throttle capped to non-positive; backup steering

The steering blend in the warning band nudges heading without blocking
forward acceleration while margins are still non-negative.

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
        self._omega_blend_prev = 0.0
        self._prev_commanded_delta: float = 0.0

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
        # Seed the rate-limiter from the actual current delta on first tick.
        if self._prev_commanded_delta == 0.0 and abs(state[4]) > 1e-4:
            self._prev_commanded_delta = float(state[4])

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
            u_out, override = self._filter(u_human)
        except Exception:
            self.get_logger().error(f"filter fault:\n{traceback.format_exc()}")
            u_out = u_human
            override = False

        self._publish(u_out, state, override, u_human)

    def _filter(self, u_human: np.ndarray):
        h = self.last_h
        p = self.params

        # Fixed warning-band threshold. The dual rollout in the monitor already
        # captures whether the backup policy can stop safely — effective_margin
        # must NOT scale with speed here because stopping_dist can exceed the
        # lane's max possible margin, making the filter permanently active.
        effective_margin = p.throttle_cap_margin

        backup_omega = float(self.last_backup_u0[1]) if self.last_backup_u0 is not None else 0.0

        if h >= effective_margin:
            self._omega_blend_prev = float(u_human[1])
            return u_human.copy(), False

        if h < 0:
            # Hard violation: cap forward thrust and take full backup steering.
            a_out = min(u_human[0], 0.0)
            omega_raw = backup_omega
        else:
            # Warning band (0 <= h < effective_margin): steer toward backup only.
            # Throttle is left unchanged — blending toward a_min here causes
            # active braking in the warning band, which combined with a long
            # backup-rollout horizon gives a speed-dependent brake that
            # permanently caps the truck as velocity grows.
            a_out = float(u_human[0])
            alpha = h / effective_margin
            omega_raw = (1.0 - alpha) * backup_omega + alpha * float(u_human[1])

        tau = p.steer_blend_lpf_tau_s
        if tau > 0.0:
            b = p.dt / (tau + p.dt)
            omega_raw = b * omega_raw + (1.0 - b) * self._omega_blend_prev
        self._omega_blend_prev = omega_raw
        return np.array([a_out, omega_raw]), True

    def _publish(self, u: np.ndarray, state: np.ndarray, override: bool, u_human):
        stamp = self.get_clock().now().to_msg()
        p = self.params
        # Hard rate-limit on steering angle to prevent teleportation on mode switch.
        # Clamp the commanded delta to ±max_delta_rate_rad_s * dt from last cycle.
        raw_delta = float(np.clip(state[4] + u[1] * p.dt, p.delta_min, p.delta_max))
        max_step = p.max_delta_rate_rad_s * p.dt
        limited_delta = float(np.clip(raw_delta,
                                      self._prev_commanded_delta - max_step,
                                      self._prev_commanded_delta + max_step))
        self._prev_commanded_delta = limited_delta
        u = u.copy()
        u[1] = (limited_delta - state[4]) / p.dt

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
