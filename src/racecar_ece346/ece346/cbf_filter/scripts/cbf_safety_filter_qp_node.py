#!/usr/bin/env python3
"""
safety_filter_qp_node — the "interventioner" in the Backup-CBF filter.

Role in the 3-node CBF filter: INTERVENTION
  Finds the control closest to the human's input that satisfies the CBF
  constraint, and publishes it to /control.

The CBF constraint (linearized):
  h_imp(f(x, u)) >= (1 - λ) * h_imp(x)

  Linearizing both h_imp and f around the current state x and human input u_h:
    g^T u >= c
  where:
    g = B^T ∇h         (B = ∂f/∂u = dt * I at accel/omega rows)
    c = -λ h - ∇h·(f(x,u_h)-x) + g·u_h

  If u_h already satisfies this, it is passed through unchanged.
  If not, the QP finds the closest safe control.
  If the QP is infeasible (rare), it falls back to the backup control u0.

Disabling intervention (passthrough mode):
  Set parameter enable_qp=false — the node forwards /human_control → /control
  unchanged but still publishes /safety/override = False.
  Useful for verifying the safety monitor before enabling intervention.
  Can be flipped at runtime:
    ros2 param set /safety_filter_qp_node enable_qp true

Published topics:
  /control              ServoMsg            — filtered command to truck
  /safety/override      Bool                — True when human input was changed
  /safety/u_human       Float64MultiArray   — [a, omega] human intent (debug)
  /safety/u_filtered    Float64MultiArray   — [a, omega] what was sent (debug)
"""
import traceback

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray

from racecar_msgs.msg import ServoMsg

from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.dynamics import control_jacobian, step
from ece346.cbf_filter.cbf_filter.qp import solve as qp_solve
from ece346.cbf_filter.cbf_filter.ros_utils import (
    control_to_servo_msg,
    odom_to_state,
    servo_msg_to_control,
)


class SafetyFilterQPNode(Node):
    def __init__(self):
        super().__init__("safety_filter_qp_node")
        self.params: CbfParams = declare_and_load(self)

        # enable_qp is not in CbfParams because it can be toggled at runtime.
        self.declare_parameter("enable_qp", True)
        self.enable_qp: bool = self.get_parameter("enable_qp").value

        self.delta_estimate = 0.0
        self.last_state: np.ndarray = None
        self.last_odom_time: float = None
        self.last_human_msg = None
        self.last_human_time: float = None
        self.last_h: float = None
        self.last_grad: np.ndarray = None
        self.last_safety_time: float = None
        self.last_backup_u0: np.ndarray = None
        self.last_grad_control: np.ndarray = None
        self.last_h_backup: float = None

        # After an override, keep the QP active for hysteresis_cycles more ticks.
        # This prevents rapid on/off flicker near the safe-set boundary.
        self.override_hold: int = 0

        self._setup_io()
        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info(
            f"safety_filter_qp_node ready  enable_qp={self.enable_qp}  λ={self.params.lambda_cbf}"
        )

    def _setup_io(self):
        self.create_subscription(Odometry, self.params.odom_topic, self._odom_cb, 10)
        self.create_subscription(ServoMsg, self.params.human_control_topic, self._human_cb, 10)
        self.create_subscription(Float32, "/safety/value", self._value_cb, 1)
        self.create_subscription(Float64MultiArray, "/safety/grad", self._grad_cb, 1)
        self.create_subscription(Float64MultiArray, "/safety/backup_u0", self._u0_cb, 1)
        self.create_subscription(Float64MultiArray, "/safety/grad_control", self._grad_control_cb, 1)
        self.create_subscription(Float32, "/safety/h_backup", self._h_backup_cb, 1)

        self.control_pub = self.create_publisher(ServoMsg, self.params.filtered_control_topic, 1)
        self.override_pub = self.create_publisher(Bool, "/safety/override", 1)
        self.u_human_pub = self.create_publisher(Float64MultiArray, "/safety/u_human", 1)
        self.u_filtered_pub = self.create_publisher(Float64MultiArray, "/safety/u_filtered", 1)

    # ---- Callbacks ----

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9

    def _human_cb(self, msg: ServoMsg):
        self.last_human_msg = msg
        self.last_human_time = self.get_clock().now().nanoseconds * 1e-9

    def _value_cb(self, msg: Float32):
        self.last_h = float(msg.data)
        self.last_safety_time = self.get_clock().now().nanoseconds * 1e-9

    def _grad_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 5:
            self.last_grad = np.array(msg.data, dtype=float)

    def _u0_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.last_backup_u0 = np.array(msg.data, dtype=float)

    def _grad_control_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.last_grad_control = np.array(msg.data, dtype=float)

    def _h_backup_cb(self, msg: Float32):
        self.last_h_backup = float(msg.data)

    # ---- Main loop ----

    def _step(self):
        if self.last_state is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        state = self.last_state.copy()
        p = self.params

        # Stale odom or human input → brake gently (truck might have dropped connection).
        odom_stale = self.last_odom_time is None or now - self.last_odom_time > p.stale_timeout_s
        human_stale = self.last_human_time is None or now - self.last_human_time > p.stale_timeout_s
        if odom_stale or human_stale:
            self._pub_control(np.array([p.a_min * 0.5, 0.0]), state)
            self._pub_override(True)
            return

        u_human = servo_msg_to_control(self.last_human_msg, state, p)

        # Re-read enable_qp each cycle so it can be toggled via ros2 param set.
        self.enable_qp = self.get_parameter("enable_qp").value
        if not self.enable_qp:
            self._pub_control(u_human, state)
            self._pub_override(False)
            self._pub_debug(u_human, u_human)
            return

        # Safety data not yet available → pass through until monitor is ready.
        safety_stale = (
            self.last_h is None
            or self.last_grad is None
            or self.last_safety_time is None
            or now - self.last_safety_time > p.stale_timeout_s
        )
        if safety_stale:
            self._pub_control(u_human, state)
            self._pub_override(False)
            self._pub_debug(u_human, u_human)
            return

        try:
            u_out, override = self._cbf_step(state, u_human)
        except Exception:
            self.get_logger().error(f"QP fault:\n{traceback.format_exc()}")
            u_out = self.last_backup_u0 if self.last_backup_u0 is not None else np.zeros(2)
            override = True

        self._pub_control(u_out, state)
        self._pub_override(override)
        self._pub_debug(u_human, u_out)

    def _cbf_step(self, state: np.ndarray, u_human: np.ndarray):
        """Run the CBF-QP and return (u_filtered, override_bool)."""
        h = self.last_h
        grad = self.last_grad
        p = self.params

        # Prefer the control gradient published by the monitor: it captures the
        # full H-step effect of [a, omega] on h_imp via finite differences
        # through the rollout.  B^T ∇h only sees the 1-step algebraic coupling
        # (B[2,0]=dt, B[4,1]=dt) which is near-zero for lane-dominated barriers
        # because the lane gradient lives in px/py/psi, not v/δ.
        if (self.last_grad_control is not None
                and self.last_h_backup is not None
                and self.last_backup_u0 is not None):
            g = self.last_grad_control
            h_backup0 = self.last_h_backup
            u0 = self.last_backup_u0
            # Linearization of h_imp(f(x,u)) around u0:
            #   h_imp(f(x,u)) ≈ h_backup0 + g^T (u - u0) >= (1-λ)*h
            c = (1.0 - p.lambda_cbf) * h - h_backup0 + float(g @ u0)
        else:
            # Fallback when control gradient not yet available.
            f_human = step(state, u_human, p)
            B = control_jacobian(p)
            g = B.T @ grad
            c = (
                -p.lambda_cbf * h
                - float(grad @ (f_human - state))
                + float(g @ u_human)
            )

        result = qp_solve(u_human, g, c, p)
        u_out = result.control

        if result.status == "infeasible":
            # Box and CBF half-plane are disjoint — back on emergency brake.
            if self.last_backup_u0 is not None:
                u_out = self.last_backup_u0.copy()
            self.get_logger().warn("QP infeasible — using backup u0 fallback")

        # Hysteresis: once we start overriding, keep it up for a few more
        # cycles even after the human input becomes safe again.  Without this,
        # the filter can flicker rapidly on/off at the safe-set boundary.
        deviates = np.linalg.norm(u_out - u_human, ord=np.inf) > p.passthrough_tolerance
        if deviates:
            self.override_hold = p.hysteresis_cycles
        elif self.override_hold > 0:
            self.override_hold -= 1

        return u_out, deviates or self.override_hold > 0

    # ---- Publishers ----

    def _pub_control(self, u: np.ndarray, state: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        msg = control_to_servo_msg(ServoMsg, u, state, self.params, stamp)
        self.delta_estimate = float(msg.steer)
        self.control_pub.publish(msg)

    def _pub_override(self, override: bool):
        msg = Bool()
        msg.data = bool(override)
        self.override_pub.publish(msg)

    def _pub_debug(self, u_human: np.ndarray, u_filtered: np.ndarray):
        uh = Float64MultiArray()
        uh.data = [float(u_human[0]), float(u_human[1])]
        self.u_human_pub.publish(uh)

        uf = Float64MultiArray()
        uf.data = [float(u_filtered[0]), float(u_filtered[1])]
        self.u_filtered_pub.publish(uf)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterQPNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
