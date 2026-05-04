#!/usr/bin/env python3
"""
backup_planner_node — heuristic fallback control.

Publishes a safe [a, omega] every cycle:
  - Always brakes at maximum deceleration (a = a_min).
  - Steers back toward the lane centerline proportionally to lateral error.

Published topics:
  /safety/backup_u0   Float64MultiArray   [a, omega]
"""
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.dynamics import wrap_angle
from ece346.cbf_filter.cbf_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.cbf_filter.cbf_filter.ros_utils import odom_to_state


class BackupPlannerNode(Node):
    def __init__(self):
        super().__init__("backup_planner_node")
        self.params: CbfParams = declare_and_load(self)

        if not self.params.map_file:
            try:
                self.params.map_file = (
                    get_package_share_directory("racecar_routing") + "/maps/track.osm"
                )
            except Exception:
                pass

        self.delta_estimate = 0.0
        self.last_state: np.ndarray = None
        self.last_lane_center: np.ndarray = None
        self.last_lane_yaw: float = None

        self.lane_context: LaneContext = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file, self, self.params.lane_change_cost
        )

        self.create_subscription(Odometry, self.params.odom_topic, self._odom_cb, 10)
        self.u0_pub = self.create_publisher(Float64MultiArray, "/safety/backup_u0", 1)
        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info("backup_planner_node ready")

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self._maybe_rebuild_lane(self.last_state)

    def _step(self):
        if self.last_state is None:
            return
        try:
            state = self.last_state.copy()
            p = self.params
            _, _, v, _, delta = state

            # Steer toward lane centerline.
            sample = self.lane_context.query(float(state[0]), float(state[1]))
            heading_err = wrap_angle(sample.tangent - float(state[3]))
            delta_des = np.clip(
                heading_err
                - np.arctan2(p.K_e * sample.signed_lateral_error, abs(v) + p.v_eps),
                p.delta_min,
                p.delta_max,
            )
            omega = float(np.clip(p.K_p * (delta_des - delta), p.omega_min, p.omega_max))

            msg = Float64MultiArray()
            msg.data = [float(p.a_min), omega]
            self.u0_pub.publish(msg)
        except Exception:
            self.get_logger().error(f"backup_planner fault:\n{traceback.format_exc()}")

    def _maybe_rebuild_lane(self, state: np.ndarray):
        p = state[:2]
        if self.last_lane_center is not None:
            dist_ok = (
                np.linalg.norm(p - self.last_lane_center)
                < self.params.lane_context_rebuild_distance_m
            )
            yaw_ok = (
                abs((state[3] - self.last_lane_yaw + np.pi) % (2 * np.pi) - np.pi)
                < self.params.lane_context_rebuild_yaw_rad
            )
            if dist_ok and yaw_ok:
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = p.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as e:
            self.get_logger().warn(f"lane rebuild failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BackupPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
