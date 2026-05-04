#!/usr/bin/env python3
"""
safety_monitor_node — simple heuristic safety monitor.

Computes a scalar safety value h at the current truck state and over a
10-step lookahead using the human's current intended command (not coast):

  h >= 0  →  truck is safe (positive clearance from lane + obstacles)
  h <  0  →  truck has violated a safety margin

Using the human command for the lookahead means the filter does NOT fire at
bends just because the truck hasn't turned yet — it fires only if the human's
own intended trajectory would be unsafe.

Published topics:
  /safety/value              Float32           — h (min margin over current + lookahead)
  /safety/debug_margins      Float64MultiArray — [lane, obstacle, traffic, lookahead_min]
  /safety/binding_constraint String            — which margin is binding
"""
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from racecar_msgs.msg import OdometryArray, ServoMsg

from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.dynamics import step
from ece346.cbf_filter.cbf_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.cbf_filter.cbf_filter.margins import margin_lane, margin_obstacle
from ece346.cbf_filter.cbf_filter.obstacle_memory import ObstacleMemory
from ece346.cbf_filter.cbf_filter.ros_utils import (
    marker_array_to_obstacles,
    odometry_array_to_obstacles,
    odom_to_state,
    servo_msg_to_control,
)


class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__("safety_monitor_node")
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
        self.last_human_msg = None
        self.traffic: list = []
        self.last_lane_center: np.ndarray = None
        self.last_lane_yaw: float = None

        self.lane_context: LaneContext = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file,
            self,
            self.params.lane_change_cost,
            self.params.lane_allow_lane_change,
            self.params.route_hysteresis_rad,
        )
        self.static_memory = ObstacleMemory(
            self.params.obstacle_memory_ttl_s,
            self.params.obstacle_memory_growth,
            self.params.obstacle_radius_default,
        )

        self.create_subscription(Odometry, self.params.odom_topic, self._odom_cb, 10)
        self.create_subscription(
            ServoMsg, self.params.human_control_topic, self._human_cb, 10
        )
        self.create_subscription(
            MarkerArray, self.params.static_obstacles_topic, self._static_cb, 10
        )
        self.create_subscription(
            OdometryArray, self.params.dynamic_obstacles_topic, self._dynamic_cb, 10
        )
        self.create_subscription(
            ServoMsg, self.params.filtered_control_topic, self._control_cb, 10
        )

        self.value_pub = self.create_publisher(Float32, "/safety/value", 1)
        self.debug_pub = self.create_publisher(Float64MultiArray, "/safety/debug_margins", 1)
        self.binding_pub = self.create_publisher(String, "/safety/binding_constraint", 1)

        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info("safety_monitor_node ready")

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self._maybe_rebuild_lane(self.last_state)

    def _human_cb(self, msg: ServoMsg):
        self.last_human_msg = msg

    def _control_cb(self, msg: ServoMsg):
        p = self.params
        self.delta_estimate = float(np.clip(msg.steer, p.delta_min, p.delta_max))
        if self.last_state is not None:
            self.last_state[4] = self.delta_estimate

    def _static_cb(self, msg: MarkerArray):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.static_memory.update(
            t, marker_array_to_obstacles(msg, self.params.obstacle_radius_default)
        )

    def _dynamic_cb(self, msg: OdometryArray):
        self.traffic = odometry_array_to_obstacles(msg, self.params.truck_radius_m)

    def _step(self):
        if self.last_state is None:
            return
        try:
            state = self.last_state.copy()
            p = self.params
            now = self.get_clock().now().nanoseconds * 1e-9
            obstacles = self.static_memory.get(now)
            lane = self.lane_context

            if lane.is_fallback:
                h = float(p.fallback_h_safe)
                binding = "fallback_map"
                v_msg = Float32()
                v_msg.data = h
                self.value_pub.publish(v_msg)
                d_msg = Float64MultiArray()
                d_msg.data = [h, 100.0, 100.0, h]
                self.debug_pub.publish(d_msg)
                b_msg = String()
                b_msg.data = binding
                self.binding_pub.publish(b_msg)
                return

            # Current-state margins.
            m_lane = margin_lane(state, lane, p)
            m_obs = margin_obstacle(state, obstacles, p)
            m_traf = margin_obstacle(state, self.traffic, p, p.r_safe_traf)

            # 10-step lookahead using the human's current command.
            # This prevents false triggers at bends: if the human is turning
            # into the bend, the projected path follows the bend safely.
            # Fall back to coast [0, 0] only if no human command is available.
            if self.last_human_msg is not None:
                u_lookahead = servo_msg_to_control(self.last_human_msg, state, p)
            else:
                u_lookahead = np.array([0.0, 0.0])

            x = state.copy()
            lookahead_min = min(m_lane, m_obs, m_traf)
            for _ in range(p.horizon_H):
                x = step(x, u_lookahead, p)
                lk = min(
                    margin_lane(x, lane, p),
                    margin_obstacle(x, obstacles, p),
                    margin_obstacle(x, self.traffic, p, p.r_safe_traf),
                )
                lookahead_min = min(lookahead_min, lk)

            h = min(m_lane, m_obs, m_traf, lookahead_min)

            components = {
                "lane": m_lane,
                "obstacle": m_obs,
                "traffic": m_traf,
                "lookahead": lookahead_min,
            }
            binding = min(components, key=components.get)

            # Publish.
            v_msg = Float32()
            v_msg.data = float(h)
            self.value_pub.publish(v_msg)

            d_msg = Float64MultiArray()
            d_msg.data = [float(m_lane), float(m_obs), float(m_traf), float(lookahead_min)]
            self.debug_pub.publish(d_msg)

            b_msg = String()
            b_msg.data = binding
            self.binding_pub.publish(b_msg)

        except Exception:
            self.get_logger().error(f"safety_monitor fault:\n{traceback.format_exc()}")

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
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
