#!/usr/bin/env python3
"""
safety_monitor_node — heuristic safety monitor.

Computes a scalar safety value h at the current truck state and over a
forward lookahead.  Two rollouts are evaluated and the minimum is published:

  h_backup  — rollout under the backup policy (max brake + steer to center).
              This is the theoretically correct backup-CBF value: it answers
              "if we switch to the backup policy right now, do we stay safe?"
  h_human   — rollout under the human's current command, kept to avoid
              false triggers at bends where h_backup can be pessimistic.

h = min(h_backup, h_human) is published so both conditions must hold.

  h >= 0  →  safe
  h <  0  →  violated a safety margin

Published topics:
  /safety/value              Float32           — h
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
from ece346.cbf_filter.cbf_filter.dynamics import step, wrap_angle
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
        self.last_backup_u0: np.ndarray = None
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
        self.create_subscription(
            Float64MultiArray, "/safety/backup_u0", self._backup_u0_cb, 1
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

    def _backup_u0_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.last_backup_u0 = np.array(msg.data, dtype=float)

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
            current_min = min(m_lane, m_obs, m_traf)

            # Dynamic horizon: covers stopping distance but capped at horizon_H_max.
            # Uncapped growth causes h_backup to shrink monotonically with speed —
            # a longer trajectory has more steps where margin can be small,
            # keeping h permanently below throttle_cap_margin and trapping the truck.
            v_now = float(state[2])
            H_stop = int(np.ceil(v_now / (abs(p.a_min) * p.dt))) + 1
            H = max(p.horizon_H, min(H_stop, p.horizon_H_max))

            # Backup-policy rollout: the theoretically correct h_imp value.
            # "If we switch to the backup policy right now, do we stay safe?"
            # The policy is recomputed at every step so the trajectory correctly
            # tracks lane curves — a frozen omega from the start would cut across
            # bends and give a systematically pessimistic h_backup.
            x = state.copy()
            lookahead_backup = current_min
            for _ in range(H):
                x = step(x, self._backup_policy(x), p)
                lk = min(
                    margin_lane(x, lane, p),
                    margin_obstacle(x, obstacles, p),
                    margin_obstacle(x, self.traffic, p, p.r_safe_traf),
                )
                lookahead_backup = min(lookahead_backup, lk)

            # Human-command rollout: kept to avoid false triggers at bends.
            # If the human is steering into a bend, the backup rollout (which
            # steers to center) can be pessimistic — the human's path is safer.
            if self.last_human_msg is not None:
                u_human = servo_msg_to_control(self.last_human_msg, state, p)
            else:
                u_human = np.array([0.0, 0.0])
            x = state.copy()
            lookahead_human = current_min
            for _ in range(H):
                x = step(x, u_human, p)
                lk = min(
                    margin_lane(x, lane, p),
                    margin_obstacle(x, obstacles, p),
                    margin_obstacle(x, self.traffic, p, p.r_safe_traf),
                )
                lookahead_human = min(lookahead_human, lk)

            # Both conditions must hold: backup feasibility AND human-path safety.
            lookahead_min = min(lookahead_backup, lookahead_human)
            h = min(current_min, lookahead_min)

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

    def _backup_policy(self, x: np.ndarray) -> np.ndarray:
        """Compute the backup policy command at state x, querying lane at x.

        Mirrors cbf_backup_planner_node logic exactly: max brake + proportional
        steer toward lane center. Recomputing at each rollout step means the
        backup trajectory correctly follows lane curves instead of using a
        stale omega from the rollout starting point.
        """
        p = self.params
        _, _, v, _, delta = x
        sample = self.lane_context.query(float(x[0]), float(x[1]))
        heading_err = wrap_angle(sample.tangent - float(x[3]))
        delta_des = float(np.clip(
            heading_err - np.arctan2(p.K_e * sample.signed_lateral_error, abs(v) + p.v_eps),
            p.delta_min, p.delta_max,
        ))
        omega = float(np.clip(p.K_p * (delta_des - delta), p.omega_min, p.omega_max))
        return np.array([p.a_min, omega])

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
