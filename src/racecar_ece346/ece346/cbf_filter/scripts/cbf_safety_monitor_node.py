#!/usr/bin/env python3
"""
safety_monitor_node — computes the implicit barrier value h_imp(x) and its
gradient ∇h_imp(x).

Role in the 3-node CBF filter: MONITOR
  "Is it safe for the truck to commit to the backup policy right now?"

  h_imp(x) >= 0  →  YES: applying the backup policy for the next H steps
                        keeps the truck out of every failure set.
  h_imp(x) < 0  →  NO:  the truck is already in a situation where the backup
                        policy cannot guarantee safety (e.g., obstacle too close).

How it works
  1. Receives the backup trajectory from backup_planner_node.
  2. Evaluates margin_total (lane + obstacle + traffic + kinematic) at each
     of the H+1 states in the trajectory.
  3. h_imp = min of all those margins.
  4. ∇h_imp computed via finite differences: perturb each of the 5 state
     dimensions by grad_eps, re-run the H-step rollout, measure the change.
     (6 rollouts total per control cycle — fast enough at 20 Hz in numpy.)
  5. Publishes h_imp, ∇h_imp, and which constraint is binding.

Published topics:
  /safety/value              Float32            — h_imp value
  /safety/grad               Float64MultiArray  — ∇h_imp, length 5
  /safety/binding_constraint String             — "lane" / "obstacle" / ...
  /safety/margins_along_traj Float64MultiArray  — per-step total margin (for plots)
"""
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from racecar_msgs.msg import OdometryArray

from ece346.cbf_filter.cbf_filter.backup_policy import brake_and_recenter
from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.dynamics import rollout
from ece346.cbf_filter.cbf_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.cbf_filter.cbf_filter.margins import MarginContext, margin_total
from ece346.cbf_filter.cbf_filter.obstacle_memory import ObstacleMemory
from ece346.cbf_filter.cbf_filter.ros_utils import (
    marker_array_to_obstacles,
    odometry_array_to_obstacles,
    odom_to_state,
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
        self.last_traj_data: list = None
        self.last_traj_time: float = None
        self.traffic: list = []

        self.lane_context: LaneContext = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file, self, self.params.lane_change_cost
        )
        self.last_lane_center: np.ndarray = None
        self.last_lane_yaw: float = None

        self.static_memory = ObstacleMemory(
            self.params.obstacle_memory_ttl_s,
            self.params.obstacle_memory_growth,
            self.params.obstacle_radius_default,
        )

        self._setup_io()
        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info("safety_monitor_node ready")

    def _setup_io(self):
        self.create_subscription(Odometry, self.params.odom_topic, self._odom_cb, 10)
        self.create_subscription(Float64MultiArray, "/safety/backup_traj", self._traj_cb, 1)
        self.create_subscription(MarkerArray, self.params.static_obstacles_topic, self._static_cb, 10)
        self.create_subscription(OdometryArray, self.params.dynamic_obstacles_topic, self._dynamic_cb, 10)

        self.value_pub = self.create_publisher(Float32, "/safety/value", 1)
        self.grad_pub = self.create_publisher(Float64MultiArray, "/safety/grad", 1)
        self.binding_pub = self.create_publisher(String, "/safety/binding_constraint", 1)
        self.margins_pub = self.create_publisher(Float64MultiArray, "/safety/margins_along_traj", 1)

    # ---- Callbacks ----

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self._maybe_rebuild_lane(self.last_state)

    def _traj_cb(self, msg: Float64MultiArray):
        self.last_traj_data = list(msg.data)
        self.last_traj_time = self.get_clock().now().nanoseconds * 1e-9

    def _static_cb(self, msg: MarkerArray):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.static_memory.update(t, marker_array_to_obstacles(msg, self.params.obstacle_radius_default))

    def _dynamic_cb(self, msg: OdometryArray):
        self.traffic = odometry_array_to_obstacles(msg, self.params.truck_radius_m)

    # ---- Main loop ----

    def _step(self):
        if self.last_state is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        state = self.last_state.copy()

        # Wait until backup_planner has published a trajectory.
        if (self.last_traj_data is None
                or self.last_traj_time is None
                or now - self.last_traj_time > self.params.stale_timeout_s):
            return

        try:
            traj = np.array(self.last_traj_data, dtype=float).reshape(
                self.params.horizon_H + 1, 5
            )
        except Exception:
            self.get_logger().warn("Malformed backup_traj message — wrong number of floats?")
            return

        try:
            obstacles = self.static_memory.get(now)
            ctx = MarginContext(self.lane_context, obstacles, self.traffic, self.params)

            # Evaluate margin at every state in the backup trajectory.
            margins = []
            labels = []
            for s in traj:
                val, lbl = margin_total(s, ctx)
                margins.append(val)
                labels.append(lbl)

            h_imp = float(min(margins))
            binding = labels[int(np.argmin(margins))]

            # Gradient via finite differences.
            # We perturb the CURRENT state (not the trajectory), re-run the
            # H-step rollout from the perturbed state, and measure Δh_imp.
            grad = self._gradient(state, h_imp, ctx)

            self._publish(h_imp, grad, binding, margins)

        except Exception:
            self.get_logger().error(f"safety_monitor fault:\n{traceback.format_exc()}")

    def _gradient(self, state: np.ndarray, h0: float, ctx: MarginContext) -> np.ndarray:
        """Finite-difference ∇h_imp — 5 extra rollouts."""
        grad = np.zeros(5, dtype=float)
        lane = self.lane_context
        p = self.params
        for i in range(5):
            x_pert = state.copy()
            x_pert[i] += p.grad_eps
            traj_pert = rollout(x_pert, lambda x: brake_and_recenter(x, lane, p), p)
            h_pert = min(margin_total(s, ctx)[0] for s in traj_pert)
            grad[i] = (h_pert - h0) / p.grad_eps
        return grad

    # ---- Lane context ----

    def _maybe_rebuild_lane(self, state: np.ndarray):
        p = state[:2]
        if self.last_lane_center is not None:
            if (np.linalg.norm(p - self.last_lane_center) < self.params.lane_context_rebuild_distance_m
                    and abs((state[3] - self.last_lane_yaw + np.pi) % (2 * np.pi) - np.pi)
                    < self.params.lane_context_rebuild_yaw_rad):
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = p.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as e:
            self.get_logger().warn(f"lane rebuild failed: {e}")

    # ---- Publishers ----

    def _publish(self, h_imp: float, grad: np.ndarray, binding: str, margins: list):
        v = Float32()
        v.data = h_imp
        self.value_pub.publish(v)

        g = Float64MultiArray()
        g.data = [float(x) for x in grad]
        self.grad_pub.publish(g)

        b = String()
        b.data = binding
        self.binding_pub.publish(b)

        m = Float64MultiArray()
        m.data = [float(x) for x in margins]
        self.margins_pub.publish(m)


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
