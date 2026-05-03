#!/usr/bin/env python3
"""
ILQR Safety Filter Node — Fallback / Monitor / Arbiter in one node.

Background thread:  _planner_loop() at planner_rate_hz (10 Hz)
  → runs ILQRSolver from current state, stores (X*, U*) under _plan_lock

Foreground timer:   control_step() at control_rate_hz (20 Hz)
  → SafetyMonitor.certify(x, u_h, ctx, U*) → pass u_h or u*[0]
"""
import threading
import time
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from racecar_msgs.msg import OdometryArray, ServoMsg

from ece346.Final_Project.safety_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.Final_Project.safety_filter.obstacle_memory import ObstacleMemory
from ece346.Final_Project.safety_filter.ros_utils import (
    control_to_servo_msg,
    marker_array_to_obstacles,
    odom_to_state,
    odometry_array_to_obstacles,
    servo_msg_to_control,
)

from ece346.ilqr_safety_filter.safety_filter.config import declare_and_load
from ece346.ilqr_safety_filter.safety_filter.ilqr_solver import ILQRSolver
from ece346.ilqr_safety_filter.safety_filter.jax_context import build_context, dummy_context
from ece346.ilqr_safety_filter.safety_filter.monitor import SafetyMonitor


class ILQRSafetyFilterNode(Node):
    def __init__(self):
        super().__init__("ilqr_safety_filter_node")
        self.params = declare_and_load(self)

        if not self.params.map_file:
            try:
                self.params.map_file = (
                    get_package_share_directory("racecar_routing") + "/maps/track.osm"
                )
            except Exception:
                pass

        # State
        self.delta_estimate = 0.0
        self.last_state: np.ndarray = None
        self.last_odom_time: float = None
        self.last_human_msg = None
        self.last_human_time: float = None
        self.last_lane_center: np.ndarray = None
        self.last_lane_yaw: float = None

        self.lane_context = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file,
            self,
            self.params.lane_change_cost,
        )

        self.static_memory = ObstacleMemory(
            self.params.obstacle_memory_ttl_s,
            self.params.obstacle_memory_growth,
            self.params.obstacle_radius_default,
        )
        self.traffic = []

        # ILQR planner + monitor
        self.solver = ILQRSolver(self.params)
        self.monitor = SafetyMonitor(self.params)

        # Shared plan (protected by lock)
        self._plan_lock = threading.Lock()
        self._current_X: np.ndarray = None
        self._current_U: np.ndarray = None

        self.override_hold: int = 0

        self._setup_io()

        # JAX JIT warmup before first control tick
        self.get_logger().info("ILQR safety filter: running JAX warmup...")
        self._jax_warmup()
        self.get_logger().info("ILQR warmup complete.")

        # Start background planner thread
        self._planner_running = True
        self._planner_thread = threading.Thread(
            target=self._planner_loop, daemon=True
        )
        self._planner_thread.start()

        self.timer = self.create_timer(
            1.0 / self.params.control_rate_hz, self.control_step
        )
        self.get_logger().info(
            f"ilqr_safety_filter_node ready: "
            f"odom={self.params.odom_topic}, "
            f"human={self.params.human_control_topic}, "
            f"out={self.params.filtered_control_topic}"
        )

    def _jax_warmup(self):
        dummy_state = np.zeros(5, dtype=np.float64)
        dummy_state[2] = 0.5   # non-zero speed avoids degenerate Jacobian
        ctx = dummy_context(self.params)
        self.solver.warmup(dummy_state, ctx)
        self.monitor.solver.warmup(dummy_state, ctx)

    # ------------------------------------------------------------------
    # I/O setup
    # ------------------------------------------------------------------

    def _setup_io(self):
        self.odom_sub = self.create_subscription(
            Odometry, self.params.odom_topic, self.odom_callback, 10
        )
        self.human_sub = self.create_subscription(
            ServoMsg,
            self.params.human_control_topic,
            self.human_callback,
            10,
        )
        self.static_sub = self.create_subscription(
            MarkerArray,
            self.params.static_obstacles_topic,
            self.static_obstacles_callback,
            10,
        )
        self.dynamic_sub = self.create_subscription(
            OdometryArray,
            self.params.dynamic_obstacles_topic,
            self.dynamic_obstacles_callback,
            10,
        )

        self.command_pub = self.create_publisher(
            ServoMsg, self.params.filtered_control_topic, 1
        )
        self.vhat_pub = self.create_publisher(Float32, "/safety/V_hat", 1)
        self.override_pub = self.create_publisher(Bool, "/safety/override", 1)
        self.binding_pub = self.create_publisher(String, "/safety/binding_constraint", 1)
        self.u_human_pub = self.create_publisher(Float64MultiArray, "/safety/u_human", 1)
        self.u_filtered_pub = self.create_publisher(Float64MultiArray, "/safety/u_filtered", 1)
        self.plan_states_pub = self.create_publisher(Float64MultiArray, "/safety/plan_states", 1)
        self.plan_margins_pub = self.create_publisher(Float64MultiArray, "/safety/plan_margins", 1)
        self.plan_path_pub = self.create_publisher(Path, "/safety/plan_path", 1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9
        self._maybe_rebuild_lane_context(self.last_state)

    def human_callback(self, msg):
        self.last_human_msg = msg
        self.last_human_time = self.get_clock().now().nanoseconds * 1e-9

    def static_obstacles_callback(self, msg: MarkerArray):
        t_now = self.get_clock().now().nanoseconds * 1e-9
        self.static_memory.update(
            t_now,
            marker_array_to_obstacles(msg, self.params.obstacle_radius_default),
        )

    def dynamic_obstacles_callback(self, msg: OdometryArray):
        self.traffic = odometry_array_to_obstacles(msg, self.params.truck_radius_m)

    # ------------------------------------------------------------------
    # Lane context
    # ------------------------------------------------------------------

    def _maybe_rebuild_lane_context(self, state: np.ndarray):
        p = state[:2]
        if self.last_lane_center is not None:
            dist = np.linalg.norm(p - self.last_lane_center)
            yaw_d = abs((state[3] - self.last_lane_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            if (
                dist < self.params.lane_context_rebuild_distance_m
                and yaw_d < self.params.lane_context_rebuild_yaw_rad
            ):
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = p.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as exc:
            self.get_logger().warn(
                f"lane context rebuild failed, using previous/fallback: {exc}"
            )

    # ------------------------------------------------------------------
    # Background planner thread
    # ------------------------------------------------------------------

    def _planner_loop(self):
        while self._planner_running and rclpy.ok():
            t0 = time.monotonic()
            try:
                if self.last_state is not None:
                    state = self.last_state.copy()
                    ctx = self._build_context(state)

                    with self._plan_lock:
                        prev_U = self._current_U

                    if prev_U is not None:
                        U_init = np.vstack([prev_U[1:], prev_U[-1:]])
                    else:
                        U_init = None

                    X, U, info = self.solver.solve(state, ctx, U_init=U_init)

                    with self._plan_lock:
                        self._current_X = X
                        self._current_U = U

            except Exception:
                self.get_logger().warn(
                    f"planner fault:\n{traceback.format_exc()}"
                )

            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, 1.0 / self.params.planner_rate_hz - elapsed)
            time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Foreground control step (monitor + arbiter)
    # ------------------------------------------------------------------

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_state is None:
            return

        state = self.last_state.copy()

        odom_stale = (
            self.last_odom_time is None
            or now - self.last_odom_time > self.params.stale_timeout_s
        )
        if odom_stale:
            self._safe_fallback(state)
            return

        with self._plan_lock:
            X_star = self._current_X
            U_star = self._current_U

        if X_star is None or U_star is None:
            self._safe_fallback(state)
            return

        u_h = self._human_control(state)

        try:
            ctx = self._build_context(state)
            is_safe, V_hat, binding = self.monitor.certify(state, u_h, ctx, U_star)

            if is_safe and self.override_hold == 0:
                u_out = u_h
                override = False
            else:
                u_out = U_star[0].copy()
                override = True
                if is_safe is False:
                    self.override_hold = self.params.hysteresis_cycles

            if self.override_hold > 0:
                self.override_hold -= 1

            u_out = np.clip(
                u_out,
                [self.params.a_min, self.params.omega_min],
                [self.params.a_max, self.params.omega_max],
            )

            self._publish_command(u_out, state)
            self._publish_debug(V_hat, binding, u_h, u_out, override, X_star, ctx)

        except Exception:
            self.get_logger().error(
                f"control_step fault:\n{traceback.format_exc()}"
            )
            self._safe_fallback(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, state: np.ndarray):
        now = self.get_clock().now().nanoseconds * 1e-9
        obstacles = self.static_memory.get(now)
        v_ref = float(np.clip(state[2], 0.0, self.params.v_max))
        return build_context(
            obstacles, self.traffic, self.lane_context, v_ref, self.params
        )

    def _human_control(self, state: np.ndarray) -> np.ndarray:
        if self.last_human_msg is None:
            return np.zeros(2, dtype=float)
        return servo_msg_to_control(self.last_human_msg, state, self.params)

    def _publish_command(self, u: np.ndarray, state: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        msg = control_to_servo_msg(ServoMsg, u, state, self.params, stamp)
        self.delta_estimate = float(msg.steer)
        self.command_pub.publish(msg)

    def _safe_fallback(self, state: np.ndarray):
        u = np.array([self.params.a_min * 0.5, 0.0])
        self._publish_command(u, state)

    def _publish_debug(
        self,
        V_hat: float,
        binding: str,
        u_h: np.ndarray,
        u_out: np.ndarray,
        override: bool,
        X_star: np.ndarray,
        ctx,
    ):
        v_msg = Float32()
        v_msg.data = float(V_hat)
        self.vhat_pub.publish(v_msg)

        o_msg = Bool()
        o_msg.data = bool(override)
        self.override_pub.publish(o_msg)

        b_msg = String()
        b_msg.data = str(binding)
        self.binding_pub.publish(b_msg)

        uh_msg = Float64MultiArray()
        uh_msg.data = [float(v) for v in u_h]
        self.u_human_pub.publish(uh_msg)

        uf_msg = Float64MultiArray()
        uf_msg.data = [float(v) for v in u_out]
        self.u_filtered_pub.publish(uf_msg)

        ps_msg = Float64MultiArray()
        ps_msg.data = [float(v) for v in X_star.flatten()]
        self.plan_states_pub.publish(ps_msg)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for x in X_star:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x[0])
            pose.pose.position.y = float(x[1])
            pose.pose.orientation.z = float(np.sin(x[3] / 2.0))
            pose.pose.orientation.w = float(np.cos(x[3] / 2.0))
            path_msg.poses.append(pose)
        self.plan_path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ILQRSafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node._planner_running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
