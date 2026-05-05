"""ROS2 heuristic safety filter that reuses the FinalProject CBF margins.

Heuristics handled here:
- stale_input_brake_recenter: brake and recenter when odom/teleop is stale.
- immediate_forward_obstacle_brake: stop for close obstacles straight ahead.
- dynamic_traffic_brake: stop for dynamic obstacles inside the traffic margin.
- lane_boundary_recenter: slow and steer back when near/outside lane bounds.
- predicted_lane_departure: override steering if human rollout leaves the lane.
- corner_pre_turn: pre-steer and cap speed before sharp route heading changes.
- obstacle_approach_slowdown: brake progressively before emergency distance.
- side_obstacle_lane_bias: steer away from nearby side obstacles when lane room exists.
- kinematic_limit_recovery: avoid commands that push speed/steering past limits.
- candidate_margin_check: choose the candidate with the best next CBF margin.
"""

from dataclasses import dataclass
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from racecar_msgs.msg import OdometryArray
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from .backup_policy import (
    brake_and_recenter,
    emergency_brake,
    lane_recovery_control,
    recenter_control,
)
from .barrier import evaluate_barrier, implicit_barrier_value
from .config import declare_and_load
from .dynamics import clip_control, rollout, step, wrap_angle
from .lane_context import LaneContext, LaneletContextBuilder, heading_error_to_lane
from .margins import MarginContext, margin_components, margin_lane
from .ros_utils import (
    ackermann_msg_to_control,
    control_to_ackermann_msg,
    marker_array_to_obstacles,
    odom_to_state,
    odometry_array_to_obstacles,
)


@dataclass
class HeuristicDecision:
    control: np.ndarray
    status: str


class CbfHeuristicSafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_node")
        self.params = declare_and_load(self)
        if not self.params.map_file:
            self.params.map_file = (
                get_package_share_directory("racecar_routing") + "/maps/track.osm"
            )

        self.delta_estimate = 0.0
        self.last_state = None
        self.last_odom_time = None
        self.last_human_msg = None
        self.last_human_time = None
        self.last_lane_center = None
        self.last_lane_yaw = None
        self.lane_context = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file,
            self,
            self.params.lane_change_cost,
            self.params.lane_allow_lane_change,
            self.params.route_hysteresis_rad,
        )
        self.static_obstacles = []
        self.traffic = []
        self.override_hold = 0

        self._setup_io()
        self.timer = self.create_timer(
            1.0 / max(self.params.control_rate_hz, 1e-3),
            self.control_step,
        )
        self.get_logger().info(
            "cbf-heuristic safety_filter_node ready: "
            f"odom={self.params.odom_topic}, human={self.params.human_control_topic}, "
            f"map={self.params.map_file}, out={self.params.filtered_control_topic}"
        )

    def _setup_io(self):
        self.odom_sub = self.create_subscription(
            Odometry, self.params.odom_topic, self.odom_callback, 10
        )
        self.human_sub = self.create_subscription(
            AckermannDriveStamped,
            self.params.human_control_topic,
            self.human_callback,
            10,
        )
        self.command_pub = self.create_publisher(
            AckermannDriveStamped,
            self.params.filtered_control_topic,
            1,
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
        self.value_pub = self.create_publisher(Float32, "/safety/value", 1)
        self.grad_pub = self.create_publisher(Float64MultiArray, "/safety/grad", 1)
        self.override_pub = self.create_publisher(Bool, "/safety/override", 1)
        self.binding_pub = self.create_publisher(String, "/safety/binding_constraint", 1)
        self.status_pub = self.create_publisher(String, "/safety/status", 1)
        self.margins_pub = self.create_publisher(Float64MultiArray, "/safety/margins", 1)
        self.u_human_pub = self.create_publisher(Float64MultiArray, "/safety/u_human", 1)
        self.u_filtered_pub = self.create_publisher(Float64MultiArray, "/safety/u_filtered", 1)
        self.backup_path_pub = self.create_publisher(Path, "/safety/backup_traj", 1)

    def odom_callback(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9
        self._maybe_rebuild_lane_context(self.last_state)

    def human_callback(self, msg: AckermannDriveStamped):
        self.last_human_msg = msg
        self.last_human_time = self.get_clock().now().nanoseconds * 1e-9

    def static_obstacles_callback(self, msg: MarkerArray):
        self.static_obstacles = marker_array_to_obstacles(
            msg,
            self.params.obstacle_radius_default,
        )

    def dynamic_obstacles_callback(self, msg: OdometryArray):
        self.traffic = odometry_array_to_obstacles(msg, self.params.truck_radius_m)

    def _human_control(self, state: np.ndarray) -> np.ndarray:
        if self.last_human_msg is None:
            return np.array([0.0, 0.0], dtype=float)
        return ackermann_msg_to_control(self.last_human_msg, state, self.params)

    def _publish_command(self, u: np.ndarray, state: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        msg = control_to_ackermann_msg(AckermannDriveStamped, u, state, self.params, stamp)
        self.delta_estimate = msg.drive.steering_angle
        self.command_pub.publish(msg)

    def _publish_debug(
        self,
        result,
        component_margins: dict,
        u_human: np.ndarray,
        u_filtered: np.ndarray,
        override: bool,
        status: str,
    ):
        if not self.params.publish_debug:
            return

        value_msg = Float32()
        value_msg.data = float(result.value)
        self.value_pub.publish(value_msg)

        grad_msg = Float64MultiArray()
        grad_msg.data = [float(v) for v in result.gradient]
        self.grad_pub.publish(grad_msg)

        override_msg = Bool()
        override_msg.data = bool(override)
        self.override_pub.publish(override_msg)

        binding_msg = String()
        binding_msg.data = result.binding_constraint
        self.binding_pub.publish(binding_msg)

        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

        margins_msg = Float64MultiArray()
        margins_msg.data = [
            float(component_margins["lane"]),
            float(component_margins["obstacle"]),
            float(component_margins["forward_obstacle"]),
            float(component_margins["traffic"]),
            float(component_margins["kinematic"]),
        ]
        self.margins_pub.publish(margins_msg)

        u_h_msg = Float64MultiArray()
        u_h_msg.data = [float(v) for v in u_human]
        self.u_human_pub.publish(u_h_msg)

        u_f_msg = Float64MultiArray()
        u_f_msg.data = [float(v) for v in u_filtered]
        self.u_filtered_pub.publish(u_f_msg)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for state in result.trajectory:
            path_msg.poses.append(self._pose_from_state(state))
        self.backup_path_pub.publish(path_msg)

    def _pose_from_state(self, state: np.ndarray):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.orientation.z = float(np.sin(state[3] / 2.0))
        pose.pose.orientation.w = float(np.cos(state[3] / 2.0))
        return pose

    def _maybe_rebuild_lane_context(self, state: np.ndarray):
        position = state[:2]
        if self.last_lane_center is not None:
            distance_delta = np.linalg.norm(position - self.last_lane_center)
            yaw_delta = abs((state[3] - self.last_lane_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            if (
                distance_delta < self.params.lane_context_rebuild_distance_m
                and yaw_delta < self.params.lane_context_rebuild_yaw_rad
            ):
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = position.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as exc:
            self.get_logger().warn(f"lane context rebuild failed, using previous/fallback lane: {exc}")

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_state is None:
            return

        state = self.last_state.copy()
        ctx = MarginContext(self.lane_context, self.static_obstacles, self.traffic, self.params)

        odom_stale = self.last_odom_time is None or now - self.last_odom_time > self.params.stale_timeout_s
        human_stale = self.last_human_time is None or now - self.last_human_time > self.params.stale_timeout_s
        try:
            result = evaluate_barrier(state, ctx)
            component_margins = margin_components(state, ctx)
            u_human = self._human_control(state)

            if odom_stale or human_stale:
                u_filtered = brake_and_recenter(state, self.lane_context, self.params)
                status = "stale_input_brake_recenter"
            else:
                decision = self._select_heuristic_control(
                    state,
                    ctx,
                    component_margins,
                    u_human,
                )
                u_filtered = decision.control
                status = decision.status

            deviates = np.linalg.norm(u_filtered - u_human, ord=np.inf) > self.params.passthrough_tolerance
            if deviates:
                self.override_hold = self.params.hysteresis_cycles
            elif self.override_hold > 0:
                self.override_hold -= 1
                status = "hysteresis"
            override = deviates or self.override_hold > 0 or status.endswith("brake_recenter")
            self._publish_command(u_filtered, state)
            self._publish_debug(result, component_margins, u_human, u_filtered, override, status)

        except Exception:
            self.get_logger().error(f"safety filter fault:\n{traceback.format_exc()}")
            u_backup = brake_and_recenter(state, self.lane_context, self.params)
            self._publish_command(u_backup, state)

    def _select_heuristic_control(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        margins: dict,
        u_human: np.ndarray,
    ) -> HeuristicDecision:
        # immediate_forward_obstacle_brake: stop when a static obstacle is close
        # in the forward corridor.
        if margins["forward_obstacle"] < self.params.forward_obstacle_brake_margin_m:
            return HeuristicDecision(
                emergency_brake(state, self.params),
                "immediate_forward_obstacle_brake",
            )

        # dynamic_traffic_brake: moving obstacles get a wider margin because
        # they can enter the ego lane between control ticks.
        if margins["traffic"] < self.params.traffic_guard_margin_m:
            return HeuristicDecision(
                emergency_brake(state, self.params),
                "dynamic_traffic_brake",
            )

        # lane_boundary_recenter: if the footprint is near the lane edge, slow
        # down and steer toward the lane tangent and centerline.
        if margins["lane"] < self.params.lane_guard_margin_m:
            return HeuristicDecision(
                lane_recovery_control(state, ctx.lane, self.params, u_human),
                "lane_boundary_recenter",
            )

        candidates = [(u_human, "human_passthrough")]
        for control, label in (
            self._predicted_lane_departure_control(state, ctx, u_human),
            self._corner_pre_turn_control(state, ctx, u_human),
            self._obstacle_approach_slowdown_control(state, ctx, margins, u_human),
            self._side_obstacle_lane_bias_control(state, ctx, u_human),
            self._kinematic_limit_recovery_control(state, margins, u_human),
        ):
            if control is not None:
                candidates.append((control, label))

        chosen_control, chosen_label = candidates[-1]
        if margins["obstacle"] < self.params.obstacle_guard_margin_m:
            candidates.append((brake_and_recenter(state, ctx.lane, self.params), "candidate_brake_recenter"))
            candidates.append((emergency_brake(state, self.params), "candidate_emergency_brake"))
            chosen_control, chosen_label = self._best_candidate_by_margin(state, ctx, candidates)

        return HeuristicDecision(clip_control(chosen_control, self.params), chosen_label)

    def _predicted_lane_departure_control(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        u_human: np.ndarray,
    ) -> tuple:
        if ctx.lane.is_fallback:
            return None, ""

        horizon = max(3, min(10, self.params.horizon_H // 2))
        predicted = rollout(state, lambda _: u_human, self.params, horizon=horizon)
        min_lane_margin = min(margin_lane(x, ctx.lane, self.params) for x in predicted)
        if min_lane_margin >= self.params.lane_guard_margin_m:
            return None, ""

        recovery = lane_recovery_control(state, ctx.lane, self.params, u_human)
        accel = min(recovery[0], u_human[0], 0.0)
        return np.array([accel, recovery[1]], dtype=float), "predicted_lane_departure"

    def _corner_pre_turn_control(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        u_human: np.ndarray,
    ) -> tuple:
        if ctx.lane.is_fallback or len(ctx.lane.centerline) < 3:
            return None, ""

        lookahead_tangent = self._lookahead_lane_tangent(state, ctx.lane)
        if lookahead_tangent is None:
            return None, ""

        current_error, sample = heading_error_to_lane(state, ctx.lane)
        upcoming_error = wrap_angle(lookahead_tangent - float(state[3]))
        tangent_change = abs(wrap_angle(lookahead_tangent - sample.tangent))
        if tangent_change < 0.22:
            return None, ""

        desired_delta = np.clip(
            upcoming_error - np.arctan2(self.params.K_e * sample.signed_lateral_error, abs(state[2]) + self.params.v_eps),
            self.params.delta_min,
            self.params.delta_max,
        )
        omega = np.clip(self.params.K_p * (desired_delta - state[4]), self.params.omega_min, self.params.omega_max)
        corner_speed = min(self.params.lane_recovery_speed_mps, 0.55)
        accel = min(u_human[0], self.params.lane_recovery_accel_gain * (corner_speed - state[2]))
        if abs(upcoming_error) < abs(current_error) and state[2] < corner_speed:
            accel = min(accel, u_human[0])
        return np.array([accel, omega], dtype=float), "corner_pre_turn"

    def _obstacle_approach_slowdown_control(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        margins: dict,
        u_human: np.ndarray,
    ) -> tuple:
        if not ctx.obstacles:
            return None, ""
        if margins["forward_obstacle"] >= self.params.forward_obstacle_distance_m:
            return None, ""

        clear_span = max(
            self.params.forward_obstacle_distance_m - self.params.forward_obstacle_brake_margin_m,
            1e-6,
        )
        risk = np.clip(
            (self.params.forward_obstacle_distance_m - margins["forward_obstacle"]) / clear_span,
            0.0,
            1.0,
        )
        target_speed = (1.0 - risk) * self.params.v_max
        accel = min(u_human[0], self.params.lane_recovery_accel_gain * (target_speed - state[2]))
        omega = recenter_control(state, ctx.lane, self.params, accel)[1]
        return np.array([accel, omega], dtype=float), "obstacle_approach_slowdown"

    def _side_obstacle_lane_bias_control(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        u_human: np.ndarray,
    ) -> tuple:
        if ctx.lane.is_fallback or not ctx.obstacles:
            return None, ""

        closest = self._closest_side_obstacle(state, ctx)
        if closest is None:
            return None, ""

        lateral, longitudinal, distance = closest
        _, sample = heading_error_to_lane(state, ctx.lane)
        left_margin = sample.width_left - sample.signed_lateral_error
        right_margin = sample.width_right + sample.signed_lateral_error
        steer_away_sign = -np.sign(lateral)
        has_room = right_margin > self.params.lane_guard_margin_m if steer_away_sign < 0 else left_margin > self.params.lane_guard_margin_m
        if not has_room:
            return brake_and_recenter(state, ctx.lane, self.params), "side_obstacle_no_lane_room_brake"

        risk = np.clip(1.0 - distance / max(self.params.forward_obstacle_distance_m, 1e-6), 0.0, 1.0)
        desired_delta = np.clip(
            state[4] + steer_away_sign * risk * 0.18,
            self.params.delta_min,
            self.params.delta_max,
        )
        omega = np.clip(self.params.K_p * (desired_delta - state[4]), self.params.omega_min, self.params.omega_max)
        accel = min(u_human[0], self.params.lane_recovery_accel_gain * (0.7 - state[2]))
        if longitudinal < self.params.forward_obstacle_brake_margin_m:
            accel = min(accel, self.params.a_min)
        return np.array([accel, omega], dtype=float), "side_obstacle_lane_bias"

    def _kinematic_limit_recovery_control(
        self,
        state: np.ndarray,
        margins: dict,
        u_human: np.ndarray,
    ) -> tuple:
        _, _, v, _, delta = state
        near_speed_limit = v > self.params.v_max - self.params.r_safe_kin and u_human[0] > 0.0
        near_left_limit = delta > self.params.delta_max - self.params.r_safe_kin and u_human[1] > 0.0
        near_right_limit = delta < self.params.delta_min + self.params.r_safe_kin and u_human[1] < 0.0
        if not (near_speed_limit or near_left_limit or near_right_limit or margins["kinematic"] < 0.0):
            return None, ""

        accel = min(u_human[0], 0.0) if near_speed_limit else u_human[0]
        omega = u_human[1]
        if near_left_limit or near_right_limit:
            omega = np.clip(self.params.K_p * (0.0 - delta), self.params.omega_min, self.params.omega_max)
        return np.array([accel, omega], dtype=float), "kinematic_limit_recovery"

    def _best_candidate_by_margin(
        self,
        state: np.ndarray,
        ctx: MarginContext,
        candidates: list,
    ) -> tuple:
        # candidate_margin_check: when current margins are poor, choose the
        # candidate that gives the best short-term implicit CBF margin.
        scored = [
            (self._next_barrier_value(state, control, ctx), control, label)
            for control, label in candidates
        ]
        _, control, label = max(scored, key=lambda item: item[0])
        return control, label

    def _next_barrier_value(self, state: np.ndarray, control: np.ndarray, ctx: MarginContext) -> float:
        h_next, _, _, _ = implicit_barrier_value(step(state, control, self.params), ctx)
        return float(h_next)

    def _lookahead_lane_tangent(self, state: np.ndarray, lane: LaneContext):
        position = np.asarray(state[:2], dtype=float)
        distances = np.linalg.norm(lane.centerline - position, axis=1)
        start_idx = int(np.argmin(distances))
        if start_idx >= len(lane.centerline) - 2:
            return None

        lookahead_m = np.clip(0.6 + max(0.0, state[2]) * 1.2, 0.6, 1.8)
        traveled = 0.0
        for idx in range(start_idx, len(lane.centerline) - 1):
            segment = lane.centerline[idx + 1] - lane.centerline[idx]
            traveled += float(np.linalg.norm(segment))
            if traveled >= lookahead_m:
                return float(np.arctan2(segment[1], segment[0]))
        return float(lane.tangent[-1])

    def _closest_side_obstacle(self, state: np.ndarray, ctx: MarginContext):
        position = np.asarray(state[:2], dtype=float)
        forward = np.array([np.cos(state[3]), np.sin(state[3])], dtype=float)
        left = np.array([-forward[1], forward[0]], dtype=float)
        corridor_half_width = 0.5 * self.params.forward_obstacle_width_m + self.params.truck_radius_m

        best = None
        for obs in ctx.obstacles:
            relative = np.asarray(obs.position[:2], dtype=float) - position
            longitudinal = float(relative @ forward)
            if longitudinal < -obs.radius or longitudinal > self.params.forward_obstacle_distance_m:
                continue
            lateral = float(relative @ left)
            lateral_clearance = abs(lateral) - corridor_half_width - obs.radius
            if lateral_clearance <= 0.0 or lateral_clearance > self.params.obstacle_guard_margin_m:
                continue
            distance = float(np.linalg.norm(relative) - obs.radius - self.params.truck_radius_m)
            if best is None or distance < best[2]:
                best = (lateral, longitudinal, distance)
        return best


def main(args=None):
    rclpy.init(args=args)
    node = CbfHeuristicSafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
