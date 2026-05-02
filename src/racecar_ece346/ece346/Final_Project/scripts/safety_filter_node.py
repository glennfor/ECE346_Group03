#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from ece346.Final_Project.safety_filter.backup_policy import (
    brake_and_recenter,
    lane_recovery_control,
)
from ece346.Final_Project.safety_filter.barrier import evaluate_barrier, implicit_barrier_value
from ece346.Final_Project.safety_filter.config import declare_and_load
from ece346.Final_Project.safety_filter.dynamics import control_jacobian, step
from ece346.Final_Project.safety_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.Final_Project.safety_filter.margins import MarginContext, margin_components
from ece346.Final_Project.safety_filter.obstacle_memory import ObstacleMemory
from ece346.Final_Project.safety_filter.qp import solve_box_halfspace_qp
from ece346.Final_Project.safety_filter.ros_utils import (
    ackermann_msg_to_control,
    control_to_ackermann_msg,
    control_to_servo_msg,
    marker_array_to_obstacles,
    odom_to_state,
    odometry_array_to_obstacles,
    servo_msg_to_control,
)

from racecar_msgs.msg import OdometryArray, ServoMsg

try:
    from ackermann_msgs.msg import AckermannDriveStamped
except Exception:
    AckermannDriveStamped = None


class SafetyFilterNode(Node):
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
        )

        self.static_memory = ObstacleMemory(
            self.params.obstacle_memory_ttl_s,
            self.params.obstacle_memory_growth,
            self.params.obstacle_radius_default,
        )
        self.traffic = []
        self.override_hold = 0

        self._setup_io()
        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self.control_step)
        self.get_logger().info(
            f"safety_filter_node ready: mode={self.params.command_mode}, "
            f"odom={self.params.odom_topic}, human={self.params.human_control_topic}, "
            f"out={self.params.filtered_control_topic}"
        )

    def _setup_io(self):
        self.odom_sub = self.create_subscription(
            Odometry, self.params.odom_topic, self.odom_callback, 10
        )

        if self.params.command_mode == "ackermann":
            if AckermannDriveStamped is None:
                raise RuntimeError("ackermann_msgs is unavailable")
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
        else:
            self.human_sub = self.create_subscription(
                ServoMsg,
                self.params.human_control_topic,
                self.human_callback,
                10,
            )
            self.command_pub = self.create_publisher(
                ServoMsg,
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

    def _maybe_rebuild_lane_context(self, state: np.ndarray):
        p = state[:2]
        if self.last_lane_center is not None:
            distance_delta = np.linalg.norm(p - self.last_lane_center)
            yaw_delta = abs((state[3] - self.last_lane_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            if (
                distance_delta < self.params.lane_context_rebuild_distance_m
                and yaw_delta < self.params.lane_context_rebuild_yaw_rad
            ):
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = p.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as exc:
            self.get_logger().warn(f"lane context rebuild failed, using previous/fallback lane: {exc}")

    def _human_control(self, state: np.ndarray) -> np.ndarray:
        if self.last_human_msg is None:
            return np.array([0.0, 0.0], dtype=float)
        if self.params.command_mode == "ackermann":
            return ackermann_msg_to_control(self.last_human_msg, state, self.params)
        return servo_msg_to_control(self.last_human_msg, state, self.params)

    def _publish_command(self, u: np.ndarray, state: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        if self.params.command_mode == "ackermann":
            msg = control_to_ackermann_msg(AckermannDriveStamped, u, state, self.params, stamp)
            self.delta_estimate = msg.drive.steering_angle
        else:
            msg = control_to_servo_msg(ServoMsg, u, state, self.params, stamp)
            self.delta_estimate = msg.steer
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
            pose = self._pose_from_state(state)
            path_msg.poses.append(pose)
        self.backup_path_pub.publish(path_msg)

    def _pose_from_state(self, state: np.ndarray):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.orientation.z = float(np.sin(state[3] / 2.0))
        pose.pose.orientation.w = float(np.cos(state[3] / 2.0))
        return pose

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_state is None:
            return

        state = self.last_state.copy()
        obstacles = self.static_memory.get(now)
        ctx = MarginContext(self.lane_context, obstacles, self.traffic, self.params)

        odom_stale = self.last_odom_time is None or now - self.last_odom_time > self.params.stale_timeout_s
        human_stale = self.last_human_time is None or now - self.last_human_time > self.params.stale_timeout_s

        try:
            result = evaluate_barrier(state, ctx)
            component_margins = margin_components(state, ctx)
            u_human = self._human_control(state)

            if odom_stale or human_stale:
                u_filtered = brake_and_recenter(state, self.lane_context, self.params)
                status = "fallback_stale"
            elif component_margins["lane"] < self.params.lane_guard_margin_m:
                u_filtered = lane_recovery_control(state, self.lane_context, self.params)
                status = "lane_guard_recenter"
            else:
                u_backup = brake_and_recenter(state, self.lane_context, self.params)
                h_human_next = self._next_barrier_value(state, u_human, ctx)

                f_human = step(state, u_human, self.params)
                g = control_jacobian(self.params).T @ result.gradient
                c = (
                    -self.params.lambda_cbf * result.value
                    - float(result.gradient @ (f_human - state))
                    + float(g @ u_human)
                )
                qp_result = solve_box_halfspace_qp(u_human, g, c, self.params)
                u_filtered = qp_result.control
                status = qp_result.status

                if result.value < 0.0 and h_human_next > result.value + self.params.recovery_h_improvement:
                    u_filtered = u_human
                    status = "recovery_human_improves_h"
                elif status != "optimal":
                    u_filtered, status = self._best_recovery_control(
                        state,
                        ctx,
                        [
                            (u_backup, "fallback_qp_infeasible"),
                            (u_human, "recovery_human"),
                        ],
                    )
                elif self.params.exact_safety_check:
                    h_next = self._next_barrier_value(state, u_filtered, ctx)
                    threshold = (1.0 - self.params.lambda_cbf) * result.value
                    if h_next < threshold - 1e-6:
                        u_filtered, status = self._best_recovery_control(
                            state,
                            ctx,
                            [
                                (u_backup, "fallback_exact_check"),
                                (u_filtered, "qp_exact_best_effort"),
                                (u_human, "recovery_human"),
                            ],
                        )

            deviates = np.linalg.norm(u_filtered - u_human, ord=np.inf) > self.params.passthrough_tolerance
            if deviates:
                self.override_hold = self.params.hysteresis_cycles
            elif self.override_hold > 0:
                self.override_hold -= 1
                status = "hysteresis"

            override = deviates or self.override_hold > 0 or status.startswith("fallback")
            self._publish_command(u_filtered, state)
            self._publish_debug(result, component_margins, u_human, u_filtered, override, status)

        except Exception:
            self.get_logger().error(f"safety filter fault:\n{traceback.format_exc()}")
            u_backup = brake_and_recenter(state, self.lane_context, self.params)
            self._publish_command(u_backup, state)

    def _next_barrier_value(self, state: np.ndarray, control: np.ndarray, ctx: MarginContext) -> float:
        h_next, _, _, _ = implicit_barrier_value(step(state, control, self.params), ctx)
        return float(h_next)

    def _best_recovery_control(self, state: np.ndarray, ctx: MarginContext, candidates: list) -> tuple:
        scored = [
            (self._next_barrier_value(state, control, ctx), control, label)
            for control, label in candidates
        ]
        _, control, label = max(scored, key=lambda item: item[0])
        return control, label


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

