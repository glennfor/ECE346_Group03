"""ROS 2 node: implicit Backup-CBF with box QP (Ackermann Final Project pipeline).

``main()`` runs the Lanelet-backed implicit Backup-CBF node. ``heuristic_main()`` runs
the legacy grid-search ``CbfQpSafetyFilterNode`` (routing Path + ``CbfQpFilter``).
"""

import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from racecar_msgs.msg import OdometryArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from visualization_msgs.msg import MarkerArray

from dataclasses import fields, replace
from typing import Optional

from .backup_policy import brake_and_recenter
from .barrier import evaluate_barrier, implicit_barrier_value
from .config import CbfQpConfig, DEFAULT_CONFIG_PATH, declare_and_load, load_config
from .filter import CbfQpFilter, FilterCommand
from .geometry import obstacles_from_msg, path_from_msg, state_from_odom
from .dynamics import control_jacobian, step
from .guards import select_hard_guard_control
from .lane_context import LaneContext, LaneletContextBuilder
from .margins import MarginContext, margin_components
from .obstacle_memory import ObstacleMemory
from .qp import solve_box_halfspace_qp
from .ros_utils import (
    ackermann_msg_to_control,
    control_to_ackermann_msg,
    marker_array_to_obstacles,
    odom_to_state,
    odometry_array_to_obstacles,
)


class CbfSafetyFilterNode(Node):
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
        self.timer = self.create_timer(1.0 / max(self.params.control_rate_hz, 1e-3), self.control_step)
        self.get_logger().info(
            f"cbf_qp safety_filter_node ready: ackermann "
            f"odom={self.params.odom_topic}, human={self.params.human_control_topic}, "
            f"out={self.params.filtered_control_topic}"
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

        if self.params.publish_debug:
            self.value_pub = self.create_publisher(Float32, "/safety/value", 1)
            self.grad_pub = self.create_publisher(Float64MultiArray, "/safety/grad", 1)
            self.override_pub = self.create_publisher(Bool, "/safety/override", 1)
            self.binding_pub = self.create_publisher(String, "/safety/binding_constraint", 1)
            self.status_pub = self.create_publisher(String, "/safety/status", 1)
            self.margins_pub = self.create_publisher(Float64MultiArray, "/safety/margins", 1)
            self.u_human_pub = self.create_publisher(Float64MultiArray, "/safety/u_human", 1)
            self.u_filtered_pub = self.create_publisher(Float64MultiArray, "/safety/u_filtered", 1)
            self.backup_path_pub = self.create_publisher(Path, "/safety/backup_traj", 1)
        else:
            self.value_pub = None

    def odom_callback(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9
        self._maybe_rebuild_lane_context(self.last_state)

    def human_callback(self, msg: AckermannDriveStamped):
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
            self.get_logger().warn(
                f"lane context rebuild failed, using previous/fallback lane: {exc}"
            )

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
        for st in result.trajectory:
            pose = self._pose_from_state(st)
            path_msg.poses.append(pose)
        self.backup_path_pub.publish(path_msg)

    def _pose_from_state(self, state: np.ndarray) -> PoseStamped:
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
            else:
                guard = select_hard_guard_control(
                    state,
                    self.lane_context,
                    component_margins,
                    self.params,
                    u_human,
                )
                if guard.control is not None:
                    u_filtered = guard.control
                    status = guard.status
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

    def _best_recovery_control(self, state: np.ndarray, ctx: MarginContext, candidates: list):
        scored = [
            (self._next_barrier_value(state, control, ctx), control, label)
            for control, label in candidates
        ]
        _, control, label = max(scored, key=lambda item: item[0])
        return control, label


class CbfQpSafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_node")

        self.declare_parameter("cbf_qp_config", str(DEFAULT_CONFIG_PATH))
        config_path = self.get_parameter("cbf_qp_config").value
        self.config = self._declare_and_read_config(load_config(config_path))
        self.filter = CbfQpFilter(self.config)

        self._latest_teleop: Optional[AckermannDriveStamped] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_obstacles: Optional[MarkerArray] = None
        self._latest_path: Optional[Path] = None
        self._teleop_time: Optional[float] = None
        self._odom_time: Optional[float] = None
        self._obstacle_time: Optional[float] = None
        self._path_time: Optional[float] = None
        self._last_log_time = 0.0

        self.pub = self.create_publisher(
            AckermannDriveStamped, self.config.drive_topic, 1
        )
        self.create_subscription(
            AckermannDriveStamped,
            self.config.teleop_topic,
            self._teleop_cb,
            1,
        )
        self.create_subscription(Odometry, self.config.odom_topic, self._odom_cb, 1)
        self.create_subscription(
            MarkerArray,
            self.config.static_obs_topic,
            self._obstacles_cb,
            1,
        )
        self.create_subscription(Path, self.config.routing_path_topic, self._path_cb, 1)
        self.create_timer(1.0 / max(self.config.publish_rate, 1e-3), self._tick)

        self.get_logger().info(
            "cbf_qp safety_filter_node ready: "
            f"{self.config.teleop_topic} + {self.config.odom_topic} + "
            f"{self.config.static_obs_topic} + {self.config.routing_path_topic} "
            f"-> {self.config.drive_topic}"
        )

    def _declare_and_read_config(self, config: CbfQpConfig) -> CbfQpConfig:
        values = {}
        for field in fields(CbfQpConfig):
            default = getattr(config, field.name)
            self.declare_parameter(field.name, list(default) if isinstance(default, tuple) else default)
            value = self.get_parameter(field.name).value
            if isinstance(default, tuple) and isinstance(value, list):
                value = tuple(value)
            values[field.name] = value
        return replace(config, **values)

    def _teleop_cb(self, msg: AckermannDriveStamped):
        self._latest_teleop = msg
        self._teleop_time = self._now_sec()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_time = self._now_sec()

    def _obstacles_cb(self, msg: MarkerArray):
        self._latest_obstacles = msg
        self._obstacle_time = self._now_sec()

    def _path_cb(self, msg: Path):
        self._latest_path = msg
        self._path_time = self._now_sec()

    def _tick(self):
        if self._latest_teleop is None:
            return

        now = self._now_sec()
        if self._is_stale(self._teleop_time, self.config.teleop_timeout_sec):
            self._publish_drive(
                speed=0.0,
                steering=0.0,
                source=self._latest_teleop,
            )
            self._log_periodic(now, "stale teleop; publishing stop")
            return

        if self._latest_odom is None or self._is_stale(
            self._odom_time, self.config.odom_timeout_sec
        ):
            self._publish_drive(
                speed=self.config.stale_odom_speed,
                steering=self.config.stale_odom_steering,
                source=self._latest_teleop,
            )
            self._log_periodic(now, "missing/stale odom; publishing failsafe")
            return

        obstacle_msg = (
            None
            if self._is_stale(self._obstacle_time, self.config.obstacle_timeout_sec)
            else self._latest_obstacles
        )
        path_msg = (
            None
            if self._is_stale(self._path_time, self.config.path_timeout_sec)
            else self._latest_path
        )

        command = self.filter.filter_command(
            human_speed=self._latest_teleop.drive.speed,
            human_steering=self._latest_teleop.drive.steering_angle,
            state=state_from_odom(self._latest_odom),
            path=path_from_msg(path_msg),
            obstacles=obstacles_from_msg(
                obstacle_msg,
                self.config.obstacle_radius_buffer,
            ),
        )
        self._publish_drive(
            speed=command.speed,
            steering=command.steering_angle,
            source=self._latest_teleop,
        )
        if command.is_override:
            self._log_command(now, command)

    def _publish_drive(
        self,
        speed: float,
        steering: float,
        source: AckermannDriveStamped,
    ):
        out = AckermannDriveStamped()
        out.header = source.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive = source.drive
        out.drive.speed = float(speed)
        out.drive.steering_angle = float(steering)
        self.pub.publish(out)

    def _log_command(self, now: float, command: FilterCommand):
        lane = "none" if command.lane_margin is None else f"{command.lane_margin:.2f}"
        obs = (
            "none"
            if command.obstacle_margin is None
            else f"{command.obstacle_margin:.2f}"
        )
        self._log_periodic(
            now,
            "cbf override "
            f"reason={command.reason} speed={command.speed:.2f} "
            f"steer={command.steering_angle:.2f} lane={lane} obs={obs}",
        )

    def _log_periodic(self, now: float, message: str):
        if now - self._last_log_time >= self.config.log_period_sec:
            self.get_logger().info(message)
            self._last_log_time = now

    def _is_stale(self, stamp_sec: Optional[float], timeout_sec: float) -> bool:
        if stamp_sec is None:
            return True
        return self._now_sec() - stamp_sec > timeout_sec

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

def heuristic_main(args=None):
    rclpy.init(args=args)
    node = CbfQpSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CbfSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
