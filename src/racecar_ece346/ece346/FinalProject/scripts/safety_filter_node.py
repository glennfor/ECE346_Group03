#!/usr/bin/env python3
"""
Final Project — Safety Filter (STUDENT SKELETON)

You will build a ROS2 node that sits between a human driver (PS4 joystick)
and the vehicle, intervening only when necessary to keep the car safe.

Pipeline:

    /joy --> joy_to_ackermann --> /teleop ┐
                                           │
                             /SLAM/Pose ───┤---> [safety_filter_node] ---> /drive
                                           │
                       /Obstacles/Static ──┘

Topics you will work with:

    /teleop            ackermann_msgs/AckermannDriveStamped  (human command in)
    <odom_topic>       nav_msgs/Odometry                     (vehicle state in)
    /Obstacles/Static  visualization_msgs/MarkerArray        (obstacles in)
    /drive             ackermann_msgs/AckermannDriveStamped  (safe command out)

------------------------------------------------------------------------------
Task 1: Build the node plumbing (subscribers, publisher, timer).
Task 2: Implement the safety filter logic.
------------------------------------------------------------------------------

You may:
  - Add imports, helper methods, and ROS parameters to THIS file.
  - Add your own files anywhere under FinalProject/ — e.g. an `ilqr/`
    subfolder with your ILQR solver, an `ilqr_params.yaml` with cost
    weights, utility modules, etc. Load yaml configs from within this
    node using `open()` + `yaml.safe_load()`, or declare a ROS param
    for the config path and set it in `final_project_*.yaml`.
  - Edit any yaml under `FinalProject/config/` — add parameters, tune
    thresholds, change topic names. All existing yaml values are
    documented and intended to be tunable.

You should NOT need to rewrite the launch files, the plumbing nodes
(joy_to_ackermann, drive_to_servo, etc.), or the CMakeLists top-level
install lists. Only touch those if you are adding a new standalone
executable — in which case ask first.
"""

import copy
import math
from dataclasses import dataclass, fields, replace
from typing import List, Optional, Sequence, Tuple

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
# from ece346.FinalProject.cbf_heuristic.node import main as cbf_heuristic_main
# from ece346.FinalProject.cbf_qp.node import main as cbf_qp_main
# from ece346.FinalProject.ilqr.node import main as ilqr_main
from ece346.FinalProject.ilqr.config import (DEFAULT_CONFIG_PATH, IlqrQpConfig,
                                             load_config)
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float
    steering_angle: float = 0.0


@dataclass(frozen=True)
class PathPoint:
    x: float
    y: float
    left_width: float
    right_width: float
    speed_limit: float


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx, qy, qz, qw):
    """Extract yaw (heading, rad) from a quaternion. Useful for Task 2."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def state_from_odom(odom: Odometry, steering_angle: float) -> VehicleState:
    pose = odom.pose.pose
    twist = odom.twist.twist
    return VehicleState(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=yaw_from_quat(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
        speed=max(0.0, float(twist.linear.x)),
        steering_angle=float(steering_angle),
    )


def path_from_msg(path_msg: Optional[Path]) -> List[PathPoint]:
    path: List[PathPoint] = []
    if path_msg is None:
        return path

    for pose_stamped in path_msg.poses:
        pose = pose_stamped.pose
        left_width = float(pose.orientation.x)
        right_width = float(pose.orientation.y)
        if left_width <= 0.0 and right_width <= 0.0:
            continue
        path.append(
            PathPoint(
                x=float(pose.position.x),
                y=float(pose.position.y),
                left_width=max(0.0, left_width),
                right_width=max(0.0, right_width),
                speed_limit=max(0.0, float(pose.orientation.z)),
            )
        )
    return path


def fallback_path_ahead(
    state: VehicleState,
    length: float,
    width: float,
    speed_limit: float,
    samples: int,
) -> List[PathPoint]:
    count = max(2, int(samples))
    half_width = max(0.05, 0.5 * float(width))
    points: List[PathPoint] = []
    for idx in range(count):
        distance = float(length) * idx / max(1, count - 1)
        points.append(
            PathPoint(
                x=state.x + distance * math.cos(state.yaw),
                y=state.y + distance * math.sin(state.yaw),
                left_width=half_width,
                right_width=half_width,
                speed_limit=max(0.0, float(speed_limit)),
            )
        )
    return points


def obstacles_from_msg(
    marker_array: Optional[MarkerArray],
    radius_buffer: float,
) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    if marker_array is None:
        return obstacles

    for marker in marker_array.markers:
        if getattr(marker, "action", 0) == 2:
            continue
        sx = max(0.0, float(marker.scale.x))
        sy = max(0.0, float(marker.scale.y))
        obstacles.append(
            Obstacle(
                x=float(marker.pose.position.x),
                y=float(marker.pose.position.y),
                radius=0.5 * math.hypot(sx, sy) + float(radius_buffer),
            )
        )
    return obstacles


def closest_path_projection(
    state: VehicleState,
    path: Sequence[PathPoint],
) -> Optional[Tuple[PathPoint, float, float]]:
    if len(path) < 2:
        return None

    best = None
    best_dist_sq = float("inf")
    for idx in range(len(path) - 1):
        p0 = path[idx]
        p1 = path[idx + 1]
        vx = p1.x - p0.x
        vy = p1.y - p0.y
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1e-9:
            continue

        t = clamp(
            ((state.x - p0.x) * vx + (state.y - p0.y) * vy) / seg_len_sq,
            0.0,
            1.0,
        )
        proj_x = p0.x + t * vx
        proj_y = p0.y + t * vy
        dx = state.x - proj_x
        dy = state.y - proj_y
        dist_sq = dx * dx + dy * dy
        if dist_sq >= best_dist_sq:
            continue

        tangent_yaw = math.atan2(vy, vx)
        lateral = -math.sin(tangent_yaw) * dx + math.cos(tangent_yaw) * dy
        best = (
            PathPoint(
                x=proj_x,
                y=proj_y,
                left_width=(1.0 - t) * p0.left_width + t * p1.left_width,
                right_width=(1.0 - t) * p0.right_width + t * p1.right_width,
                speed_limit=(1.0 - t) * p0.speed_limit + t * p1.speed_limit,
            ),
            lateral,
            tangent_yaw,
        )
        best_dist_sq = dist_sq
    return best


def step_state(
    state: VehicleState,
    speed_cmd: float,
    steering_cmd: float,
    wheelbase: float,
    dt: float,
) -> VehicleState:
    speed = max(0.0, float(speed_cmd))
    steering = float(steering_cmd)
    yaw_rate = speed * math.tan(steering) / max(float(wheelbase), 1e-3)
    yaw_mid = state.yaw + 0.5 * yaw_rate * dt
    return VehicleState(
        x=state.x + speed * math.cos(yaw_mid) * dt,
        y=state.y + speed * math.sin(yaw_mid) * dt,
        yaw=wrap_angle(state.yaw + yaw_rate * dt),
        speed=speed,
        steering_angle=steering,
    )


def rollout_constant_command(
    state: VehicleState,
    speed_cmd: float,
    steering_cmd: float,
    wheelbase: float,
    dt: float,
    horizon_sec: float,
    response_delay_sec: float,
    min_projection_speed: float,
) -> List[VehicleState]:
    steps = max(1, int(math.ceil(float(horizon_sec) / max(float(dt), 1e-3))))
    delayed_steps = max(
        0,
        int(math.ceil(float(response_delay_sec) / max(float(dt), 1e-3))),
    )
    states = [state]
    current = state
    for idx in range(steps):
        speed = current.speed if idx < delayed_steps else speed_cmd
        if speed_cmd > 0.0:
            speed = max(speed, min_projection_speed)
        current = step_state(current, speed, steering_cmd, wheelbase, dt)
        states.append(current)
    return states


def lane_margin(
    state: VehicleState,
    path: Sequence[PathPoint],
    safety_buffer: float,
) -> Optional[float]:
    projection = closest_path_projection(state, path)
    if projection is None:
        return None
    point, lateral, _ = projection
    left_clearance = point.left_width - lateral
    right_clearance = point.right_width + lateral
    return min(left_clearance, right_clearance) - safety_buffer


def min_lane_margin(
    states: Sequence[VehicleState],
    path: Sequence[PathPoint],
    safety_buffer: float,
) -> Optional[float]:
    margins = [
        margin
        for margin in (lane_margin(state, path, safety_buffer) for state in states)
        if margin is not None
    ]
    return min(margins) if margins else None


def min_obstacle_margin(
    states: Sequence[VehicleState],
    obstacles: Sequence[Obstacle],
    safety_buffer: float,
) -> Optional[float]:
    if not obstacles:
        return None
    best = float("inf")
    for state in states:
        for obstacle in obstacles:
            best = min(
                best,
                math.hypot(state.x - obstacle.x, state.y - obstacle.y)
                - obstacle.radius
                - safety_buffer,
            )
    return best


def forward_obstacle_margin(
    state: VehicleState,
    obstacles: Sequence[Obstacle],
    safety_buffer: float,
    forward_width: float,
    max_distance: float,
) -> Optional[float]:
    if not obstacles:
        return None

    best = float("inf")
    cos_yaw = math.cos(state.yaw)
    sin_yaw = math.sin(state.yaw)
    for obstacle in obstacles:
        dx = obstacle.x - state.x
        dy = obstacle.y - state.y
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        lateral_limit = forward_width + obstacle.radius + safety_buffer
        if 0.0 <= forward <= max_distance and abs(lateral) <= lateral_limit:
            best = min(best, forward - obstacle.radius - safety_buffer)
    return None if best == float("inf") else best


class SafetyFilterNode(Node):

    # =========================================================================
    # TASK 1 — Node setup (subscribers, publisher, timer)
    # =========================================================================
    #
    # Fill in __init__ below so that the node:
    #   1. Declares ROS parameters for each topic name and for the publish rate.
    #      Hint: use self.declare_parameter('name', default_value). The yaml
    #      file (final_project_*.yaml) will override these at launch time.
    #      Required parameter names (match the yaml):
    #          teleop_topic, drive_topic, odom_topic, static_obs_topic,
    #          publish_rate
    #
    #   2. Creates three subscribers that cache the latest message each:
    #          /teleop            -> AckermannDriveStamped
    #          <odom_topic>       -> Odometry
    #          /Obstacles/Static  -> MarkerArray
    #      Hint: self.create_subscription(MsgType, topic, callback, queue_size)
    #      Each callback can be a one-liner that stores msg into an instance
    #      attribute (e.g. self._latest_teleop = msg).
    #
    #   3. Creates one publisher:
    #          /drive             -> AckermannDriveStamped
    #      Hint: self.create_publisher(MsgType, topic, queue_size)
    #
    #   4. Creates a timer at `publish_rate` Hz that calls a method which
    #      invokes self.safety_filter(...) and publishes the result.
    #      Hint: self.create_timer(period_sec, callback)
    #
    # For reference, open any other node in this repo (e.g.
    # FinalProject/scripts/joy_to_ackermann_node.py) to see the same pattern.
    # =========================================================================

    def __init__(self):
        super().__init__('safety_filter_node')

        # ---- TODO(Task 1.1): declare ROS parameters ----
        self.declare_parameter("ilqr_config", str(DEFAULT_CONFIG_PATH))
        config_path = self.get_parameter("ilqr_config").value
        config = load_config(config_path)


        # ---- TODO(Task 1.2): read parameter values ----
        self.config = self._declare_and_read_config(config)
        self._latest_teleop: Optional[AckermannDriveStamped] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_obs: Optional[MarkerArray] = None
        self._latest_path: Optional[Path] = None
        self._teleop_time: Optional[float] = None
        self._odom_time: Optional[float] = None
        self._obstacle_time: Optional[float] = None
        self._path_time: Optional[float] = None
        self._last_log_time = 0.0
        self._last_steering = 0.0

        # Use two independent ILQR planners: one scores the current human command,
        # and one searches for the best safe override when the human command is bad.
        # This node keeps the final safety filter simple: it scores short bicycle
        # rollouts directly instead of running a second optimizer in the callback.

        # ---- TODO(Task 1.3): create subscribers ----
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

        # ---- TODO(Task 1.4): create the publisher ----
        self.pub = self.create_publisher(AckermannDriveStamped, self.config.drive_topic, 1)

        # ---- TODO(Task 1.5): create a timer at publish_rate Hz ----
        self.create_timer(1.0 / max(self.config.publish_rate, 1e-3), self._publish_filtered)

        # self.get_logger().info(
        #     f"safety_filter_node ready: {teleop_topic} + {odom_topic} "
        #     f"+ {obs_topic} -> {drive_topic}"
        # )
        self.get_logger().info(
            "ilqr safety_filter_node ready: "
            f"{self.config.teleop_topic} + {self.config.odom_topic} + "
            f"{self.config.static_obs_topic} + {self.config.routing_path_topic} "
            f"-> {self.config.drive_topic}"
        )

    def _declare_and_read_config(self, config: IlqrQpConfig) -> IlqrQpConfig:
        values = {}
        for field in fields(IlqrQpConfig):
            default = getattr(config, field.name)
            self.declare_parameter(
                field.name,
                list(default) if isinstance(default, tuple) else default,
            )
            value = self.get_parameter(field.name).value
            if isinstance(default, tuple) and isinstance(value, list):
                value = tuple(value)
            values[field.name] = value
        return replace(config, **values)

    # ---- TODO(Task 1.6): implement callbacks ----

    def _teleop_cb(self, msg: AckermannDriveStamped):
        self._latest_teleop = msg
        self._teleop_time = self._now_sec()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_time = self._now_sec()

    def _obstacles_cb(self, msg: MarkerArray):
        self._latest_obs = msg
        self._obstacle_time = self._now_sec()

    def _path_cb(self, msg: Path):
        self._latest_path = msg
        self._path_time = self._now_sec()

    # ---- TODO(Task 1.7): the timer callback ----
    # Should:
    #   - return early if no teleop has arrived yet (self._latest_teleop is None)
    #   - call self.safety_filter(teleop=..., odom=..., obstacles=...)
    #   - if the returned command is not None, update its header.stamp to now
    #     and publish it on /drive
    #
    # def _publish_filtered(self):
    #
    def _publish_filtered(self):
        if self._latest_teleop is None:
            return

        command = self.safety_filter(
            teleop=self._latest_teleop,
            odom=self._latest_odom,
            obstacles=self._latest_obs,
        )
        if command is None:
            return

        command.header.stamp = self.get_clock().now().to_msg()
        self._last_steering = float(command.drive.steering_angle)
        self.pub.publish(command)

    # =========================================================================
    # TASK 2 — Safety filter implementation
    # =========================================================================
    #
    # Start as a passthrough, then add real safety logic.
    #
    # You are free to pick any approach (or your own). Add helper methods,
    # extra parameters, even a sub-folder of modules — this skeleton will
    # get out of the way.
    # =========================================================================

    def safety_filter(self, teleop, odom, obstacles):
        """
        Args
        ----
        teleop : ackermann_msgs.msg.AckermannDriveStamped   (or None)
            Human's desired command. Useful fields:
                teleop.drive.speed           float   m/s     target forward speed
                teleop.drive.steering_angle  float   rad     target steering angle
                teleop.drive.acceleration    float   m/s^2   usually 0 from the joy
                teleop.header.stamp          Time            when the command was issued

        odom : nav_msgs.msg.Odometry                        (or None)
            Vehicle state. Useful fields:
                odom.pose.pose.position.x       float   m       map-frame x
                odom.pose.pose.position.y       float   m       map-frame y
                odom.pose.pose.position.z       float   m       usually 0
                odom.pose.pose.orientation      Quaternion (.x .y .z .w)
                    → use yaw_from_quat(..) above to get heading in rad
                odom.twist.twist.linear.x       float   m/s     forward velocity
                odom.twist.twist.angular.z      float   rad/s   yaw rate

        obstacles : visualization_msgs.msg.MarkerArray      (or None)
            Static obstacles (cubes). Useful fields:
                obstacles.markers               list[Marker]
                for m in obstacles.markers:
                    m.pose.position.x / .y / .z float   m       obstacle center
                    m.scale.x / .y / .z         float   m       cube size (x=y=z typically)
                    m.id                        int             obstacle id
                    m.ns                        str             namespace

        Returns
        -------
        ackermann_msgs.msg.AckermannDriveStamped
            The command to publish on /drive. Set `.drive.speed` (m/s) and
            `.drive.steering_angle` (rad). Header.stamp is overwritten for you.
            Return None to skip publishing this tick.
        """
        # ---- TODO(Task 2): replace this passthrough ----
        if teleop is None:
            return None

        now = self._now_sec()
        human_speed = self._clip_speed(teleop.drive.speed)
        human_steering = self._clip_steering(teleop.drive.steering_angle)

        if self._is_stale(self._teleop_time, self.config.teleop_timeout_sec):
            self._log_periodic(now, "stale teleop; publishing stop")
            return self._make_command(teleop, 0.0, 0.0)

        if odom is None or self._is_stale(self._odom_time, self.config.odom_timeout_sec):
            self._log_periodic(now, "missing/stale odom; publishing failsafe")
            return self._make_command(
                teleop,
                self.config.stale_odom_speed,
                self.config.stale_odom_steering,
            )

        obstacle_msg = (
            None
            if self._is_stale(self._obstacle_time, self.config.obstacle_timeout_sec)
            else obstacles
        )
        path_msg = (
            None
            if self._is_stale(self._path_time, self.config.path_timeout_sec)
            else self._latest_path
        )

        state = state_from_odom(odom, self._last_steering)
        obstacle_list = obstacles_from_msg(
            obstacle_msg,
            self.config.obstacle_radius_buffer,
        )
        path = path_from_msg(path_msg)
        if len(path) < 2 and self.config.fallback_path_enabled:
            path = fallback_path_ahead(
                state=state,
                length=self.config.fallback_path_length,
                width=self.config.fallback_path_width,
                speed_limit=max(human_speed, self.config.min_projection_speed),
                samples=self.config.fallback_path_samples,
            )

        if self.config.require_path_for_lane_filter and len(path) < 2:
            self._log_periodic(now, "missing path; publishing stop")
            return self._make_command(teleop, self.config.no_solution_speed, human_steering)

        human_rollout = self._rollout(state, human_speed, human_steering)
        lane_buffer, obstacle_buffer = self._safety_buffers(human_speed)
        human_lane_margin = (
            min_lane_margin(human_rollout, path, lane_buffer) if path else None
        )
        human_obstacle_margin = min_obstacle_margin(
            human_rollout,
            obstacle_list,
            obstacle_buffer,
        )

        # A direct obstacle check reacts faster than waiting for the full rollout score.
        forward_margin = forward_obstacle_margin(
            state=VehicleState(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                speed=max(state.speed, human_speed),
                steering_angle=state.steering_angle,
            ),
            obstacles=obstacle_list,
            safety_buffer=obstacle_buffer,
            forward_width=self.config.forward_obstacle_width,
            max_distance=self.config.forward_obstacle_distance,
        )
        ttc = (
            None
            if forward_margin is None or max(state.speed, human_speed) <= 1e-3
            else max(0.0, forward_margin) / max(state.speed, human_speed)
        )
        if (
            forward_margin is not None
            and forward_margin <= self.config.emergency_brake_margin
        ) or (ttc is not None and ttc <= self.config.ttc_hard_sec):
            self._log_periodic(now, "obstacle ahead; publishing stop")
            return self._make_command(teleop, 0.0, human_steering)

        if self._margins_are_safe(
            human_lane_margin,
            human_obstacle_margin,
            self.config.soft_margin,
        ):
            return self._make_command(teleop, human_speed, human_steering)

        lane_command = self._lane_recovery_command(
            state=state,
            path=path,
            human_speed=human_speed,
            human_steering=human_steering,
            lane_margin_value=human_lane_margin,
        )
        if lane_command is not None:
            speed, steering = lane_command
            rollout = self._rollout(state, speed, steering)
            lane = min_lane_margin(rollout, path, lane_buffer) if path else None
            obstacle = min_obstacle_margin(rollout, obstacle_list, obstacle_buffer)
            if self._margins_are_safe(lane, obstacle, self.config.hard_margin):
                self._log_periodic(now, "lane recovery override")
                return self._make_command(teleop, speed, steering)

        best = self._best_safe_candidate(
            state=state,
            path=path,
            obstacles=obstacle_list,
            human_speed=human_speed,
            human_steering=human_steering,
            lane_buffer=lane_buffer,
            obstacle_buffer=obstacle_buffer,
        )
        if best is not None:
            speed, steering, reason = best
            self._log_periodic(now, f"{reason} override")
            return self._make_command(teleop, speed, steering)

        self._log_periodic(now, "no safe candidate; publishing stop")
        steering = human_steering if self.config.no_solution_keep_steering else 0.0
        return self._make_command(teleop, self.config.no_solution_speed, steering)

    def _rollout(self, state: VehicleState, speed: float, steering: float) -> List[VehicleState]:
        return rollout_constant_command(
            state=state,
            speed_cmd=speed,
            steering_cmd=steering,
            wheelbase=self.config.wheelbase,
            dt=self.config.dt,
            horizon_sec=self.config.horizon_sec,
            response_delay_sec=self.config.response_delay_sec,
            min_projection_speed=self.config.min_projection_speed,
        )

    def _safety_buffers(self, speed: float) -> Tuple[float, float]:
        stopping = speed * speed / (2.0 * max(self.config.max_decel, 1e-3))
        lane_buffer = (
            self.config.vehicle_radius
            + self.config.lane_margin
            + self.config.localization_buffer
        )
        obstacle_buffer = (
            self.config.vehicle_radius
            + self.config.localization_buffer
            + self.config.stopping_buffer
            + 0.25 * stopping
        )
        return lane_buffer, obstacle_buffer

    def _margins_are_safe(
        self,
        lane_margin_value: Optional[float],
        obstacle_margin_value: Optional[float],
        threshold: float,
    ) -> bool:
        if lane_margin_value is not None and lane_margin_value < threshold:
            return False
        if obstacle_margin_value is not None and obstacle_margin_value < threshold:
            return False
        return True

    def _lane_recovery_command(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
        human_speed: float,
        human_steering: float,
        lane_margin_value: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        if (
            len(path) < 2
            or lane_margin_value is None
            or lane_margin_value >= self.config.soft_margin
        ):
            return None

        projection = closest_path_projection(state, path)
        if projection is None:
            return None

        _, lateral, tangent_yaw = projection
        heading_error = wrap_angle(state.yaw - tangent_yaw)
        steering = -(
            self.config.path_lateral_gain * lateral
            + self.config.path_heading_gain * heading_error
        )
        speed_scale = self.config.path_corner_speed_scale
        return (
            self._clip_speed(human_speed * speed_scale),
            self._clip_steering(
                steering if abs(steering) > abs(human_steering) else human_steering
            ),
        )

    def _best_safe_candidate(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
        human_speed: float,
        human_steering: float,
        lane_buffer: float,
        obstacle_buffer: float,
    ) -> Optional[Tuple[float, float, str]]:
        best = None
        best_score = float("inf")
        for scale in self.config.speed_scales:
            for offset in self.config.steering_offsets:
                speed = self._clip_speed(human_speed * float(scale))
                steering = self._clip_steering(human_steering + float(offset))
                rollout = self._rollout(state, speed, steering)
                lane = min_lane_margin(rollout, path, lane_buffer) if path else None
                obstacle = min_obstacle_margin(rollout, obstacles, obstacle_buffer)
                if not self._margins_are_safe(lane, obstacle, self.config.hard_margin):
                    continue

                # Prefer commands close to the driver unless a margin is tight.
                tightest = min(
                    value
                    for value in (lane, obstacle)
                    if value is not None
                ) if lane is not None or obstacle is not None else self.config.soft_margin
                safety_reward = max(0.0, self.config.soft_margin - tightest)
                score = (
                    self.config.speed_weight * abs(human_speed - speed)
                    + self.config.steering_weight * abs(human_steering - steering)
                    + self.config.obstacle_violation_weight * safety_reward
                )
                if score < best_score:
                    best_score = score
                    best = (speed, steering, "candidate")
        return best



    def _make_command(self, source, speed, steering):
        command = copy.deepcopy(source)
        command.drive.speed = self._clip_speed(speed)
        command.drive.steering_angle = self._clip_steering(steering)
        return command

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return max(lower, min(self.config.max_speed, float(speed)))

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return max(-limit, min(limit, float(steering)))

    def _is_stale(self, stamp_sec: Optional[float], timeout_sec: float) -> bool:
        if stamp_sec is None:
            return True
        return self._now_sec() - stamp_sec > timeout_sec

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_periodic(self, now: float, message: str):
        if now - self._last_log_time >= self.config.log_period_sec:
            self.get_logger().info(message)
            self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



    # ===========================
    # Test any other safety filter here
    # ===========================

    # ILQR Safety Filter
    # ilqr_main(args=args)

    # # CBF-QP Safety Filter
    # cbf_qp_main(args=args)

    # CBF-Heuristic Safety Filter
    # cbf_heuristic_main(args=args)


if __name__ == '__main__':
    main()


# implement a simple heuristic based CBF safety filter
# such that it steers to center of the lane when the vehicle is going to going to go off the lane (with adjusted speed based on the distance to the lane)

# stop if obstcale is directly in front of the vehicle
# if dynamic obstacles are in the way, stop 

# simple implementation
