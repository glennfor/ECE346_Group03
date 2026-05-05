"""Geometry, rollout, and ROS message conversion helpers for ILQR-QP."""

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class Margins:
    lane: Optional[float]
    obstacle: Optional[float]

    @property
    def minimum(self) -> Optional[float]:
        values = [value for value in (self.lane, self.obstacle) if value is not None]
        return min(values) if values else None

    def is_safe(self, threshold: float = 0.0) -> bool:
        return self.minimum is None or self.minimum >= threshold


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def state_from_odom(odom, steering_angle: float = 0.0) -> VehicleState:
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


def path_from_msg(path_msg) -> List[PathPoint]:
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


def obstacles_from_msg(marker_array, radius_buffer: float) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    if marker_array is None:
        return obstacles
    for marker in marker_array.markers:
        if getattr(marker, "action", 0) == 2:
            continue
        sx = max(0.0, float(marker.scale.x))
        sy = max(0.0, float(marker.scale.y))
        radius = 0.5 * math.hypot(sx, sy) + radius_buffer
        obstacles.append(
            Obstacle(
                x=float(marker.pose.position.x),
                y=float(marker.pose.position.y),
                radius=radius,
            )
        )
    return obstacles


def step_state(
    state: VehicleState,
    speed_cmd: float,
    steering_cmd: float,
    wheelbase: float,
    dt: float,
) -> VehicleState:
    speed = max(0.0, float(speed_cmd))
    steering = float(steering_cmd)
    yaw_rate = speed * math.tan(steering) / max(wheelbase, 1e-3)
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
    min_projection_speed: float = 0.0,
) -> List[VehicleState]:
    steps = max(1, int(math.ceil(horizon_sec / dt)))
    delayed_steps = max(0, int(math.ceil(response_delay_sec / dt)))
    states: List[VehicleState] = [state]
    current = state
    for step in range(steps):
        speed = current.speed if step < delayed_steps else speed_cmd
        if speed_cmd > 0.0:
            speed = max(speed, min_projection_speed)
        current = step_state(
            state=current,
            speed_cmd=speed,
            steering_cmd=steering_cmd,
            wheelbase=wheelbase,
            dt=dt,
        )
        states.append(current)
    return states


def rollout_command_sequence(
    state: VehicleState,
    commands: Sequence[Tuple[float, float]],
    wheelbase: float,
    dt: float,
) -> List[VehicleState]:
    states: List[VehicleState] = [state]
    current = state
    for speed, steering in commands:
        current = step_state(current, speed, steering, wheelbase, dt)
        states.append(current)
    return states


def closest_path_projection(
    x: float,
    y: float,
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
        t = clamp(((x - p0.x) * vx + (y - p0.y) * vy) / seg_len_sq, 0.0, 1.0)
        proj_x = p0.x + t * vx
        proj_y = p0.y + t * vy
        dx = x - proj_x
        dy = y - proj_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq:
            left_width = (1.0 - t) * p0.left_width + t * p1.left_width
            right_width = (1.0 - t) * p0.right_width + t * p1.right_width
            speed_limit = (1.0 - t) * p0.speed_limit + t * p1.speed_limit
            tangent_yaw = math.atan2(vy, vx)
            signed_lateral = -math.sin(tangent_yaw) * dx + math.cos(tangent_yaw) * dy
            best = (
                PathPoint(proj_x, proj_y, left_width, right_width, speed_limit),
                signed_lateral,
                tangent_yaw,
            )
            best_dist_sq = dist_sq
    return best


def lane_margin(
    state: VehicleState,
    path: Sequence[PathPoint],
    safety_buffer: float,
) -> Optional[float]:
    projection = closest_path_projection(state.x, state.y, path)
    if projection is None:
        return None
    point, lateral, _ = projection
    left_clearance = point.left_width - lateral
    right_clearance = point.right_width + lateral
    return min(left_clearance, right_clearance) - safety_buffer


def heading_error(state: VehicleState, path: Sequence[PathPoint]) -> float:
    projection = closest_path_projection(state.x, state.y, path)
    if projection is None:
        return 0.0
    _, _, tangent_yaw = projection
    return wrap_angle(state.yaw - tangent_yaw)


def min_lane_margin(
    states: Iterable[VehicleState],
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
    states: Iterable[VehicleState],
    obstacles: Sequence[Obstacle],
    safety_buffer: float,
) -> Optional[float]:
    if not obstacles:
        return None
    best_margin = float("inf")
    for state in states:
        for obstacle in obstacles:
            center_dist = math.hypot(state.x - obstacle.x, state.y - obstacle.y)
            margin = center_dist - obstacle.radius - safety_buffer
            best_margin = min(best_margin, margin)
    return best_margin


def trajectory_margins(
    states: Sequence[VehicleState],
    path: Sequence[PathPoint],
    obstacles: Sequence[Obstacle],
    lane_buffer: float,
    obstacle_buffer: float,
    require_path: bool,
) -> Margins:
    lane = min_lane_margin(states, path, lane_buffer) if path else None
    if require_path and lane is None:
        lane = -lane_buffer
    return Margins(
        lane=lane,
        obstacle=min_obstacle_margin(states, obstacles, obstacle_buffer),
    )
