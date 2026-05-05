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


@dataclass(frozen=True)
class PathProjection:
    point: PathPoint
    lateral: float
    tangent_yaw: float
    segment_index: int
    segment_t: float


@dataclass(frozen=True)
class PathTrackingError:
    mean_abs_lateral: float
    max_abs_lateral: float
    mean_abs_heading: float
    max_abs_heading: float
    end_abs_heading: float
    missing_path: bool = False


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


def fallback_path_ahead(
    state: VehicleState,
    length: float,
    width: float,
    speed_limit: float,
    samples: int = 8,
) -> List[PathPoint]:
    steps = max(2, int(samples))
    half_width = max(0.05, 0.5 * float(width))
    path: List[PathPoint] = []
    for idx in range(steps):
        distance = float(length) * idx / max(1, steps - 1)
        path.append(
            PathPoint(
                x=state.x + distance * math.cos(state.yaw),
                y=state.y + distance * math.sin(state.yaw),
                left_width=half_width,
                right_width=half_width,
                speed_limit=max(0.0, float(speed_limit)),
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
    projection = closest_path_projection_full(x, y, path)
    if projection is None:
        return None
    return projection.point, projection.lateral, projection.tangent_yaw


def closest_path_projection_full(
    x: float,
    y: float,
    path: Sequence[PathPoint],
) -> Optional[PathProjection]:
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
        is_better = dist_sq < best_dist_sq - 1e-9
        is_forward_tie = (
            best is not None
            and abs(dist_sq - best_dist_sq) <= 1e-9
            and idx > best.segment_index
        )
        if is_better or is_forward_tie:
            left_width = (1.0 - t) * p0.left_width + t * p1.left_width
            right_width = (1.0 - t) * p0.right_width + t * p1.right_width
            speed_limit = (1.0 - t) * p0.speed_limit + t * p1.speed_limit
            tangent_yaw = math.atan2(vy, vx)
            signed_lateral = -math.sin(tangent_yaw) * dx + math.cos(tangent_yaw) * dy
            best = PathProjection(
                point=PathPoint(proj_x, proj_y, left_width, right_width, speed_limit),
                lateral=signed_lateral,
                tangent_yaw=tangent_yaw,
                segment_index=idx,
                segment_t=t,
            )
            best_dist_sq = dist_sq
    return best


def path_lateral_error(
    state: VehicleState,
    path: Sequence[PathPoint],
) -> Optional[float]:
    projection = closest_path_projection_full(state.x, state.y, path)
    return None if projection is None else projection.lateral


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
    projection = closest_path_projection_full(state.x, state.y, path)
    if projection is None:
        return 0.0
    return wrap_angle(state.yaw - projection.tangent_yaw)


def trajectory_path_tracking_error(
    states: Sequence[VehicleState],
    path: Sequence[PathPoint],
) -> PathTrackingError:
    if len(path) < 2 or not states:
        return PathTrackingError(0.0, 0.0, 0.0, 0.0, 0.0, missing_path=True)

    lateral_errors: List[float] = []
    heading_errors: List[float] = []
    for state in states:
        projection = closest_path_projection_full(state.x, state.y, path)
        if projection is None:
            continue
        lateral_errors.append(abs(projection.lateral))
        heading_errors.append(abs(wrap_angle(state.yaw - projection.tangent_yaw)))

    if not lateral_errors or not heading_errors:
        return PathTrackingError(0.0, 0.0, 0.0, 0.0, 0.0, missing_path=True)

    return PathTrackingError(
        mean_abs_lateral=sum(lateral_errors) / len(lateral_errors),
        max_abs_lateral=max(lateral_errors),
        mean_abs_heading=sum(heading_errors) / len(heading_errors),
        max_abs_heading=max(heading_errors),
        end_abs_heading=heading_errors[-1],
    )


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


def forward_obstacle_margin(
    state: VehicleState,
    obstacles: Sequence[Obstacle],
    safety_buffer: float,
    forward_width: float,
    max_distance: float,
) -> Optional[float]:
    if not obstacles:
        return None

    best_margin = float("inf")
    cos_yaw = math.cos(state.yaw)
    sin_yaw = math.sin(state.yaw)
    for obstacle in obstacles:
        dx = obstacle.x - state.x
        dy = obstacle.y - state.y
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        lateral_limit = forward_width + obstacle.radius + safety_buffer
        if 0.0 <= forward <= max_distance and abs(lateral) <= lateral_limit:
            margin = forward - obstacle.radius - safety_buffer
            best_margin = min(best_margin, margin)

    return None if best_margin == float("inf") else best_margin


def obstacle_time_to_collision(
    state: VehicleState,
    obstacles: Sequence[Obstacle],
    safety_buffer: float,
    forward_width: float,
    max_distance: float,
    min_speed: float = 1e-3,
) -> Optional[float]:
    margin = forward_obstacle_margin(
        state=state,
        obstacles=obstacles,
        safety_buffer=safety_buffer,
        forward_width=forward_width,
        max_distance=max_distance,
    )
    if margin is None or state.speed < min_speed:
        return None
    return max(0.0, margin) / max(state.speed, min_speed)


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
