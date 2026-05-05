from typing import List

import numpy as np

from .config import SafetyFilterParams
from .obstacle_memory import Obstacle


def odom_to_state(msg, delta_estimate: float, params: SafetyFilterParams) -> np.ndarray:
    q = msg.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array(
        [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            np.clip(msg.twist.twist.linear.x, params.v_min, params.v_max),
            yaw,
            np.clip(delta_estimate, params.delta_min, params.delta_max),
        ],
        dtype=float,
    )


def ackermann_msg_to_control(msg, state: np.ndarray, params: SafetyFilterParams) -> np.ndarray:
    desired_speed = np.clip(float(msg.drive.speed), params.v_min, params.v_max)
    desired_delta = np.clip(float(msg.drive.steering_angle), params.delta_min, params.delta_max)
    accel = (desired_speed - state[2]) / params.dt
    omega = (desired_delta - state[4]) / params.dt
    return np.array([accel, omega], dtype=float)


def control_to_ackermann_msg(msg_type, u: np.ndarray, state: np.ndarray, params: SafetyFilterParams, stamp):
    msg = msg_type()
    msg.header.stamp = stamp
    msg.drive.speed = float(np.clip(state[2] + u[0] * params.dt, params.v_min, params.v_max))
    msg.drive.steering_angle = float(np.clip(state[4] + u[1] * params.dt, params.delta_min, params.delta_max))
    return msg


def marker_array_to_obstacles(msg, default_radius: float) -> List[Obstacle]:
    obstacles = []
    for marker in msg.markers:
        if getattr(marker, "action", 0) == 2:
            continue
        scale_radius = 0.5 * max(float(marker.scale.x), float(marker.scale.y), default_radius * 2.0)
        obstacles.append(
            Obstacle(
                tag_id=int(marker.id),
                position=np.array([marker.pose.position.x, marker.pose.position.y], dtype=float),
                radius=scale_radius,
            )
        )
    return obstacles


def odometry_array_to_obstacles(msg, radius: float) -> List[Obstacle]:
    obstacles = []
    for idx, odom in enumerate(msg.odometry_array):
        obstacles.append(
            Obstacle(
                tag_id=idx,
                position=np.array([odom.pose.pose.position.x, odom.pose.pose.position.y], dtype=float),
                radius=radius,
            )
        )
    return obstacles
