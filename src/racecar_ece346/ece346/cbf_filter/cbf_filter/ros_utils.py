"""
Message conversion utilities — keep all ROS serialization/deserialization here.
None of the other library modules should import ROS types.

ServoMsg field meanings (as used by the ECE346 simulator):
  throttle  — maps directly to acceleration (m/s²) in the bicycle model
  steer     — maps directly to steering angle (rad) in the bicycle model
  reverse   — unused by the filter (set to False)

This means:
  - control_to_servo_msg stores the NEW steering angle (delta + omega*dt) as steer,
    NOT the rate omega.  The simulator's Bicycle4D reads steer as delta directly.
  - servo_msg_to_control reads steer as the desired delta and back-calculates
    the required omega = (desired_delta - current_delta) / dt.
"""
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import CbfParams
from .obstacle_memory import Obstacle


def odom_to_state(msg, delta_estimate: float, p: CbfParams) -> np.ndarray:
    """nav_msgs/Odometry  →  [px, py, v, psi, delta]."""
    quat = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    ]
    psi = R.from_quat(quat).as_euler("xyz", degrees=False)[-1]
    return np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        np.clip(msg.twist.twist.linear.x, p.v_min, p.v_max),
        psi,
        np.clip(delta_estimate, p.delta_min, p.delta_max),
    ], dtype=float)


def servo_msg_to_control(msg, state: np.ndarray, p: CbfParams) -> np.ndarray:
    """ServoMsg  →  [a, omega] in bicycle-model units."""
    desired_delta = np.clip(float(msg.steer), p.delta_min, p.delta_max)
    omega = (desired_delta - state[4]) / p.dt
    return np.array([float(msg.throttle), omega], dtype=float)


def control_to_servo_msg(msg_type, u: np.ndarray, state: np.ndarray, p: CbfParams, stamp):
    """[a, omega]  →  ServoMsg."""
    msg = msg_type()
    msg.header.stamp = stamp
    throttle = float(np.clip(u[0], p.a_min, p.a_max))
    # Don't command reverse braking when the truck is already stopped.
    if state[2] <= 0.02 and throttle < 0.0:
        throttle = 0.0
    msg.throttle = throttle
    # Convert omega back to the new steering angle for this timestep.
    msg.steer = float(np.clip(state[4] + u[1] * p.dt, p.delta_min, p.delta_max))
    msg.reverse = False
    return msg


def marker_array_to_obstacles(msg, default_radius: float) -> List[Obstacle]:
    """visualization_msgs/MarkerArray  →  list of Obstacles."""
    result = []
    for marker in msg.markers:
        r = 0.5 * max(float(marker.scale.x), float(marker.scale.y), default_radius * 2.0)
        result.append(Obstacle(
            tag_id=int(marker.id),
            position=np.array([marker.pose.position.x, marker.pose.position.y]),
            radius=r,
        ))
    return result


def odometry_array_to_obstacles(msg, radius: float) -> List[Obstacle]:
    """racecar_msgs/OdometryArray  →  list of Obstacles."""
    result = []
    for idx, odom in enumerate(msg.odometry_array):
        result.append(Obstacle(
            tag_id=idx,
            position=np.array([odom.pose.pose.position.x, odom.pose.pose.position.y]),
            radius=radius,
        ))
    return result
