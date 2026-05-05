from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .config import SafetyFilterParams
from .lane_context import LaneContext
from .obstacle_memory import Obstacle


@dataclass
class MarginContext:
    lane: LaneContext
    obstacles: List[Obstacle]
    traffic: List[Obstacle]
    params: SafetyFilterParams


def footprint_points(x: np.ndarray, params: SafetyFilterParams) -> List[np.ndarray]:
    px, py, _, psi, _ = x
    heading = np.array([np.cos(psi), np.sin(psi)], dtype=float)
    offsets = np.array([-0.35, 0.0, 0.35]) * params.truck_length_m
    base = np.array([px, py], dtype=float)
    return [base + offset * heading for offset in offsets]


def margin_lane(x: np.ndarray, lane: LaneContext, params: SafetyFilterParams) -> float:
    if lane.is_fallback:
        return 100.0

    margins = []
    for point in footprint_points(x, params):
        sample = lane.query(float(point[0]), float(point[1]))
        left_margin = sample.width_left - sample.signed_lateral_error
        right_margin = sample.width_right + sample.signed_lateral_error
        margins.append(min(left_margin, right_margin) - params.truck_radius_m - params.r_safe_lane)
    return float(min(margins))


def margin_obstacle(
    x: np.ndarray,
    obstacles: Iterable[Obstacle],
    params: SafetyFilterParams,
    safety_margin: float = None,
    lane: LaneContext = None,
) -> float:
    obs_list = list(obstacles)
    if not obs_list:
        return 100.0

    safety_margin = params.r_safe_obs if safety_margin is None else safety_margin
    values = []
    filtered_obstacles = []
    for obs in obs_list:
        if lane is not None and not lane.is_fallback:
            sample = lane.query(float(obs.position[0]), float(obs.position[1]))
            outside_left = sample.signed_lateral_error > sample.width_left + obs.radius
            outside_right = -sample.signed_lateral_error > sample.width_right + obs.radius
            if outside_left or outside_right:
                continue
        filtered_obstacles.append(obs)

    if not filtered_obstacles:
        return 100.0

    for point in footprint_points(x, params):
        p = np.asarray(point[:2], dtype=float)
        values.extend(
            float(np.linalg.norm(p - obs.position[:2]) - params.truck_radius_m - obs.radius - safety_margin)
            for obs in filtered_obstacles
        )
    return min(values)


def margin_forward_obstacle(
    x: np.ndarray,
    obstacles: Iterable[Obstacle],
    params: SafetyFilterParams,
) -> float:
    obs_list = list(obstacles)
    if not obs_list:
        return 100.0

    px, py, v, psi, _ = x
    position = np.array([px, py], dtype=float)
    forward = np.array([np.cos(psi), np.sin(psi)], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    corridor_half_width = 0.5 * params.forward_obstacle_width_m + params.truck_radius_m
    lookahead = params.forward_obstacle_distance_m + max(0.0, v) * 0.5

    margins = []
    for obs in obs_list:
        relative = np.asarray(obs.position[:2], dtype=float) - position
        longitudinal = float(relative @ forward)
        lateral = abs(float(relative @ left))
        lateral_clearance = lateral - corridor_half_width - obs.radius
        if longitudinal < -obs.radius or longitudinal > lookahead + obs.radius:
            continue
        if lateral_clearance > 0.0:
            continue
        margins.append(longitudinal - params.truck_radius_m - obs.radius)

    return min(margins) if margins else 100.0


def margin_kinematic(x: np.ndarray, params: SafetyFilterParams) -> float:
    _, _, v, _, delta = x
    values = [
        delta - params.delta_min - params.r_safe_kin,
        params.delta_max - delta - params.r_safe_kin,
        params.v_max - v - params.r_safe_kin,
    ]
    return float(min(values))


def margin_total(x: np.ndarray, ctx: MarginContext) -> Tuple[float, str]:
    components = margin_components(x, ctx)
    label = min(components, key=components.get)
    return float(components[label]), label


def margin_components(x: np.ndarray, ctx: MarginContext) -> dict:
    return {
        "lane": margin_lane(x, ctx.lane, ctx.params),
        "obstacle": margin_obstacle(x, ctx.obstacles, ctx.params, lane=ctx.lane),
        "forward_obstacle": margin_forward_obstacle(x, ctx.obstacles, ctx.params),
        "traffic": margin_obstacle(x, ctx.traffic, ctx.params, ctx.params.r_safe_traf, ctx.lane),
        "kinematic": margin_kinematic(x, ctx.params),
    }
