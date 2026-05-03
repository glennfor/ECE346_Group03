from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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
) -> float:
    obs_list = list(obstacles)
    if not obs_list:
        return 100.0

    safety_margin = params.r_safe_obs if safety_margin is None else safety_margin
    values = []
    for point in footprint_points(x, params):
        p = np.asarray(point[:2], dtype=float)
        values.extend(
            float(np.linalg.norm(p - obs.position[:2]) - params.truck_radius_m - obs.radius - safety_margin)
            for obs in obs_list
        )
    return min(values)


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
        "obstacle": margin_obstacle(x, ctx.obstacles, ctx.params),
        "traffic": margin_obstacle(x, ctx.traffic, ctx.params, ctx.params.r_safe_traf),
        "kinematic": margin_kinematic(x, ctx.params),
    }


def margin_grad(x: np.ndarray, ctx: MarginContext, eps: float = 1e-4) -> np.ndarray:
    """
    Forward finite-difference gradient of margin_total w.r.t. the 5-dim state.

    This perturbs one state at a time and re-evaluates the margin at that single
    state — NOT the whole-rollout minimum — so the function is smooth and the
    finite difference is reliable.  This is the ∇m(x_k) term in the paper's
    sensitivity formula  ∇h(x_0) = Q(t*)ᵀ ∇m(x_{t*}).
    """
    m0, _ = margin_total(x, ctx)
    grad = np.zeros(5, dtype=float)
    for i in range(5):
        xp = np.asarray(x, dtype=float).copy()
        xp[i] += eps
        mp, _ = margin_total(xp, ctx)
        grad[i] = (mp - m0) / eps
    return grad

