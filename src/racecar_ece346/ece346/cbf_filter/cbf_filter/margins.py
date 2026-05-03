"""
Safety margin functions — ℓ(x).

Each function returns a signed distance:
  > 0  →  truck is safe in that respect (how much room we have)
  = 0  →  exactly at the boundary
  < 0  →  already violated

margin_total returns the MINIMUM over all components. The implicit barrier
h_imp(x) = min over the backup trajectory of margin_total.

Lane margin
-----------
We approximate the truck footprint as 3 circles spaced along the body
axis (front, center, rear). Each circle has radius truck_radius_m.
The lane margin at a state is the minimum signed distance from any of
those circles to the nearest lane boundary, minus a safety buffer r_safe_lane.

Obstacle margin
---------------
For each obstacle, we compute: distance(footprint_point, obstacle_center)
minus (truck_radius + obstacle_radius + r_safe_obs).
We take the minimum over all footprint points and all obstacles.

Kinematic margin
----------------
A soft barrier that keeps us inside the steering-angle and speed envelopes.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import CbfParams
from .lane_context import LaneContext
from .obstacle_memory import Obstacle


@dataclass
class MarginContext:
    """All the context needed to evaluate margins at a state."""
    lane: LaneContext
    obstacles: List[Obstacle]   # static obstacles
    traffic: List[Obstacle]     # dynamic traffic vehicles
    params: CbfParams


def _footprint_points(x: np.ndarray, p: CbfParams) -> List[np.ndarray]:
    """3 circles spanning the truck body: rear bumper, mid-wheelbase, front bumper.

    State [px, py] is the rear-axle position.  Assuming equal front/rear overhang:
      overhang = (truck_length - wheelbase) / 2
      rear bumper  = -overhang from rear axle
      front bumper = wheelbase + overhang from rear axle
    """
    px, py, _, psi, _ = x
    heading = np.array([np.cos(psi), np.sin(psi)])
    overhang = (p.truck_length_m - p.wheelbase_m) / 2.0
    offsets = np.array([-overhang, p.wheelbase_m / 2.0, p.wheelbase_m + overhang])
    base = np.array([px, py])
    return [base + off * heading for off in offsets]


def margin_lane(x: np.ndarray, lane: LaneContext, p: CbfParams) -> float:
    """
    Signed distance from the truck footprint to the lane boundary.
    Positive = inside lane with clearance.  Negative = outside lane.
    """
    values = []
    for pt in _footprint_points(x, p):
        s = lane.query(float(pt[0]), float(pt[1]))
        # left_margin:  how far the left side of the circle is from the left boundary
        # right_margin: how far the right side is from the right boundary
        # signed_lateral_error > 0 means truck is to the left, so left margin shrinks
        left_m = s.width_left - s.signed_lateral_error - p.truck_radius_m - p.r_safe_lane
        right_m = s.width_right + s.signed_lateral_error - p.truck_radius_m - p.r_safe_lane
        values.append(min(left_m, right_m))
    return float(min(values))


def margin_obstacle(
    x: np.ndarray,
    obstacles: List[Obstacle],
    p: CbfParams,
    r_safe: Optional[float] = None,
) -> float:
    """
    Minimum signed distance from the truck footprint to any obstacle.
    Returns 100.0 when there are no obstacles (so it never constrains the QP).
    """
    if not obstacles:
        return 100.0

    r_safe = p.r_safe_obs if r_safe is None else r_safe
    values = []
    for pt in _footprint_points(x, p):
        p2 = np.asarray(pt[:2])
        for obs in obstacles:
            d = float(np.linalg.norm(p2 - obs.position[:2]))
            values.append(d - p.truck_radius_m - obs.radius - r_safe)
    return float(min(values))


def margin_kinematic(x: np.ndarray, p: CbfParams) -> float:
    """
    How far inside the steering-angle and speed upper-bound envelopes we are.
    Note: v_min=0 is NOT included here — a stopped truck is safe (the backup
    policy brakes to v=0), so penalising low speed would make h_imp permanently
    negative after any braking manoeuvre.
    """
    _, _, v, _, delta = x
    return float(min(
        delta - p.delta_min - p.r_safe_kin,
        p.delta_max - delta - p.r_safe_kin,
        p.v_max - v - p.r_safe_kin,
    ))


def margin_components(x: np.ndarray, ctx: MarginContext) -> dict:
    return {
        "lane": margin_lane(x, ctx.lane, ctx.params),
        "obstacle": margin_obstacle(x, ctx.obstacles, ctx.params),
        "traffic": margin_obstacle(x, ctx.traffic, ctx.params, ctx.params.r_safe_traf),
        "kinematic": margin_kinematic(x, ctx.params),
    }


def margin_total(x: np.ndarray, ctx: MarginContext) -> Tuple[float, str]:
    """Returns (min_margin, binding_constraint_name)."""
    comps = margin_components(x, ctx)
    label = min(comps, key=comps.get)
    return float(comps[label]), label
