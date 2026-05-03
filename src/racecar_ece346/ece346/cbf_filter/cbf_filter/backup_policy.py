"""
Backup policy: brake-and-recenter.

The policy does two things simultaneously:
  1. Brakes at maximum deceleration (a = a_min).
  2. Steers back toward the lane centerline using a P-controller.

This is the "fallback controller" in the Backup-CBF framework.  The implicit
barrier h_imp(x) represents: "if we switch to this policy RIGHT NOW, the
truck stays safe for the next H steps."

Design rationale
----------------
  - Any state that is close to the lane center with low speed is trivially
    safe — the truck just stops.  This is the forward-invariant backup set Ω.
  - From any reasonable on-track state, brake-and-recenter drives the truck
    into Ω by slowing it down while steering toward the centerline.
  - The policy is closed-form (no optimization), runs in microseconds, and
    is smooth enough for the finite-difference gradient to be well-conditioned.

Sign convention note
--------------------
  signed_lateral_error > 0  →  truck is to the LEFT of the centerline.
  To correct: steer right → negative delta desired.
  So the cross-track correction is  -arctan2(K_e * e, |v| + v_eps).
  (The spec shows a + sign but uses the opposite lateral-error convention.)
"""

import numpy as np

from .config import CbfParams
from .dynamics import clip_control
from .lane_context import LaneContext, heading_error_to_lane


def brake_and_recenter(x: np.ndarray, lane: LaneContext, p: CbfParams) -> np.ndarray:
    """
    Returns [a_min, omega] that brakes hard and steers toward the centerline.
    """
    delta_psi, sample = heading_error_to_lane(x, lane)
    _, _, v, _, delta = x

    # Desired steering angle: align heading with the lane tangent, plus a
    # lateral correction proportional to the cross-track error.
    delta_des = np.clip(
        delta_psi - np.arctan2(p.K_e * sample.signed_lateral_error, abs(v) + p.v_eps),
        p.delta_min,
        p.delta_max,
    )

    omega = np.clip(p.K_p * (delta_des - delta), p.omega_min, p.omega_max)
    return clip_control(np.array([p.a_min, omega], dtype=float), p)
