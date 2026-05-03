import numpy as np

from .config import SafetyFilterParams
from .dynamics import clip_control
from .lane_context import LaneContext, heading_error_to_lane


def brake_and_recenter(x: np.ndarray, lane: LaneContext, params: SafetyFilterParams) -> np.ndarray:
    return recenter_control(x, lane, params, params.a_min)


def lane_recovery_control(
    x: np.ndarray,
    lane: LaneContext,
    params: SafetyFilterParams,
    u_human: np.ndarray = None,
) -> np.ndarray:
    _, _, v, _, _ = x
    human_accel = 0.0 if u_human is None else float(u_human[0])
    if human_accel <= 1e-3:
        accel = 0.0 if v <= 0.02 else params.a_min
    else:
        speed_limit_accel = params.lane_recovery_accel_gain * (params.lane_recovery_speed_mps - v)
        accel = min(human_accel, speed_limit_accel)
    return recenter_control(x, lane, params, accel)


def recenter_control(
    x: np.ndarray,
    lane: LaneContext,
    params: SafetyFilterParams,
    accel: float,
) -> np.ndarray:
    delta_psi, sample = heading_error_to_lane(x, lane)
    _, _, v, _, delta = x

    delta_des = np.clip(
        delta_psi - np.arctan2(params.K_e * sample.signed_lateral_error, abs(v) + params.v_eps),
        params.delta_min,
        params.delta_max,
    )
    omega = np.clip(params.K_p * (delta_des - delta), params.omega_min, params.omega_max)
    return clip_control(np.array([accel, omega], dtype=float), params)

