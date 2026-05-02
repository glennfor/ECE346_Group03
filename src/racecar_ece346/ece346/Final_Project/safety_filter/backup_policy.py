import numpy as np

from .config import SafetyFilterParams
from .dynamics import clip_control
from .lane_context import LaneContext, heading_error_to_lane


def brake_and_recenter(x: np.ndarray, lane: LaneContext, params: SafetyFilterParams) -> np.ndarray:
    delta_psi, sample = heading_error_to_lane(x, lane)
    _, _, v, _, delta = x

    delta_des = np.clip(
        delta_psi - np.arctan2(params.K_e * sample.signed_lateral_error, abs(v) + params.v_eps),
        params.delta_min,
        params.delta_max,
    )
    omega = np.clip(params.K_p * (delta_des - delta), params.omega_min, params.omega_max)
    return clip_control(np.array([params.a_min, omega], dtype=float), params)

