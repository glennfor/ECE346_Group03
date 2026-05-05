from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backup_policy import emergency_brake, lane_recovery_control
from .config import SafetyFilterParams
from .lane_context import LaneContext


@dataclass
class GuardResult:
    control: Optional[np.ndarray]
    status: str


def select_hard_guard_control(
    state: np.ndarray,
    lane: LaneContext,
    margins: dict,
    params: SafetyFilterParams,
    u_human: np.ndarray,
) -> GuardResult:
    if margins["obstacle"] < params.obstacle_guard_margin_m:
        return GuardResult(emergency_brake(state, params), "obstacle_guard_brake")
    if margins["traffic"] < params.traffic_guard_margin_m:
        return GuardResult(emergency_brake(state, params), "traffic_guard_brake")
    if margins["lane"] < params.lane_guard_margin_m:
        return GuardResult(lane_recovery_control(state, lane, params, u_human), "lane_guard_recenter")
    return GuardResult(None, "no_guard")
