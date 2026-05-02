from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class SafetyFilterParams:
    wheelbase_m: float = 0.324
    truck_radius_m: float = 0.274
    truck_length_m: float = 0.46

    a_min: float = -1.0
    a_max: float = 1.0
    omega_min: float = -3.0
    omega_max: float = 3.0
    delta_min: float = -0.26
    delta_max: float = 0.31
    v_min: float = -0.3
    v_max: float = 0.4

    dt: float = 0.05
    horizon_H: int = 30
    control_rate_hz: float = 20.0

    r_safe_lane: float = 0.05
    r_safe_obs: float = 0.05
    r_safe_traf: float = 0.10
    r_safe_kin: float = 0.02
    lane_guard_margin_m: float = 0.05
    lane_recovery_speed_mps: float = 0.2
    lane_recovery_accel_gain: float = 4.0
    obstacle_radius_default: float = 0.12
    obstacle_memory_ttl_s: float = 1.0
    obstacle_memory_growth: float = 0.5

    K_e: float = 1.0
    K_p: float = 5.0
    v_eps: float = 0.3

    lambda_cbf: float = 0.4
    qp_R_accel: float = 1.0
    qp_R_omega: float = 1.0
    grad_eps: float = 1e-3
    exact_safety_check: bool = True
    hysteresis_cycles: int = 5
    passthrough_tolerance: float = 1e-3
    stale_timeout_s: float = 0.25
    recovery_h_improvement: float = 1e-3

    command_mode: str = "servo"
    odom_topic: str = "/slam_pose"
    human_control_topic: str = "/human_control"
    filtered_control_topic: str = "/control"
    static_obstacles_topic: str = "/Obstacles/Static"
    dynamic_obstacles_topic: str = "/Obstacles/Dynamic"
    map_file: str = ""
    lane_change_cost: float = 1.0
    lane_context_rebuild_distance_m: float = 0.25
    lane_context_rebuild_yaw_rad: float = 0.35
    publish_debug: bool = True


def parameter_defaults() -> Dict[str, Any]:
    return SafetyFilterParams().__dict__.copy()


def declare_and_load(node: Any, names: Iterable[str] = None) -> SafetyFilterParams:
    defaults = parameter_defaults()
    selected_names = list(names) if names is not None else list(defaults.keys())

    for name in selected_names:
        node.declare_parameter(name, defaults[name])

    values = defaults.copy()
    for name in selected_names:
        value = node.get_parameter(name).value
        values[name] = value

    values["horizon_H"] = int(values["horizon_H"])
    values["hysteresis_cycles"] = int(values["hysteresis_cycles"])
    return SafetyFilterParams(**values)

