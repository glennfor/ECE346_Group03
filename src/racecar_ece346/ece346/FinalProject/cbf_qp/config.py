from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("cbf_qp.yaml")


@dataclass
class SafetyFilterParams:
    wheelbase_m: float = 0.324
    truck_radius_m: float = 0.274
    truck_length_m: float = 0.46

    a_min: float = -4.0
    a_max: float = 3.0
    omega_min: float = -6.0
    omega_max: float = 6.0
    delta_min: float = -0.34
    delta_max: float = 0.34
    v_min: float = 0.0
    v_max: float = 1.0

    dt: float = 0.05
    horizon_H: int = 30
    control_rate_hz: float = 30.0

    r_safe_lane: float = 0.05
    r_safe_obs: float = 0.05
    r_safe_traf: float = 0.10
    r_safe_kin: float = 0.02
    lane_guard_margin_m: float = 0.05
    obstacle_guard_margin_m: float = 0.35
    traffic_guard_margin_m: float = 0.45
    forward_obstacle_width_m: float = 0.45
    forward_obstacle_distance_m: float = 1.4
    forward_obstacle_brake_margin_m: float = 0.25
    lane_recovery_speed_mps: float = 0.65
    lane_recovery_accel_gain: float = 5.0
    obstacle_radius_default: float = 0.12
    obstacle_memory_ttl_s: float = 1.0
    obstacle_memory_growth: float = 0.5

    K_e: float = 2.2
    K_p: float = 7.0
    v_eps: float = 0.3

    lambda_cbf: float = 0.4
    qp_R_accel: float = 1.0
    qp_R_omega: float = 1.0
    grad_eps: float = 1e-3
    exact_safety_check: bool = True
    hysteresis_cycles: int = 5
    passthrough_tolerance: float = 1e-3
    stale_timeout_s: float = 0.5
    recovery_h_improvement: float = 1e-3

    command_mode: str = "ackermann"
    odom_topic: str = "/slam_pose"
    human_control_topic: str = "/teleop"
    filtered_control_topic: str = "/drive"
    static_obstacles_topic: str = "/Obstacles/Static"
    dynamic_obstacles_topic: str = "/Obstacles/Dynamic"
    routing_path_topic: str = "/Routing/Path"
    map_file: str = ""
    lane_change_cost: float = 1.0
    lane_allow_lane_change: bool = True
    route_hysteresis_rad: float = 0.25
    lane_context_rebuild_distance_m: float = 0.25
    lane_context_rebuild_yaw_rad: float = 0.35
    publish_debug: bool = True

    teleop_topic: str = "/teleop"
    drive_topic: str = "/drive"
    static_obs_topic: str = "/Obstacles/Static"
    publish_rate: float = 30.0


def parameter_defaults() -> Dict[str, Any]:
    return load_config().__dict__.copy()


def _flatten_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def load_config(path: str = "") -> SafetyFilterParams:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r") as stream:
            raw = yaml.safe_load(stream) or {}

    values = _flatten_config(raw)
    valid_names = {field.name for field in fields(SafetyFilterParams)}
    kwargs = {key: value for key, value in values.items() if key in valid_names}
    return SafetyFilterParams(**kwargs)


def declare_and_load(node: Any, names: Iterable[str] = None) -> SafetyFilterParams:
    node.declare_parameter("cbf_qp_config", str(DEFAULT_CONFIG_PATH))
    config_path = str(node.get_parameter("cbf_qp_config").value)
    defaults = load_config(config_path).__dict__.copy()
    selected_names = list(names) if names is not None else list(defaults.keys())

    for name in selected_names:
        node.declare_parameter(name, defaults[name])

    values = defaults.copy()
    for name in selected_names:
        values[name] = node.get_parameter(name).value

    values["human_control_topic"] = values["teleop_topic"]
    values["filtered_control_topic"] = values["drive_topic"]
    values["static_obstacles_topic"] = values["static_obs_topic"]
    values["control_rate_hz"] = float(values["publish_rate"])
    values["command_mode"] = "ackermann"
    values["horizon_H"] = int(values["horizon_H"])
    values["hysteresis_cycles"] = int(values["hysteresis_cycles"])
    return SafetyFilterParams(**values)

