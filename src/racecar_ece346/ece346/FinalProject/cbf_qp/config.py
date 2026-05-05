from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


# ---------------------------------------------------------------------------
# Implicit Backup-CBF ROS parameters (student ``final_project_*.yaml``)
# ---------------------------------------------------------------------------


@dataclass
class SafetyFilterParams:
    wheelbase_m: float = 0.324
    truck_radius_m: float = 0.14
    truck_length_m: float = 0.38

    a_min: float = -5.0
    a_max: float = 5.0
    omega_min: float = -6.0
    omega_max: float = 6.0
    delta_min: float = -0.34
    delta_max: float = 0.34
    v_min: float = 0.0
    v_max: float = 1.0

    dt: float = 1.0 / 30.0
    horizon_H: int = 30
    control_rate_hz: float = 30.0

    r_safe_lane: float = 0.02
    r_safe_obs: float = 0.12
    r_safe_traf: float = 0.18
    r_safe_kin: float = 0.0
    lane_guard_margin_m: float = 0.08
    obstacle_guard_margin_m: float = 0.35
    traffic_guard_margin_m: float = 0.45
    lane_recovery_speed_mps: float = 0.25
    lane_recovery_accel_gain: float = 4.0
    obstacle_radius_default: float = 0.14
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
    hysteresis_cycles: int = 15
    passthrough_tolerance: float = 0.05
    stale_timeout_s: float = 0.5
    recovery_h_improvement: float = 1e-3

    command_mode: str = "ackermann"
    odom_topic: str = "/slam_pose"
    human_control_topic: str = "/teleop"
    filtered_control_topic: str = "/drive"
    static_obstacles_topic: str = "/Obstacles/Static"
    dynamic_obstacles_topic: str = "/Obstacles/Dynamic"
    map_file: str = ""
    lane_change_cost: float = 1.0
    lane_context_rebuild_distance_m: float = 0.5
    lane_context_rebuild_yaw_rad: float = 0.35
    publish_debug: bool = True


_BRIDGED_FIELD_NAMES = {
    "human_control_topic",
    "filtered_control_topic",
    "static_obstacles_topic",
    "control_rate_hz",
    "dynamic_obstacles_topic",
    "map_file",
    "publish_debug",
    "command_mode",
}


def parameter_defaults() -> Dict[str, Any]:
    """Mutable defaults for declaring ROS parameters."""
    defaults = SafetyFilterParams().__dict__.copy()
    del defaults["human_control_topic"]
    del defaults["filtered_control_topic"]
    del defaults["static_obstacles_topic"]
    del defaults["dynamic_obstacles_topic"]
    del defaults["control_rate_hz"]
    del defaults["map_file"]
    del defaults["publish_debug"]
    del defaults["command_mode"]
    return defaults


def _declare_aliases(node: Any) -> None:
    node.declare_parameter("teleop_topic", "/teleop")
    node.declare_parameter("drive_topic", "/drive")
    node.declare_parameter("static_obs_topic", "/Obstacles/Static")
    node.declare_parameter("publish_rate", 30.0)
    node.declare_parameter("dynamic_obstacles_topic", "/Obstacles/Dynamic")
    node.declare_parameter("map_file", "")
    node.declare_parameter("publish_debug", True)
    node.declare_parameter("routing_path_topic", "/Routing/Path")


def declare_and_load(node: Any, names: Iterable[str] = None) -> SafetyFilterParams:
    """
    Load SafetyFilter params from ROS, bridging Final Project topic names:

    ``teleop_topic`` / ``drive_topic`` / ``static_obs_topic`` / ``publish_rate``
    map to internal routing fields used by Final_Project.
    """
    _declare_aliases(node)
    defaults = parameter_defaults()
    selected_names = sorted(defaults.keys()) if names is None else list(names)
    skipped = sorted(_BRIDGED_FIELD_NAMES - {"command_mode"})

    for name in selected_names:
        if name in skipped:
            continue
        node.declare_parameter(name, defaults[name])

    values = {}
    values["human_control_topic"] = node.get_parameter("teleop_topic").value
    values["filtered_control_topic"] = node.get_parameter("drive_topic").value
    values["static_obstacles_topic"] = node.get_parameter("static_obs_topic").value
    values["control_rate_hz"] = float(node.get_parameter("publish_rate").value)
    values["dynamic_obstacles_topic"] = node.get_parameter("dynamic_obstacles_topic").value
    values["map_file"] = node.get_parameter("map_file").value
    values["publish_debug"] = bool(node.get_parameter("publish_debug").value)
    values["command_mode"] = "ackermann"

    int_fields = {"horizon_H", "hysteresis_cycles"}
    bool_fields = {"exact_safety_check"}

    for name in selected_names:
        if name in skipped:
            continue
        value = node.get_parameter(name).value
        if name in int_fields:
            value = int(value)
        if name in bool_fields:
            value = bool(value)
        values[name] = value

    return SafetyFilterParams(**values)


# ---------------------------------------------------------------------------
# Heuristic grid-search filter (optional second backend via ``heuristic_main``)
# YAML: ``cbf_qp.yaml`` flattened sections (topics / timing / vehicle / …)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).with_name("cbf_qp.yaml")


@dataclass(frozen=True)
class CbfQpConfig:
    teleop_topic: str = "/teleop"
    drive_topic: str = "/drive"
    odom_topic: str = "/slam_pose"
    static_obs_topic: str = "/Obstacles/Static"
    routing_path_topic: str = "/Routing/Path"

    publish_rate: float = 30.0
    teleop_timeout_sec: float = 0.5
    odom_timeout_sec: float = 0.5
    obstacle_timeout_sec: float = 1.0
    path_timeout_sec: float = 5.0

    wheelbase: float = 0.324
    max_speed: float = 1.0
    min_speed: float = 0.0
    max_steering_angle: float = 0.34
    max_decel: float = 5.0
    response_delay_sec: float = 0.12

    horizon_sec: float = 2.0
    dt: float = 0.1
    min_projection_speed: float = 0.25

    vehicle_radius: float = 0.24
    obstacle_radius_buffer: float = 0.12
    lane_margin: float = 0.18
    localization_buffer: float = 0.08
    stopping_buffer: float = 0.10
    cbf_gamma: float = 0.55
    require_path_for_lane_filter: bool = False

    speed_samples: tuple = (1.0, 0.85, 0.65, 0.45, 0.25, 0.0)
    steering_offsets: tuple = (
        0.0,
        -0.08,
        0.08,
        -0.16,
        0.16,
        -0.26,
        0.26,
        -0.34,
        0.34,
    )
    speed_weight: float = 1.0
    steering_weight: float = 1.25
    brake_weight: float = 0.25
    lane_violation_weight: float = 700.0
    obstacle_violation_weight: float = 1200.0
    progress_weight: float = 0.03

    stale_odom_speed: float = 0.0
    stale_odom_steering: float = 0.0
    no_solution_speed: float = 0.0
    no_solution_keep_steering: bool = True
    allow_reverse: bool = False

    log_period_sec: float = 0.5


def _flatten_heuristic_yaml(data: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for value in data.values():
        if isinstance(value, Mapping):
            flattened.update(value)
    return flattened


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def load_config(path: str = "") -> CbfQpConfig:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r") as stream:
            raw = yaml.safe_load(stream) or {}

    values = _flatten_heuristic_yaml(raw)
    valid_names = {field.name for field in fields(CbfQpConfig)}
    kwargs = {
        key: _coerce_value(value)
        for key, value in values.items()
        if key in valid_names
    }
    return CbfQpConfig(**kwargs)
