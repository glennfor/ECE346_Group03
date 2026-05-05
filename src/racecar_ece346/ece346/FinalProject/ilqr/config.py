"""Configuration loading for the ILQR-QP safety filter."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("ilqr_filter.yaml")


@dataclass(frozen=True)
class IlqrQpConfig:
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
    log_period_sec: float = 0.5

    wheelbase: float = 0.324
    max_speed: float = 1.0
    min_speed: float = 0.0
    max_steering_angle: float = 0.34
    max_decel: float = 5.0
    max_accel: float = 2.5
    max_steering_rate: float = 4.0
    response_delay_sec: float = 0.10
    allow_reverse: bool = False

    horizon_sec: float = 2.0
    dt: float = 0.10
    min_projection_speed: float = 0.35
    vehicle_radius: float = 0.24
    obstacle_radius_buffer: float = 0.12
    lane_margin: float = 0.16
    localization_buffer: float = 0.08
    stopping_buffer: float = 0.12
    require_path_for_lane_filter: bool = False

    ilqr_iterations: int = 4
    speed_scales: tuple = (1.0, 0.85, 0.65, 0.45, 0.25, 0.0)
    steering_offsets: tuple = (0.0, -0.08, 0.08, -0.16, 0.16, -0.26, 0.26, -0.34, 0.34)
    recovery_steering_offsets: tuple = (-0.34, -0.24, -0.14, 0.0, 0.14, 0.24, 0.34)

    speed_weight: float = 1.0
    steering_weight: float = 1.3
    brake_weight: float = 0.25
    steering_rate_weight: float = 0.04
    lane_violation_weight: float = 900.0
    obstacle_violation_weight: float = 1400.0
    progress_weight: float = 0.04
    heading_weight: float = 0.15

    soft_margin: float = 0.10
    hard_margin: float = 0.00
    no_solution_speed: float = 0.0
    no_solution_keep_steering: bool = True
    stale_odom_speed: float = 0.0
    stale_odom_steering: float = 0.0


def _flatten_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def _coerce_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def load_config(path: str = "") -> IlqrQpConfig:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r") as stream:
            raw = yaml.safe_load(stream) or {}

    values = _flatten_config(raw)
    valid_names = {field.name for field in fields(IlqrQpConfig)}
    kwargs = {
        key: _coerce_value(value)
        for key, value in values.items()
        if key in valid_names
    }
    return IlqrQpConfig(**kwargs)
