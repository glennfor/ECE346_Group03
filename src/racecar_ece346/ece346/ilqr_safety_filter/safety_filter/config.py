from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class ILQRSafetyParams:
    # Truck physical constants — from Lab1/sim YAML
    wheelbase_m: float = 0.257
    truck_radius_m: float = 0.13
    truck_length_m: float = 0.40

    # Control limits
    a_min: float = -5.0
    a_max: float = 5.0
    omega_min: float = -6.0
    omega_max: float = 6.0
    delta_min: float = -0.35
    delta_max: float = 0.35
    v_min: float = 0.0
    v_max: float = 5.0

    # Discretization
    dt: float = 0.05          # 20 Hz control, 0.05 s step

    # ILQR solver
    horizon_H: int = 30
    ilqr_max_iters: int = 10
    ilqr_tol: float = 1e-3
    lm_init: float = 1e-6
    lm_min: float = 1e-8
    lm_max: float = 1.0
    lm_scale_up: float = 10.0
    lm_scale_down: float = 10.0

    # Cost weights (barrier dominates all others by ~1000x)
    w_barrier: float = 1000.0
    tau: float = 0.1
    w_centerline: float = 1.0
    w_heading: float = 0.5
    w_speed: float = 0.1
    R_accel: float = 0.1
    R_omega: float = 0.05
    soft_min_beta: float = 50.0

    # Safety margins
    r_safe_lane: float = 0.05
    r_safe_obs: float = 0.05
    r_safe_traf: float = 0.10
    r_safe_kin: float = 0.02
    obstacle_radius_default: float = 0.06
    obstacle_memory_ttl_s: float = 1.0
    obstacle_memory_growth: float = 0.5

    # Monitor / arbiter
    hysteresis_cycles: int = 5
    passthrough_tolerance: float = 1e-3
    stale_timeout_s: float = 0.25

    # JAX fixed array sizes — MUST match at runtime; changes require restart
    max_obstacles: int = 10
    n_lane_pts: int = 100

    # ROS interface
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

    # Rates
    planner_rate_hz: float = 10.0
    control_rate_hz: float = 20.0
    publish_debug: bool = True


def parameter_defaults() -> Dict[str, Any]:
    return ILQRSafetyParams().__dict__.copy()


def declare_and_load(node: Any, names: Iterable[str] = None) -> ILQRSafetyParams:
    defaults = parameter_defaults()
    selected_names = list(names) if names is not None else list(defaults.keys())

    for name in selected_names:
        node.declare_parameter(name, defaults[name])

    values = defaults.copy()
    for name in selected_names:
        values[name] = node.get_parameter(name).value

    # Coerce integer fields
    for int_field in ("horizon_H", "ilqr_max_iters", "hysteresis_cycles",
                      "max_obstacles", "n_lane_pts"):
        values[int_field] = int(values[int_field])

    return ILQRSafetyParams(**values)
