"""
CbfParams — single dataclass that holds every tunable knob.
All node parameters come from here; nothing is hardcoded in the nodes.
"""
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class CbfParams:
    # ---- Truck geometry ----
    wheelbase_m: float = 0.257
    truck_radius_m: float = 0.13
    truck_length_m: float = 0.40

    # ---- Control limits ----
    a_min: float = -5.0
    a_max: float = 5.0
    omega_min: float = -6.0
    omega_max: float = 6.0
    delta_min: float = -0.35
    delta_max: float = 0.35
    v_min: float = 0.0
    v_max: float = 5.0

    # ---- Discretization ----
    dt: float = 0.05
    horizon_H: int = 10          # steps for the safety lookahead (H * dt = lookahead time)
    control_rate_hz: float = 20.0

    # ---- Heuristic backup policy (brake + steer toward center) ----
    K_e: float = 2.5             # cross-track error gain (tuned for stability)
    K_p: float = 6.0             # steering-angle P gain on omega (lower → less sway)
    v_eps: float = 0.3           # prevents division by zero at low speed
    backup_omega_lpf_tau_s: float = 0.08  # low-pass ω backup (0 = off)

    # ---- Safety margins ----
    r_safe_lane: float = 0.01    # buffer beyond geometric clearance for lane
    r_safe_obs: float = 0.03     # buffer for static obstacles
    r_safe_traf: float = 0.06    # buffer for dynamic traffic

    # ---- Safety filter thresholds ----
    # throttle_cap_margin is the minimum geometric margin; the filter also computes
    # a speed-dependent stopping distance and uses whichever is larger.
    throttle_cap_margin: float = 0.12
    steer_blend_lpf_tau_s: float = 0.10  # EMA on blended ω during intervention (0 = off)

    # ---- Lane graph / route stability ----
    lane_allow_lane_change: bool = True  # False can yield empty routes; True matches original map API
    route_hysteresis_rad: float = 0.25    # keep prior route if still near-optimal by this heading gap

    # ---- Monitor fallback (map not loaded) ----
    fallback_h_safe: float = 1.0e3  # published h when LaneContext.is_fallback (avoid bogus interventions)

    # ---- Obstacle memory ----
    obstacle_radius_default: float = 0.06
    obstacle_memory_ttl_s: float = 1.0
    obstacle_memory_growth: float = 0.5

    # ---- Staleness ----
    stale_timeout_s: float = 0.3

    # ---- ROS topics ----
    odom_topic: str = "/slam_pose"
    human_control_topic: str = "/human_control"
    filtered_control_topic: str = "/control"
    static_obstacles_topic: str = "/Obstacles/Static"
    dynamic_obstacles_topic: str = "/Obstacles/Dynamic"

    # ---- Lane context ----
    map_file: str = ""
    lane_change_cost: float = 1.0
    lane_context_rebuild_distance_m: float = 0.5
    lane_context_rebuild_yaw_rad: float = 0.5


def declare_and_load(node: Any, names: Iterable[str] = None) -> CbfParams:
    """Declare ROS parameters and return a populated CbfParams."""
    defaults = CbfParams().__dict__.copy()
    selected = list(names) if names is not None else list(defaults.keys())

    for name in selected:
        node.declare_parameter(name, defaults[name])

    values = defaults.copy()
    for name in selected:
        values[name] = node.get_parameter(name).value

    values["horizon_H"] = int(values["horizon_H"])
    return CbfParams(**values)
