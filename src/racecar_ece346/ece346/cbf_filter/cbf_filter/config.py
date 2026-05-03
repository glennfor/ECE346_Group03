"""
CbfParams — single dataclass that holds every tunable knob.
All node parameters come from here; nothing is hardcoded in the nodes.

Physical constants are taken from the simulator's Bicycle4D class (wheelbase=0.257)
and the ILQR filter config (truck_radius=0.13), which have been validated.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class CbfParams:
    # ---- Truck geometry ----
    wheelbase_m: float = 0.257          # matches simulator Bicycle4D
    truck_radius_m: float = 0.13        # bounding-circle radius of 1/10-scale truck
    truck_length_m: float = 0.40

    # ---- Control limits ----
    a_min: float = -5.0                 # max braking (m/s²)
    a_max: float = 5.0
    omega_min: float = -6.0             # max steering rate (rad/s)
    omega_max: float = 6.0
    delta_min: float = -0.35            # steering angle limits (rad)
    delta_max: float = 0.35
    v_min: float = 0.0
    v_max: float = 5.0

    # ---- Discretization ----
    dt: float = 0.05                    # 20 Hz control rate
    horizon_H: int = 30                 # rollout length (steps).  H * dt = lookahead time.
                                        # Set H = ceil(1.5 * T_stop / dt) after measuring
                                        # stopping time T_stop on the real truck.
    control_rate_hz: float = 20.0

    # ---- Backup policy (brake-and-recenter) ----
    K_e: float = 1.0                    # cross-track error gain
    K_p: float = 5.0                    # steering rate proportional gain
    v_eps: float = 0.3                  # prevents division by zero at low speed

    # ---- Safety margins ----
    # Signed-distance buffer on top of geometric clearance.
    # Increasing these makes the filter more conservative.
    r_safe_lane: float = 0.05
    r_safe_obs: float = 0.05
    r_safe_traf: float = 0.10
    r_safe_kin: float = 0.02

    # ---- Obstacle memory ----
    obstacle_radius_default: float = 0.06   # standard ECE346 cube half-size
    obstacle_memory_ttl_s: float = 1.0      # keep last-seen obstacle for this long
    obstacle_memory_growth: float = 0.5     # radius grows (m/s) as detection ages

    # ---- CBF-QP ----
    # lambda_cbf ∈ (0, 1]: controls how fast the barrier is allowed to decay.
    # 1 = non-decrease required; 0 = unconstrained.
    # Increase if the truck gets too close to obstacles.
    # Decrease if intervention is too aggressive.
    lambda_cbf: float = 0.4
    qp_R_accel: float = 1.0             # QP cost weight on changing acceleration
    qp_R_omega: float = 1.0             # QP cost weight on changing steering rate

    # Gradient finite-difference step (m, m/s, rad, ...)
    grad_eps: float = 1e-3

    # ---- Hysteresis ----
    hysteresis_cycles: int = 5          # keep QP active this many cycles after override ends
    passthrough_tolerance: float = 1e-3 # treat QP solution as unchanged if within this

    # ---- Staleness ----
    stale_timeout_s: float = 0.3        # ignore odom/safety/human msgs older than this

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

    # Ensure integer types are correct (ROS can return them as float).
    values["horizon_H"] = int(values["horizon_H"])
    values["hysteresis_cycles"] = int(values["hysteresis_cycles"])
    return CbfParams(**values)
