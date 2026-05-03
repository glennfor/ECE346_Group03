"""
ILQRContext — fixed-shape JAX pytree passed into all JIT-compiled cost and
margin functions.

Using NamedTuple so JAX auto-registers it as a pytree (tuple subclass).
All fields are JAX arrays with shapes that NEVER change at runtime:
  obs_xy_r  / traf_xy_r : (MAX_OBS, 3) — [x, y, radius]
  obs_valid / traf_valid : (MAX_OBS,)   — 1.0 = active, 0.0 = padding
  lane_pts               : (N_LANE, 2)
  lane_tans              : (N_LANE,)
  lane_wl / lane_wr      : (N_LANE,)
  cost_params            : (9,)  [w_b, tau, w_c, w_psi, w_v, R_a, R_w, beta, v_ref]
  kin_params             : (6,)  [delta_min, delta_max, v_max, r_safe_kin, truck_len, truck_r]
"""
from typing import NamedTuple
from typing import List

import numpy as np
import jax.numpy as jnp


class ILQRContext(NamedTuple):
    obs_xy_r:    jnp.ndarray   # (MAX_OBS, 3)
    obs_valid:   jnp.ndarray   # (MAX_OBS,)
    traf_xy_r:   jnp.ndarray   # (MAX_OBS, 3)
    traf_valid:  jnp.ndarray   # (MAX_OBS,)
    lane_pts:    jnp.ndarray   # (N_LANE, 2)
    lane_tans:   jnp.ndarray   # (N_LANE,)
    lane_wl:     jnp.ndarray   # (N_LANE,)
    lane_wr:     jnp.ndarray   # (N_LANE,)
    cost_params: jnp.ndarray   # (9,)
    kin_params:  jnp.ndarray   # (6,)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_context(obstacles, traffic, lane_ctx, v_ref: float, params) -> ILQRContext:
    """
    Convert Python obstacle lists + LaneContext to a fixed-shape ILQRContext.

    obstacles / traffic : list of Obstacle (from obstacle_memory.py)
    lane_ctx            : LaneContext (from lane_context.py)
    v_ref               : current truck speed (used as speed reference)
    params              : ILQRSafetyParams
    """
    MAX_OBS = int(params.max_obstacles)
    N_LANE  = int(params.n_lane_pts)

    obs_arr, obs_valid = _pack_obstacles(obstacles, MAX_OBS)
    traf_arr, traf_valid = _pack_obstacles(traffic, MAX_OBS)

    # Sample centerline to exactly N_LANE points
    n_pts = len(lane_ctx.centerline)
    if n_pts >= 2:
        indices = np.round(np.linspace(0, n_pts - 1, N_LANE)).astype(int)
    else:
        indices = np.zeros(N_LANE, dtype=int)

    lane_pts = lane_ctx.centerline[indices]
    lane_tans = lane_ctx.tangent[indices]
    lane_wl  = lane_ctx.width_left[indices]
    lane_wr  = lane_ctx.width_right[indices]

    cost_p = jnp.array([
        params.w_barrier,
        params.tau,
        params.w_centerline,
        params.w_heading,
        params.w_speed,
        params.R_accel,
        params.R_omega,
        params.soft_min_beta,
        float(min(v_ref, params.v_max)),
    ], dtype=jnp.float32)

    kin_p = jnp.array([
        params.delta_min,
        params.delta_max,
        params.v_max,
        params.r_safe_kin,
        params.truck_length_m,
        params.truck_radius_m,
    ], dtype=jnp.float32)

    return ILQRContext(
        obs_xy_r=jnp.array(obs_arr, dtype=jnp.float32),
        obs_valid=jnp.array(obs_valid, dtype=jnp.float32),
        traf_xy_r=jnp.array(traf_arr, dtype=jnp.float32),
        traf_valid=jnp.array(traf_valid, dtype=jnp.float32),
        lane_pts=jnp.array(lane_pts, dtype=jnp.float32),
        lane_tans=jnp.array(lane_tans, dtype=jnp.float32),
        lane_wl=jnp.array(lane_wl, dtype=jnp.float32),
        lane_wr=jnp.array(lane_wr, dtype=jnp.float32),
        cost_params=cost_p,
        kin_params=kin_p,
    )


def dummy_context(params) -> ILQRContext:
    """Return a zeroed-out context for JIT warm-up."""
    MAX_OBS = int(params.max_obstacles)
    N_LANE  = int(params.n_lane_pts)
    zero_obs = jnp.zeros((MAX_OBS, 3), dtype=jnp.float32)
    zero_mask = jnp.zeros(MAX_OBS, dtype=jnp.float32)
    # Straight lane along x-axis, width 1 m each side
    xs = jnp.linspace(0.0, 10.0, N_LANE)
    lane_pts = jnp.stack([xs, jnp.zeros(N_LANE)], axis=1).astype(jnp.float32)
    lane_tans = jnp.zeros(N_LANE, dtype=jnp.float32)
    half_w = jnp.ones(N_LANE, dtype=jnp.float32) * 0.5
    cost_p = jnp.array([
        params.w_barrier, params.tau, params.w_centerline,
        params.w_heading, params.w_speed, params.R_accel,
        params.R_omega, params.soft_min_beta, 1.0,
    ], dtype=jnp.float32)
    kin_p = jnp.array([
        params.delta_min, params.delta_max, params.v_max,
        params.r_safe_kin, params.truck_length_m, params.truck_radius_m,
    ], dtype=jnp.float32)
    return ILQRContext(
        obs_xy_r=zero_obs, obs_valid=zero_mask,
        traf_xy_r=zero_obs, traf_valid=zero_mask,
        lane_pts=lane_pts, lane_tans=lane_tans,
        lane_wl=half_w, lane_wr=half_w,
        cost_params=cost_p, kin_params=kin_p,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_obstacles(obstacles, max_obs: int):
    """Pack a list of Obstacle objects into a fixed-size numpy array."""
    arr   = np.zeros((max_obs, 3), dtype=np.float32)
    valid = np.zeros(max_obs, dtype=np.float32)
    for i, obs in enumerate(obstacles[:max_obs]):
        arr[i, 0] = float(obs.position[0])
        arr[i, 1] = float(obs.position[1])
        arr[i, 2] = float(obs.radius)
        valid[i]  = 1.0
    return arr, valid
