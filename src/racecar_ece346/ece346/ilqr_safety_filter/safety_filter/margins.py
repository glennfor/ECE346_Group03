"""
Safety margin functions — pure JAX, JIT-compilable.

All functions accept ILQRContext (a NamedTuple, so a valid pytree) and
a state x: (5,).

kin_params layout: [delta_min, delta_max, v_max, r_safe_kin, truck_len, truck_r]
cost_params layout: [w_b, tau, w_c, w_psi, w_v, R_a, R_w, beta, v_ref]

Hard vs soft min:
  margin_total_hard — used by the monitor's V̂ check (accurate worst case)
  margin_total_soft — used inside ILQR cost (differentiable soft-min)
"""
import jax
import jax.numpy as jnp

from .jax_context import ILQRContext


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _footprint_points(x: jnp.ndarray, truck_len: jnp.ndarray) -> jnp.ndarray:
    """Three inflation circles along the longitudinal axis.  Returns (3, 2)."""
    psi = x[3]
    heading = jnp.array([jnp.cos(psi), jnp.sin(psi)])
    offsets = jnp.array([-0.35, 0.0, 0.35]) * truck_len   # (3,)
    center  = x[:2]                                         # (2,)
    # (3, 2)
    return center[None, :] + offsets[:, None] * heading[None, :]


# ---------------------------------------------------------------------------
# Lane margin
# ---------------------------------------------------------------------------

@jax.jit
def margin_lane(x: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """
    Signed distance from truck footprint to nearest lane boundary.
    Positive = inside lane with clearance.
    """
    truck_len = ctx.kin_params[4]
    truck_r   = ctx.kin_params[5]
    foot_pts  = _footprint_points(x, truck_len)   # (3, 2)

    def point_margin(p: jnp.ndarray) -> jnp.ndarray:
        dists_sq = jnp.sum((ctx.lane_pts - p[None, :]) ** 2, axis=1)   # (N,)
        idx      = jnp.argmin(dists_sq)
        center_pt = ctx.lane_pts[idx]
        tan_angle = ctx.lane_tans[idx]
        # Left-hand normal (90° CCW from tangent)
        left_n    = jnp.array([-jnp.sin(tan_angle), jnp.cos(tan_angle)])
        lat_err   = jnp.dot(p - center_pt, left_n)  # +ve = left of centerline
        # Distance to left / right boundary
        left_margin  = ctx.lane_wl[idx] - lat_err  - truck_r
        right_margin = ctx.lane_wr[idx] + lat_err  - truck_r
        return jnp.minimum(left_margin, right_margin)

    margins = jax.vmap(point_margin)(foot_pts)   # (3,)
    return jnp.min(margins)


# ---------------------------------------------------------------------------
# Obstacle margin  (static or traffic — caller picks the arrays)
# ---------------------------------------------------------------------------

@jax.jit
def margin_obstacle(
    x: jnp.ndarray,
    obs_xy_r: jnp.ndarray,   # (MAX_OBS, 3)
    obs_valid: jnp.ndarray,  # (MAX_OBS,)
    ctx: ILQRContext,
    safety_margin: float = 0.0,
) -> jnp.ndarray:
    """
    Min clearance from truck footprint to any valid obstacle.
    Returns +100 when no obstacles are active.
    """
    truck_len = ctx.kin_params[4]
    truck_r   = ctx.kin_params[5]
    foot_pts  = _footprint_points(x, truck_len)   # (3, 2)

    def point_obs_margin(p: jnp.ndarray) -> jnp.ndarray:
        # Euclidean distance to each obstacle centre
        dists = jnp.sqrt(jnp.sum((obs_xy_r[:, :2] - p[None, :]) ** 2, axis=1) + 1e-12)
        clearances = dists - truck_r - obs_xy_r[:, 2] - safety_margin
        # Mask out padding rows with +100 so they don't constrain
        masked = jnp.where(obs_valid > 0.5, clearances, 100.0)
        return jnp.min(masked)

    footprint_margins = jax.vmap(point_obs_margin)(foot_pts)   # (3,)
    return jnp.min(footprint_margins)


# ---------------------------------------------------------------------------
# Kinematic margin (steering angle and speed envelope)
# ---------------------------------------------------------------------------

@jax.jit
def margin_kinematic(x: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    delta_min, delta_max, v_max, r_safe_kin = (
        ctx.kin_params[0], ctx.kin_params[1], ctx.kin_params[2], ctx.kin_params[3]
    )
    v     = x[2]
    delta = x[4]
    return jnp.minimum(
        jnp.minimum(delta - delta_min, delta_max - delta),
        jnp.minimum(v, v_max - v),
    ) - r_safe_kin


# ---------------------------------------------------------------------------
# Combined margins
# ---------------------------------------------------------------------------

@jax.jit
def margin_total_hard(x: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """Hard min over all four components.  Used by the monitor's V̂ check."""
    ml = margin_lane(x, ctx)
    mo = margin_obstacle(x, ctx.obs_xy_r,  ctx.obs_valid,  ctx, 0.0)
    mt = margin_obstacle(x, ctx.traf_xy_r, ctx.traf_valid, ctx, 0.0)
    mk = margin_kinematic(x, ctx)
    return jnp.minimum(jnp.minimum(ml, mo), jnp.minimum(mt, mk))


@jax.jit
def margin_total_soft(x: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """
    Differentiable soft-min (log-sum-exp) over four components.
    Used inside ILQR cost function where gradients must exist.
    """
    beta = ctx.cost_params[7]
    ml   = margin_lane(x, ctx)
    mo   = margin_obstacle(x, ctx.obs_xy_r,  ctx.obs_valid,  ctx, 0.0)
    mt   = margin_obstacle(x, ctx.traf_xy_r, ctx.traf_valid, ctx, 0.0)
    mk   = margin_kinematic(x, ctx)
    components = jnp.array([ml, mo, mt, mk])
    return -jax.scipy.special.logsumexp(-beta * components) / beta


# ---------------------------------------------------------------------------
# Convenience: evaluate hard margins for an entire trajectory (vmap wrapper)
# ---------------------------------------------------------------------------

def trajectory_hard_margins(X: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """X: (H+1, 5) → margins: (H+1,)  via vmap."""
    return jax.vmap(margin_total_hard, in_axes=(0, None))(X, ctx)


trajectory_hard_margins_jit = jax.jit(trajectory_hard_margins)
