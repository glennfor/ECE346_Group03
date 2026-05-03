"""
ILQR cost functions — pure JAX, JIT-compilable, auto-differentiable.

cost_params layout (index): [w_b, tau, w_c, w_psi, w_v, R_a, R_w, beta, v_ref]
                              0     1    2    3      4    5    6    7     8

stage_cost(x, u, ctx)    → scalar (all four terms)
terminal_cost(x, ctx)    → scalar (barrier + centerline + speed, no control term)
"""
import jax
import jax.numpy as jnp

from .jax_context import ILQRContext
from .margins import margin_total_soft


# ---------------------------------------------------------------------------
# Stage cost
# ---------------------------------------------------------------------------

@jax.jit
def stage_cost(x: jnp.ndarray, u: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """
    Four-term ILQR stage cost.

    Term 1 — Safety barrier (dominant, w_b ~ 1000):
        c_barrier = w_b * softplus(-ell/tau)^2

    Term 2 — Centerline tracking (small, w_c ~ 1.0):
        c_ctr = w_c * e_cross^2 + w_psi * e_heading^2

    Term 3 — Speed preserving (very small, w_v ~ 0.1):
        c_speed = w_v * (v - v_ref)^2

    Term 4 — Control regularization:
        c_ctrl = R_a * a^2 + R_w * omega^2
    """
    w_b, tau, w_c, w_psi, w_v, R_a, R_w, _beta, v_ref = (
        ctx.cost_params[0], ctx.cost_params[1], ctx.cost_params[2],
        ctx.cost_params[3], ctx.cost_params[4], ctx.cost_params[5],
        ctx.cost_params[6], ctx.cost_params[7], ctx.cost_params[8],
    )

    # --- Term 1: safety barrier ---
    ell_soft = margin_total_soft(x, ctx)
    c_barrier = w_b * jax.nn.softplus(-ell_soft / tau) ** 2

    # --- Term 2: centerline tracking ---
    # Find nearest centerline point (argmin has zero grad; correct for ILQR:
    # gradient flows through the formula after the lookup, not through argmin)
    dists_sq  = jnp.sum((ctx.lane_pts - x[:2][None, :]) ** 2, axis=1)
    idx       = jnp.argmin(dists_sq)
    center_pt = ctx.lane_pts[idx]
    tan_angle = ctx.lane_tans[idx]
    left_n    = jnp.array([-jnp.sin(tan_angle), jnp.cos(tan_angle)])
    # Cross-track error (signed, +ve = left of centerline)
    cross_track = jnp.dot(x[:2] - center_pt, left_n)
    # Heading error
    heading_err = jnp.arctan2(
        jnp.sin(x[3] - tan_angle),
        jnp.cos(x[3] - tan_angle),
    )
    c_ctr = w_c * cross_track ** 2 + w_psi * heading_err ** 2

    # --- Term 3: speed preserving ---
    c_speed = w_v * (x[2] - v_ref) ** 2

    # --- Term 4: control regularization ---
    c_ctrl = R_a * u[0] ** 2 + R_w * u[1] ** 2

    return c_barrier + c_ctr + c_speed + c_ctrl


# ---------------------------------------------------------------------------
# Terminal cost (no control term)
# ---------------------------------------------------------------------------

@jax.jit
def terminal_cost(x: jnp.ndarray, ctx: ILQRContext) -> jnp.ndarray:
    """Terminal cost at x_H: same as stage cost but without control term."""
    return stage_cost(x, jnp.zeros(2, dtype=x.dtype), ctx)
