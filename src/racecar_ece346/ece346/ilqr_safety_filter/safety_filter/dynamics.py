"""
Pure JAX bicycle dynamics.  No ROS imports.

State  x = [px, py, v, psi, delta]
Control u = [a, omega]

Provides:
  make_step(params)   → JIT-compiled JAX step(x, u) → x_next
  make_rollout(step)  → JIT-compiled JAX rollout(x0, U) → X  shape (H+1, 5)
  step_numpy(x, u, params) → plain numpy step for use in Python loops
"""
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core continuous-time bicycle equations
# ---------------------------------------------------------------------------

def _deriv_jax(x: jnp.ndarray, u: jnp.ndarray, wheelbase: float) -> jnp.ndarray:
    """ẋ = f(x, u).  All args are JAX arrays (or Python scalars for wheelbase)."""
    v, psi, delta = x[2], x[3], x[4]
    a, omega = u[0], u[1]
    delta_safe = jnp.clip(delta, -0.34, 0.34)
    return jnp.array([
        v * jnp.cos(psi),
        v * jnp.sin(psi),
        a,
        v * jnp.tan(delta_safe) / wheelbase,
        omega,
    ])


# ---------------------------------------------------------------------------
# Factory: JIT-compiled step and rollout (call once at startup)
# ---------------------------------------------------------------------------

def make_step(params):
    """
    Returns a JIT-compiled  step(x, u) → x_next  function.
    Captures dt, wheelbase, and limits as Python constants so JAX never
    retraces on value changes.
    """
    dt = float(params.dt)
    wb = float(params.wheelbase_m)
    u_min = jnp.array([params.a_min, params.omega_min])
    u_max = jnp.array([params.a_max, params.omega_max])
    x_lo  = jnp.array([-jnp.inf, -jnp.inf, params.v_min,  -jnp.inf, params.delta_min])
    x_hi  = jnp.array([ jnp.inf,  jnp.inf, params.v_max,   jnp.inf, params.delta_max])

    @jax.jit
    def step(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        u_c = jnp.clip(u, u_min, u_max)
        x_next = x + dt * _deriv_jax(x, u_c, wb)
        # Wrap heading to (-pi, pi]
        psi = jnp.arctan2(jnp.sin(x_next[3]), jnp.cos(x_next[3]))
        x_next = x_next.at[3].set(psi)
        return jnp.clip(x_next, x_lo, x_hi)

    return step


def make_rollout(step_fn):
    """
    Returns  rollout(x0, U) → X  where X has shape (H+1, 5).
    Uses jax.lax.scan for full JIT-compilability and auto-diff support.
    """
    @jax.jit
    def rollout(x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """x0: (5,),  U: (H, 2)  →  X: (H+1, 5)"""
        def body(x_k, u_k):
            x_next = step_fn(x_k, u_k)
            return x_next, x_next

        _, traj = jax.lax.scan(body, x0, U)          # traj: (H, 5)
        return jnp.concatenate([x0[None, :], traj], axis=0)   # (H+1, 5)

    return rollout


# ---------------------------------------------------------------------------
# Plain numpy step — used inside the Python forward-pass loop to avoid
# per-call JAX dispatch overhead.
# ---------------------------------------------------------------------------

def step_numpy(x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
    """Forward Euler step, fully in numpy.  Returns new state (5,)."""
    dt = params.dt
    wb = params.wheelbase_m
    a   = float(np.clip(u[0], params.a_min,     params.a_max))
    om  = float(np.clip(u[1], params.omega_min, params.omega_max))
    v, psi, delta = float(x[2]), float(x[3]), float(x[4])
    delta_s = float(np.clip(delta, -0.34, 0.34))
    xn = np.array([
        x[0] + dt * v * np.cos(psi),
        x[1] + dt * v * np.sin(psi),
        np.clip(v + dt * a, params.v_min, params.v_max),
        x[3] + dt * v * np.tan(delta_s) / wb,
        np.clip(delta + dt * om, params.delta_min, params.delta_max),
    ], dtype=float)
    # Wrap heading
    xn[3] = float((xn[3] + np.pi) % (2.0 * np.pi) - np.pi)
    return xn
