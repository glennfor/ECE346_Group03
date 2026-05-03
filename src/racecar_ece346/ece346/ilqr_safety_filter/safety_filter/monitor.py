"""
SafetyMonitor — fast V̂(x, u_h) ≥ 0 certification check.

Algorithm:
  1. Simulate one step: x_next = f(x, u_h)
  2. Shift the planner's U_star by 1 step to get U_check
  3. Roll out X_check from x_next using U_check  (O(H), no ILQR)
  4. Evaluate HARD safety margins along X_check
  5. V̂ = min(margins over H+1 steps)
  6. Certified iff V̂ ≥ 0

Using a simple rollout (not a full ILQR re-solve) keeps the monitor at ~2 ms
per call, avoiding any blocking of the 20 Hz foreground control timer.

No ROS imports.
"""
from typing import Optional, Tuple
import numpy as np
import jax.numpy as jnp

from .config import ILQRSafetyParams
from .jax_context import ILQRContext
from .dynamics import make_step, make_rollout, step_numpy
from .margins import trajectory_hard_margins_jit, margin_lane, margin_obstacle, margin_kinematic


class SafetyMonitor:
    def __init__(self, params: ILQRSafetyParams):
        self._params = params
        # JIT-compiled rollout for the fast V̂ check
        self._step_jax = make_step(params)
        self._rollout   = make_rollout(self._step_jax)

    def warmup(self, x0_dummy: np.ndarray, ctx_dummy: ILQRContext):
        """Trigger JAX JIT compilation for rollout + margins."""
        U_dummy = np.zeros((self._params.horizon_H, 2), dtype=np.float32)
        X_dummy = np.asarray(
            self._rollout(jnp.array(x0_dummy, dtype=jnp.float32),
                          jnp.array(U_dummy, dtype=jnp.float32))
        )
        trajectory_hard_margins_jit(jnp.array(X_dummy, dtype=jnp.float32), ctx_dummy)

    def certify(
        self,
        x: np.ndarray,
        u_h: np.ndarray,
        ctx: ILQRContext,
        plan_U: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, str]:
        """
        Returns (is_safe, V_hat, binding_constraint_name).

        x      : current state (5,)
        u_h    : human control (2,)
        ctx    : current ILQRContext
        plan_U : U from the planner's most recent plan — used as the rollout
                 trajectory from x_next.  If None, zeros are used.
        """
        # Step 1: one-step simulation under human input
        x_next = step_numpy(x, u_h, self._params)

        # Step 2: shift planner trajectory by 1 step
        H = self._params.horizon_H
        if plan_U is not None and plan_U.shape == (H, 2):
            U_check = np.vstack([plan_U[1:], plan_U[-1:]])
        else:
            U_check = np.zeros((H, 2), dtype=np.float64)

        # Step 3: roll out X_check from x_next  (JIT'd lax.scan, ~1 ms)
        X_check = np.asarray(
            self._rollout(
                jnp.array(x_next,   dtype=jnp.float32),
                jnp.array(U_check,  dtype=jnp.float32),
            ),
            dtype=np.float64,
        )

        # Step 4: HARD margins along X_check  (JIT'd vmap, ~1 ms)
        margins = np.asarray(
            trajectory_hard_margins_jit(
                jnp.array(X_check, dtype=jnp.float32), ctx
            ),
            dtype=np.float64,
        )

        V_hat      = float(np.min(margins))
        worst_step = int(np.argmin(margins))
        binding    = self._binding_constraint(X_check[worst_step], ctx)

        return V_hat >= 0.0, V_hat, binding

    # ------------------------------------------------------------------

    def _binding_constraint(self, x: np.ndarray, ctx: ILQRContext) -> str:
        """Identify which margin is tightest at state x."""
        x_j = jnp.array(x, dtype=jnp.float32)
        ml  = float(margin_lane(x_j, ctx))
        mo  = float(margin_obstacle(x_j, ctx.obs_xy_r,  ctx.obs_valid,  ctx, 0.0))
        mt  = float(margin_obstacle(x_j, ctx.traf_xy_r, ctx.traf_valid, ctx, 0.0))
        mk  = float(margin_kinematic(x_j, ctx))
        components = {"lane": ml, "obstacle": mo, "traffic": mt, "kinematic": mk}
        return min(components, key=components.get)
