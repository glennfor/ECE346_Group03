"""
SafetyMonitor — implements the V̂(x, u_h) ≥ 0 certification check.

Algorithm (§5 of ILQR_SAFETY_FILTER_SPEC.md):
  1. Simulate one step: x_next = f(x, u_h)
  2. Run warm-started ILQR from x_next
  3. Evaluate HARD safety margins along the resulting trajectory
  4. V̂ = min(margins over H+1 steps)
  5. Certified iff V̂ ≥ 0

Holds its own ILQRSolver instance (separate from the planner's) so the
two solvers warm-start independently.

No ROS imports.
"""
from typing import Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .config import ILQRSafetyParams
from .jax_context import ILQRContext
from .dynamics import step_numpy
from .ilqr_solver import ILQRSolver
from .margins import trajectory_hard_margins_jit, margin_lane, margin_obstacle, margin_kinematic


class SafetyMonitor:
    def __init__(self, params: ILQRSafetyParams):
        self._params  = params
        # Second ILQR solver — warm-starts from the planner's plan
        self._solver  = ILQRSolver(params)

    @property
    def solver(self) -> ILQRSolver:
        """Expose the internal solver so the node can warm it up."""
        return self._solver

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
        plan_U : U from the planner's most recent plan — used as warm start
        """
        # Step 1: one-step simulation under human input
        x_next = step_numpy(x, u_h, self._params)

        # Step 2: ILQR from x_next, warm-started from current plan
        X_star, U_star, info = self._solver.warm_start_solve(x_next, plan_U, ctx)

        # Step 3: HARD margins along X_star  (H+1 states)
        margins = np.asarray(
            trajectory_hard_margins_jit(jnp.array(X_star, dtype=jnp.float32), ctx),
            dtype=np.float64,
        )

        V_hat      = float(np.min(margins))
        worst_step = int(np.argmin(margins))
        binding    = self._binding_constraint(X_star[worst_step], ctx)

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
