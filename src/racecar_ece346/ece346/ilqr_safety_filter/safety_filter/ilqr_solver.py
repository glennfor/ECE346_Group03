"""
ILQRSolver — adapts the Lab1 backward/forward ILQR loop to the safety filter.

Key design choices:
  • Row-major shape convention: X (H+1, 5), U (H, 2)  (spec convention)
  • JAX JIT + vmap for dynamics Jacobians and cost derivatives
  • Pure numpy backward-pass sweep (mirrors Lab1 exactly, just transposed indices)
  • LM regularization: identical logic to Lab1's backward_pass()
  • Warm-start: shift previous U forward by 1 step on each solve
  • warmup(): call once at node startup to trigger all JAX JIT compilations

No ROS imports.
"""
from typing import Optional, Tuple, Dict
import time

import numpy as np
import jax
import jax.numpy as jnp

from .config import ILQRSafetyParams
from .jax_context import ILQRContext
from .dynamics import make_step, make_rollout, step_numpy
from .ilqr_costs import stage_cost, terminal_cost


class ILQRSolver:
    """
    ILQR solver for the ADAS safety filter fallback planner.

    Trajectory convention:  X: (H+1, 5),  U: (H, 2)
    """

    def __init__(self, params: ILQRSafetyParams):
        self.H          = params.horizon_H
        self.dt         = params.dt
        self.max_iters  = params.ilqr_max_iters
        self.tol        = params.ilqr_tol
        self.lm_init    = params.lm_init
        self.lm_min     = params.lm_min
        self.lm_max     = params.lm_max
        self.lm_up      = params.lm_scale_up
        self.lm_down    = params.lm_scale_down
        self._params    = params

        # JIT-compiled step and rollout (closures capture Python constants)
        self._step_jax = make_step(params)
        self._rollout  = make_rollout(self._step_jax)

        # Dynamics Jacobians: vmap(jacfwd) over time steps
        _step_jac = jax.jacfwd(self._step_jax, argnums=(0, 1))
        self._dyn_jac = jax.jit(jax.vmap(_step_jac, in_axes=(0, 0)))

        # Cost derivatives (vmapped over the H trajectory steps)
        self._sc_grad    = jax.jit(jax.vmap(
            jax.grad(stage_cost, argnums=(0, 1)), in_axes=(0, 0, None)))
        self._sc_hess_xx = jax.jit(jax.vmap(
            jax.hessian(stage_cost, argnums=0), in_axes=(0, 0, None)))
        self._sc_hess_uu = jax.jit(jax.vmap(
            jax.hessian(stage_cost, argnums=1), in_axes=(0, 0, None)))
        # ∂²c/∂u∂x  →  shape (2, 5) per step
        self._sc_hess_ux = jax.jit(jax.vmap(
            jax.jacfwd(jax.grad(stage_cost, argnums=1), argnums=0),
            in_axes=(0, 0, None)))

        # Terminal cost derivatives (single state)
        self._tc_grad = jax.jit(jax.grad(terminal_cost, argnums=0))
        self._tc_hess = jax.jit(jax.hessian(terminal_cost, argnums=0))

        # Warm-start state
        self._prev_U: Optional[np.ndarray] = None

        # Levenberg–Marquardt: track across solves
        self._lm = float(params.lm_init)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        ctx: ILQRContext,
        U_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run ILQR from x0.

        Returns:
            X      : (H+1, 5) numpy array of states
            U      : (H, 2)   numpy array of controls
            info   : dict with 'converged', 'iters', 'cost'
        """
        # Initialise control sequence
        if U_init is not None:
            U = np.asarray(U_init, dtype=np.float64)
        elif self._prev_U is not None:
            U = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
        else:
            U = np.zeros((self.H, 2), dtype=np.float64)

        U = np.clip(U,
                    [self._params.a_min,     self._params.omega_min],
                    [self._params.a_max,     self._params.omega_max])

        x0_jax = jnp.array(x0, dtype=jnp.float32)
        U_jax  = jnp.array(U,  dtype=jnp.float32)

        X = np.asarray(self._rollout(x0_jax, U_jax), dtype=np.float64)
        J = self._traj_cost(X, U, ctx)

        lm        = self._lm
        converged = False
        iters     = 0

        for i in range(self.max_iters):
            iters = i + 1
            K, k, lm, bp_ok = self._backward_pass(X, U, ctx, lm)
            if not bp_ok:
                # Backward pass failed (Q_uu not PD at max LM) — return best so far
                break

            improved = False
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01]:
                X_new, U_new = self._forward_pass(x0, X, U, K, k, alpha)
                J_new = self._traj_cost(X_new, U_new, ctx)
                if np.isfinite(J_new) and J_new < J:
                    dJ = J - J_new
                    X, U, J = X_new, U_new, J_new
                    lm = max(self.lm_min, lm / self.lm_down)
                    improved = True
                    if dJ < self.tol:
                        converged = True
                    break

            if not improved:
                lm = min(self.lm_max, lm * self.lm_up)
                if lm >= self.lm_max:
                    break
            if converged:
                break

        self._lm    = lm
        self._prev_U = U
        return X, U, {'converged': converged, 'iters': iters, 'cost': float(J)}

    def warm_start_solve(
        self,
        x0: np.ndarray,
        prev_U: Optional[np.ndarray],
        ctx: ILQRContext,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Shift a previous control sequence by 1 step, then solve."""
        if prev_U is not None:
            U_init = np.vstack([prev_U[1:], prev_U[-1:]])
        else:
            U_init = None
        return self.solve(x0, ctx, U_init=U_init)

    def warmup(self, x0_dummy: np.ndarray, ctx_dummy: ILQRContext):
        """
        Trigger all JAX JIT compilations.  Call once at node startup
        before the control loop begins.
        """
        self.solve(x0_dummy, ctx_dummy)   # cold compile
        self.solve(x0_dummy, ctx_dummy)   # warm-start path

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _traj_cost(self, X: np.ndarray, U: np.ndarray, ctx: ILQRContext) -> float:
        """Sum stage costs over H steps plus terminal cost."""
        X_jax = jnp.array(X, dtype=jnp.float32)
        U_jax = jnp.array(U, dtype=jnp.float32)
        # vmapped stage cost over H steps
        sc_fn = jax.jit(jax.vmap(stage_cost, in_axes=(0, 0, None)))
        costs = sc_fn(X_jax[:self.H], U_jax, ctx)
        J = float(jnp.sum(costs)) + float(terminal_cost(X_jax[self.H], ctx))
        return J

    def _backward_pass(
        self,
        X: np.ndarray,
        U: np.ndarray,
        ctx: ILQRContext,
        lm: float,
    ) -> Tuple:
        """
        Backward sweep — pure numpy, mirrors Lab1 backward_pass() exactly.
        Dimensions (row-major): X (H+1, 5), U (H, 2).
        Returns (K: (H,2,5), k: (H,2), lm, ok).
        """
        X_jax = jnp.array(X[:self.H], dtype=jnp.float32)
        U_jax = jnp.array(U,          dtype=jnp.float32)

        # Cost derivatives at each of the H stage steps
        grads        = self._sc_grad(X_jax, U_jax, ctx)      # tuple ((H,5), (H,2))
        q  = np.asarray(grads[0], dtype=np.float64)          # (H, 5)
        r  = np.asarray(grads[1], dtype=np.float64)          # (H, 2)
        Q_xx = np.asarray(self._sc_hess_xx(X_jax, U_jax, ctx), dtype=np.float64)  # (H,5,5)
        R_uu = np.asarray(self._sc_hess_uu(X_jax, U_jax, ctx), dtype=np.float64)  # (H,2,2)
        H_ux = np.asarray(self._sc_hess_ux(X_jax, U_jax, ctx), dtype=np.float64)  # (H,2,5)

        # Terminal cost derivatives
        q_T = np.asarray(
            self._tc_grad(jnp.array(X[self.H], dtype=jnp.float32), ctx),
            dtype=np.float64)   # (5,)
        Q_T = np.asarray(
            self._tc_hess(jnp.array(X[self.H], dtype=jnp.float32), ctx),
            dtype=np.float64)   # (5,5)

        # Dynamics Jacobians
        AB = self._dyn_jac(X_jax, U_jax)               # tuple ((H,5,5), (H,5,2))
        As = np.asarray(AB[0], dtype=np.float64)        # (H, 5, 5)
        Bs = np.asarray(AB[1], dtype=np.float64)        # (H, 5, 2)

        # Backward recursion (identical to Lab1; indices transposed to row-major)
        P   = Q_T.copy()                                # (5, 5) value-fn Hessian
        p   = q_T.copy()                                # (5,)   value-fn gradient
        Ks  = np.zeros((self.H, 2, 5), dtype=np.float64)
        ks  = np.zeros((self.H, 2),    dtype=np.float64)

        for t in range(self.H - 1, -1, -1):
            A = As[t]   # (5, 5)
            B = Bs[t]   # (5, 2)

            Q_x  = q[t] + A.T @ p                          # (5,)
            Q_u  = r[t] + B.T @ p                          # (2,)
            Q_xx_t = Q_xx[t] + A.T @ P @ A                 # (5,5)
            Q_uu_t = R_uu[t] + B.T @ P @ B                 # (2,2)
            Q_ux_t = H_ux[t] + B.T @ P @ A                 # (2,5)

            # Levenberg–Marquardt regularisation (identical to Lab1)
            reg = lm * np.eye(5)
            Q_uu_reg = R_uu[t] + B.T @ (P + reg) @ B       # (2,2)
            Q_ux_reg = H_ux[t] + B.T @ (P + reg) @ A       # (2,5)

            eigs = np.linalg.eigvalsh(Q_uu_reg)
            if not np.all(eigs > 0):
                new_lm = min(self.lm_max, lm * self.lm_up)
                return None, None, new_lm, False

            K  = -np.linalg.solve(Q_uu_reg, Q_ux_reg)      # (2,5)
            k_ = -np.linalg.solve(Q_uu_reg, Q_u)           # (2,)
            Ks[t] = K
            ks[t] = k_

            P = Q_xx_t + K.T @ Q_uu_t @ K + K.T @ Q_ux_t + Q_ux_t.T @ K
            p = Q_x    + K.T @ Q_uu_t @ k_ + K.T @ Q_u   + Q_ux_t.T @ k_

        return Ks, ks, lm, True

    def _forward_pass(
        self,
        x0: np.ndarray,
        X_bar: np.ndarray,
        U_bar: np.ndarray,
        K: np.ndarray,       # (H, 2, 5)
        k: np.ndarray,       # (H, 2)
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll forward with feedback gains.  Uses numpy step to avoid per-call
        JAX dispatch overhead in the Python loop.
        """
        X_new = np.zeros_like(X_bar)
        U_new = np.zeros_like(U_bar)
        X_new[0] = x0

        for t in range(self.H):
            dx    = X_new[t] - X_bar[t]
            # Wrap heading difference to (-pi, pi]
            dx[3] = float((dx[3] + np.pi) % (2.0 * np.pi) - np.pi)
            u     = U_bar[t] + alpha * k[t] + K[t] @ dx
            u     = np.clip(u,
                            [self._params.a_min,     self._params.omega_min],
                            [self._params.a_max,     self._params.omega_max])
            X_new[t + 1] = step_numpy(X_new[t], u, self._params)
            U_new[t]     = u

        return X_new, U_new
