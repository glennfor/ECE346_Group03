from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

from .backup_policy import brake_and_recenter
from .config import SafetyFilterParams
from .dynamics import continuous_dynamics, control_jacobian, rollout, rollout_with_sensitivity
from .margins import MarginContext, margin_grad, margin_total


@dataclass
class BarrierResult:
    value: float
    gradient: np.ndarray           # Q(t*)ᵀ @ ∇m(x_{t*})  — correct sensitivity gradient
    binding_constraint: str
    trajectory: np.ndarray
    margins: np.ndarray
    per_step_constraints: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    # Each element is (g_k, c_k) where the QP constraint is  g_k @ u >= c_k.
    # One entry per rollout step k=0..H.  The QP enforces all of them simultaneously.


def implicit_barrier_value(x: np.ndarray, ctx: MarginContext) -> tuple:
    """Value-only barrier evaluation used by the recovery path."""
    traj = rollout(
        x,
        lambda y: brake_and_recenter(y, ctx.lane, ctx.params),
        ctx.params,
    )
    margin_values = []
    labels = []
    for state in traj:
        v, label = margin_total(state, ctx)
        margin_values.append(v)
        labels.append(label)
    margins = np.asarray(margin_values, dtype=float)
    idx = int(np.argmin(margins))
    return float(margins[idx]), labels[idx], traj, margins


def evaluate_barrier(x: np.ndarray, ctx: MarginContext) -> BarrierResult:
    """
    Full barrier evaluation with correct gradient and per-step QP constraints.

    The barrier value  h(x) = min_k m(x_k)  is unchanged.

    The gradient is now  ∇h = Q(t*)ᵀ ∇m(x_{t*})  (paper Eq. 9) instead of
    the broken finite-difference of min().  Q(t*) = dx_{t*}/dx_0 is the
    sensitivity Jacobian accumulated through the backup trajectory.

    The per_step_constraints list contains one (g_k, c_k) pair per step so the
    QP can enforce the CBF condition at ALL backup steps simultaneously (paper
    Eq. 12), not only the binding step.
    """
    policy_fn = lambda y: brake_and_recenter(y, ctx.lane, ctx.params)
    traj, Q_list = rollout_with_sensitivity(x, policy_fn, ctx.params)

    margins = []
    labels = []
    for state in traj:
        m, label = margin_total(state, ctx)
        margins.append(m)
        labels.append(label)
    margins = np.asarray(margins, dtype=float)

    t_star = int(np.argmin(margins))
    h = float(margins[t_star])
    binding_label = labels[t_star]

    # Correct gradient: back-propagate ∇m at the binding state through Q(t*).
    grad_m_star = margin_grad(traj[t_star], ctx)
    gradient = Q_list[t_star].T @ grad_m_star

    # Build per-step QP constraints.
    # The continuous-time CBF condition is  ∇h · (f + g·u) ≥ -λ·h.
    # Discretising with x_{k+1} = x + dt·(f + g·u):
    #   (Bᵀ ψ_k) · u ≥  -λ·m_k  -  dt · ψ_k · f(x)
    # where  ψ_k = Q_k^T ∇m(x_k)  and  B = ∂x_1/∂u.
    B = control_jacobian(ctx.params)                       # shape (5, 2)
    f0 = continuous_dynamics(x, np.zeros(2), ctx.params)  # drift at current state

    binding_margin = float(margins[t_star])
    per_step_constraints: List[Tuple[np.ndarray, float]] = []

    # Only enforce constraints at steps whose margin is close to the minimum.
    # Steps with large margins produce an RHS so negative the constraint is
    # always satisfied and would just add noise to the QP.
    active_margin_window = max(abs(binding_margin) + 0.15, 0.15)

    for k, m_k in enumerate(margins):
        if float(m_k) > binding_margin + active_margin_window:
            continue  # constraint is trivially satisfied — skip
        grad_m_k = margin_grad(traj[k], ctx) if k != t_star else grad_m_star
        psi_k = Q_list[k].T @ grad_m_k
        g_k = B.T @ psi_k
        c_k = (-ctx.params.lambda_cbf * float(m_k)
               - float(ctx.params.dt * float(psi_k @ f0)))
        per_step_constraints.append((g_k, float(c_k)))

    return BarrierResult(
        value=h,
        gradient=gradient,
        binding_constraint=binding_label,
        trajectory=traj,
        margins=margins,
        per_step_constraints=per_step_constraints,
    )
