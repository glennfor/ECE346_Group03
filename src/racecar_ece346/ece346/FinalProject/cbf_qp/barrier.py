from dataclasses import dataclass

import numpy as np

from .backup_policy import brake_and_recenter
from .dynamics import rollout
from .margins import MarginContext, margin_total


@dataclass
class BarrierResult:
    value: float
    gradient: np.ndarray
    binding_constraint: str
    trajectory: np.ndarray
    margins: np.ndarray


def backup_rollout(x: np.ndarray, ctx: MarginContext) -> np.ndarray:
    return rollout(
        x,
        lambda y: brake_and_recenter(y, ctx.lane, ctx.params),
        ctx.params,
    )


def implicit_barrier_value(x: np.ndarray, ctx: MarginContext) -> tuple:
    traj = backup_rollout(x, ctx)
    margin_values = []
    labels = []
    for state in traj:
        value, label = margin_total(state, ctx)
        margin_values.append(value)
        labels.append(label)

    margins = np.asarray(margin_values, dtype=float)
    idx = int(np.argmin(margins))
    return float(margins[idx]), labels[idx], traj, margins


def implicit_barrier_grad(x: np.ndarray, ctx: MarginContext) -> np.ndarray:
    h0, _, _, _ = implicit_barrier_value(x, ctx)
    grad = np.zeros(5, dtype=float)
    eps = ctx.params.grad_eps

    for i in range(5):
        x_pert = np.asarray(x, dtype=float).copy()
        x_pert[i] += eps
        h_pert, _, _, _ = implicit_barrier_value(x_pert, ctx)
        grad[i] = (h_pert - h0) / eps

    return grad


def evaluate_barrier(x: np.ndarray, ctx: MarginContext) -> BarrierResult:
    value, label, traj, margins = implicit_barrier_value(x, ctx)
    gradient = implicit_barrier_grad(x, ctx)
    return BarrierResult(value, gradient, label, traj, margins)
