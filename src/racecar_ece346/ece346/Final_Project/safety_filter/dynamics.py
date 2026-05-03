from typing import Callable

import numpy as np

from .config import SafetyFilterParams


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def clip_control(u: np.ndarray, params: SafetyFilterParams) -> np.ndarray:
    return np.array(
        [
            np.clip(u[0], params.a_min, params.a_max),
            np.clip(u[1], params.omega_min, params.omega_max),
        ],
        dtype=float,
    )


def continuous_dynamics(
    x: np.ndarray,
    u: np.ndarray,
    params: SafetyFilterParams,
) -> np.ndarray:
    _, _, v, psi, delta = x
    a, omega = clip_control(u, params)
    return np.array(
        [
            v * np.cos(psi),
            v * np.sin(psi),
            a,
            v * np.tan(delta) / params.wheelbase_m,
            omega,
        ],
        dtype=float,
    )


def step(x: np.ndarray, u: np.ndarray, params: SafetyFilterParams, dt: float = None) -> np.ndarray:
    dt = params.dt if dt is None else dt
    x_next = np.asarray(x, dtype=float) + dt * continuous_dynamics(x, u, params)
    x_next[2] = float(np.clip(x_next[2], params.v_min, params.v_max))
    x_next[3] = wrap_angle(float(x_next[3]))
    x_next[4] = float(np.clip(x_next[4], params.delta_min, params.delta_max))
    return x_next


def control_jacobian(params: SafetyFilterParams, dt: float = None) -> np.ndarray:
    dt = params.dt if dt is None else dt
    jac = np.zeros((5, 2), dtype=float)
    jac[2, 0] = dt
    jac[4, 1] = dt
    return jac


def rollout(
    x0: np.ndarray,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    params: SafetyFilterParams,
    horizon: int = None,
) -> np.ndarray:
    horizon = params.horizon_H if horizon is None else horizon
    traj = np.zeros((horizon + 1, 5), dtype=float)
    traj[0] = np.asarray(x0, dtype=float)

    for k in range(horizon):
        u = policy_fn(traj[k])
        traj[k + 1] = step(traj[k], u, params)

    return traj


def _step_jacobian(
    x: np.ndarray,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    params: SafetyFilterParams,
    eps: float = 1e-4,
):
    """
    Numerically compute J = dx_{k+1}/dx_k and also return x_next so the caller
    does not have to repeat the base-case evaluation.

    The policy is re-evaluated at each perturbed state so the Jacobian captures
    the feedback in the backup controller (its steering reacts to heading error
    and lateral error, both of which change when x_k is perturbed).
    """
    x0 = np.asarray(x, dtype=float)
    x_next = step(x0, policy_fn(x0), params)
    J = np.zeros((5, 5), dtype=float)
    for i in range(5):
        xp = x0.copy()
        xp[i] += eps
        J[:, i] = (step(xp, policy_fn(xp), params) - x_next) / eps
    return J, x_next


def rollout_with_sensitivity(
    x0: np.ndarray,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    params: SafetyFilterParams,
    horizon: int = None,
):
    """
    Roll out policy_fn for `horizon` steps and accumulate the sensitivity
    Jacobians Q_k = dx_k/dx_0 at every step.

    Q_0 = I  (x_0 is x_0, no sensitivity)
    Q_{k+1} = J_k @ Q_k  where J_k = dx_{k+1}/dx_k through (policy + step)

    Returns (traj, Q_list) where traj[k] is the state at step k and
    Q_list[k] is the corresponding 5×5 sensitivity matrix.
    """
    horizon = params.horizon_H if horizon is None else horizon
    traj = np.zeros((horizon + 1, 5), dtype=float)
    traj[0] = np.asarray(x0, dtype=float)

    Q = np.eye(5, dtype=float)
    Q_list = [Q.copy()]

    for k in range(horizon):
        J_k, x_next = _step_jacobian(traj[k], policy_fn, params)
        traj[k + 1] = x_next
        Q = J_k @ Q
        # Clamp Frobenius norm to prevent numerical blow-up from accumulated
        # Jacobian products — preserves direction, kills magnitude explosion.
        q_scale = np.linalg.norm(Q, ord='fro')
        if q_scale > 1e2:
            Q = Q * (1e2 / q_scale)
        Q_list.append(Q.copy())

    return traj, Q_list

