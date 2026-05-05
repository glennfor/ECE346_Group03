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
