"""
Kinematic bicycle model for the 1/10-scale ECE346 truck.

State:   x = [px, py, v, psi, delta]
           px, py  — rear-axle position (m)
           v       — forward speed (m/s)
           psi     — heading (rad, wrapped to (-π, π])
           delta   — current steering angle (rad)

Control: u = [a, omega]
           a     — longitudinal acceleration (m/s²)
           omega — steering-angle rate (rad/s)

The reason delta is a state (not a control) is that the truck has a finite
steering-rate limit omega, which is physically significant.
"""
from typing import Callable

import numpy as np

from .config import CbfParams


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def clip_control(u: np.ndarray, p: CbfParams) -> np.ndarray:
    return np.array([
        np.clip(u[0], p.a_min, p.a_max),
        np.clip(u[1], p.omega_min, p.omega_max),
    ], dtype=float)


def continuous_dynamics(x: np.ndarray, u: np.ndarray, p: CbfParams) -> np.ndarray:
    """ẋ = f_c(x, u).  Returns the 5-vector time-derivative."""
    _, _, v, psi, delta = x
    a, omega = clip_control(u, p)
    return np.array([
        v * np.cos(psi),
        v * np.sin(psi),
        a,
        v * np.tan(delta) / p.wheelbase_m,
        omega,
    ], dtype=float)


def step(x: np.ndarray, u: np.ndarray, p: CbfParams) -> np.ndarray:
    """Forward Euler step: x_{k+1} = x_k + dt * f_c(x_k, u_k)."""
    x_next = np.asarray(x, dtype=float) + p.dt * continuous_dynamics(x, u, p)
    x_next[2] = float(np.clip(x_next[2], p.v_min, p.v_max))
    x_next[3] = wrap_angle(float(x_next[3]))
    x_next[4] = float(np.clip(x_next[4], p.delta_min, p.delta_max))
    return x_next


def control_jacobian(p: CbfParams) -> np.ndarray:
    """
    B = ∂f(x,u)/∂u evaluated analytically for the Euler bicycle model.
    Shape (5, 2).  Only rows 2 (accel affects v) and 4 (omega affects delta)
    are non-zero: B[2,0] = dt, B[4,1] = dt.
    """
    B = np.zeros((5, 2), dtype=float)
    B[2, 0] = p.dt
    B[4, 1] = p.dt
    return B


def rollout(
    x0: np.ndarray,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    p: CbfParams,
) -> np.ndarray:
    """
    Simulate policy_fn from x0 for p.horizon_H steps.
    Returns (H+1, 5) array of states (includes the initial state at row 0).
    """
    H = p.horizon_H
    traj = np.zeros((H + 1, 5), dtype=float)
    traj[0] = np.asarray(x0, dtype=float)
    for k in range(H):
        u = policy_fn(traj[k])
        traj[k + 1] = step(traj[k], u, p)
    return traj
