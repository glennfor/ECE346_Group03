from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SafetyFilterParams


@dataclass
class QPResult:
    control: np.ndarray
    status: str


def _weighted_distance(u: np.ndarray, u_ref: np.ndarray, weights: np.ndarray) -> float:
    diff = u - u_ref
    return float(np.sum(weights * diff * diff))


def _candidate_vertices(u_min: np.ndarray, u_max: np.ndarray) -> list:
    return [
        np.array([u_min[0], u_min[1]], dtype=float),
        np.array([u_min[0], u_max[1]], dtype=float),
        np.array([u_max[0], u_min[1]], dtype=float),
        np.array([u_max[0], u_max[1]], dtype=float),
    ]


def solve_box_halfspace_qp(
    u_human: np.ndarray,
    g: np.ndarray,
    c: float,
    params: SafetyFilterParams,
) -> QPResult:
    u_min = np.array([params.a_min, params.omega_min], dtype=float)
    u_max = np.array([params.a_max, params.omega_max], dtype=float)
    weights = np.array([params.qp_R_accel, params.qp_R_omega], dtype=float)
    u_ref = np.clip(np.asarray(u_human, dtype=float), u_min, u_max)
    g = np.asarray(g, dtype=float)

    if np.linalg.norm(g) < 1e-10:
        if 0.0 >= c:
            return QPResult(u_ref, "optimal")
        return QPResult(u_ref, "infeasible")

    if float(g @ u_ref) >= c - 1e-9:
        return QPResult(u_ref, "optimal")

    candidates = []

    for vertex in _candidate_vertices(u_min, u_max):
        if float(g @ vertex) >= c - 1e-9:
            candidates.append(vertex)

    for axis in (0, 1):
        other = 1 - axis
        for bound in (u_min[axis], u_max[axis]):
            if abs(g[other]) < 1e-10:
                continue
            value = (c - g[axis] * bound) / g[other]
            if u_min[other] - 1e-9 <= value <= u_max[other] + 1e-9:
                candidate = np.zeros(2, dtype=float)
                candidate[axis] = bound
                candidate[other] = np.clip(value, u_min[other], u_max[other])
                candidates.append(candidate)

    denom = float(np.sum(g * g / weights))
    if denom > 1e-10:
        projected = u_ref + ((c - float(g @ u_ref)) / denom) * (g / weights)
        if np.all(projected >= u_min - 1e-9) and np.all(projected <= u_max + 1e-9):
            candidates.append(projected)

    if not candidates:
        return QPResult(u_ref, "infeasible")

    best = min(candidates, key=lambda u: _weighted_distance(u, u_ref, weights))
    return QPResult(np.clip(best, u_min, u_max), "optimal")
