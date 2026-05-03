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

    # Weighted projection onto the CBF boundary, then clipped candidates cover any box violation.
    denom = float(np.sum(g * g / weights))
    if denom > 1e-10:
        projected = u_ref + ((c - float(g @ u_ref)) / denom) * (g / weights)
        if np.all(projected >= u_min - 1e-9) and np.all(projected <= u_max + 1e-9):
            candidates.append(projected)

    if not candidates:
        return QPResult(u_ref, "infeasible")

    best = min(candidates, key=lambda u: _weighted_distance(u, u_ref, weights))
    return QPResult(np.clip(best, u_min, u_max), "optimal")


def solve_box_multi_halfspace_qp(
    u_human: np.ndarray,
    constraints: list,
    params: "SafetyFilterParams",
) -> QPResult:
    """
    Solve  min_u ||u - u_human||²_W
    subject to  g_k @ u >= c_k  for every (g_k, c_k) in `constraints`
                u_min <= u <= u_max

    With only 2 decision variables (acceleration, steering rate) this is solved
    analytically by enumerating candidate points: the human command, weighted
    projections onto each constraint boundary, pairwise constraint intersections,
    and box corners.  The feasible candidate closest to u_human is returned.

    This is the paper's multi-constraint QP (Eq. 12): one CBF condition per
    rollout step instead of only at the binding step.
    """
    if not constraints:
        return solve_box_halfspace_qp(u_human, np.zeros(2), -1.0, params)

    u_min = np.array([params.a_min, params.omega_min], dtype=float)
    u_max = np.array([params.a_max, params.omega_max], dtype=float)
    weights = np.array([params.qp_R_accel, params.qp_R_omega], dtype=float)
    u_ref = np.clip(np.asarray(u_human, dtype=float), u_min, u_max)

    G = np.array([g for g, _ in constraints], dtype=float)   # (K, 2)
    c_vec = np.array([c for _, c in constraints], dtype=float)  # (K,)

    def feasible(u: np.ndarray) -> bool:
        return (
            np.all(G @ u >= c_vec - 1e-9)
            and np.all(u >= u_min - 1e-9)
            and np.all(u <= u_max + 1e-9)
        )

    if feasible(u_ref):
        return QPResult(u_ref, "optimal")

    candidates = []

    # Box corners.
    for v in _candidate_vertices(u_min, u_max):
        if feasible(v):
            candidates.append(v)

    # Weighted projection of u_ref onto each constraint boundary, then box-clip.
    for k in range(len(c_vec)):
        g_k = G[k]
        denom = float(np.sum(g_k * g_k / weights))
        if denom < 1e-10:
            continue
        proj = u_ref + ((c_vec[k] - float(g_k @ u_ref)) / denom) * (g_k / weights)
        proj = np.clip(proj, u_min, u_max)
        if feasible(proj):
            candidates.append(proj)

    # Intersection of each constraint boundary with each box edge.
    for k in range(len(c_vec)):
        g_k = G[k]
        for axis in (0, 1):
            other = 1 - axis
            if abs(g_k[other]) < 1e-10:
                continue
            for bound in (u_min[axis], u_max[axis]):
                val = (c_vec[k] - g_k[axis] * bound) / g_k[other]
                if u_min[other] - 1e-9 <= val <= u_max[other] + 1e-9:
                    cand = np.zeros(2, dtype=float)
                    cand[axis] = bound
                    cand[other] = np.clip(val, u_min[other], u_max[other])
                    if feasible(cand):
                        candidates.append(cand)

    # Pairwise intersections of constraint boundaries (box-clipped).
    K = len(c_vec)
    for j in range(K):
        for k in range(j + 1, K):
            A = np.array([G[j], G[k]], dtype=float)
            b = np.array([c_vec[j], c_vec[k]], dtype=float)
            if abs(np.linalg.det(A)) < 1e-10:
                continue
            try:
                u_int = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            u_int = np.clip(u_int, u_min, u_max)
            if feasible(u_int):
                candidates.append(u_int)

    if not candidates:
        return QPResult(u_ref, "infeasible")

    best = min(candidates, key=lambda u: _weighted_distance(u, u_ref, weights))
    return QPResult(np.clip(best, u_min, u_max), "optimal")

