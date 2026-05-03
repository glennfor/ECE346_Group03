"""
CBF-QP solver — closed-form solution to the 2-D box-constrained half-space QP.

Problem:
    min  (1/2) * (u - u_h)^T R (u - u_h)
    s.t. g^T u >= c           (CBF linearized constraint)
         u_min <= u <= u_max  (control box)

With R = diag(w_a, w_omega) and 2 decision variables, this has an analytical
solution.  We enumerate the feasible vertices, edge intersections, and the
weighted projection onto the CBF hyperplane, then pick the best.

No OSQP or scipy dependency needed — this is fast (microseconds) and robust.
"""
from dataclasses import dataclass

import numpy as np

from .config import CbfParams


@dataclass
class QPResult:
    control: np.ndarray   # (2,) optimal control [a, omega]
    status: str           # "optimal" or "infeasible"


def solve(
    u_h: np.ndarray,
    g: np.ndarray,
    c: float,
    p: CbfParams,
) -> QPResult:
    """
    Find the control closest to u_h that satisfies g^T u >= c and u in box.

    u_h   — (2,) human desired control [a, omega]
    g     — (2,) CBF constraint gradient direction
    c     — CBF constraint RHS scalar
    p     — CbfParams for box bounds and cost weights
    """
    u_min = np.array([p.a_min, p.omega_min], dtype=float)
    u_max = np.array([p.a_max, p.omega_max], dtype=float)
    weights = np.array([p.qp_R_accel, p.qp_R_omega], dtype=float)
    u_ref = np.clip(np.asarray(u_h, dtype=float), u_min, u_max)
    g = np.asarray(g, dtype=float)

    # If g is zero, the constraint is either trivially satisfied (c <= 0)
    # or infeasible (c > 0, no control can help).
    if np.linalg.norm(g) < 1e-10:
        if c <= 0.0:
            return QPResult(u_ref, "optimal")
        return QPResult(u_ref, "infeasible")

    # If the human input already satisfies the constraint, no change needed.
    if float(g @ u_ref) >= c - 1e-9:
        return QPResult(u_ref, "optimal")

    # Build the candidate set: all points that could be optimal.
    # These are: (a) box corners that satisfy the constraint,
    #            (b) edge-box intersection points on the CBF boundary,
    #            (c) the weighted projection of u_ref onto the CBF hyperplane.
    candidates = []

    # Box corners
    for u in [
        np.array([u_min[0], u_min[1]]),
        np.array([u_min[0], u_max[1]]),
        np.array([u_max[0], u_min[1]]),
        np.array([u_max[0], u_max[1]]),
    ]:
        if float(g @ u) >= c - 1e-9:
            candidates.append(u)

    # Points where the CBF boundary intersects each box edge.
    for axis in (0, 1):
        other = 1 - axis
        if abs(g[other]) < 1e-10:
            continue
        for bound in (u_min[axis], u_max[axis]):
            val = (c - g[axis] * bound) / g[other]
            if u_min[other] - 1e-9 <= val <= u_max[other] + 1e-9:
                u_cand = np.zeros(2)
                u_cand[axis] = bound
                u_cand[other] = np.clip(val, u_min[other], u_max[other])
                candidates.append(u_cand)

    # Weighted projection of u_ref onto the hyperplane g^T u = c.
    # With cost (u - u_h)^T diag(w) (u - u_h), the minimum on the hyperplane is:
    #   u* = u_h + ((c - g^T u_h) / (g^T (1/w) g)) * (g / w)
    denom = float(np.sum(g * g / weights))
    if denom > 1e-10:
        proj = u_ref + ((c - float(g @ u_ref)) / denom) * (g / weights)
        if np.all(proj >= u_min - 1e-9) and np.all(proj <= u_max + 1e-9):
            candidates.append(proj)

    if not candidates:
        return QPResult(u_ref, "infeasible")

    def cost(u):
        d = u - u_ref
        return float(np.sum(weights * d * d))

    best = min(candidates, key=cost)
    return QPResult(np.clip(best, u_min, u_max), "optimal")
