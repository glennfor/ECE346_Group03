"""
LaneContext — a snapshot of nearby lane geometry, stored as numpy arrays
so the backup policy and margin functions can be called without hitting
the Lanelet2 library on every step.

LaneletContextBuilder queries the Lanelet2 map once and builds a LaneContext.
It rebuilds when the truck moves more than lane_context_rebuild_distance_m
(default 0.5 m) or turns more than lane_context_rebuild_yaw_rad (default 0.5 rad).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .dynamics import wrap_angle


@dataclass
class LaneSample:
    """Projection of a position onto the nearest centerline point."""
    point: np.ndarray         # nearest centerline point (2,)
    tangent: float            # heading of centerline at that point (rad)
    signed_lateral_error: float   # positive = truck is to the LEFT of centerline
    width_left: float         # distance from centerline to left boundary
    width_right: float        # distance from centerline to right boundary


@dataclass
class LaneContext:
    """Discretized lane centerline + boundary widths."""
    centerline: np.ndarray    # (N, 2)
    width_left: np.ndarray    # (N,)  — left clearance from centerline
    width_right: np.ndarray   # (N,)  — right clearance from centerline
    tangent: np.ndarray       # (N,)  — heading at each centerline point

    @classmethod
    def from_centerline(
        cls,
        centerline: np.ndarray,
        width_left: np.ndarray,
        width_right: np.ndarray,
    ) -> "LaneContext":
        centerline = np.asarray(centerline, dtype=float)
        if centerline.shape[0] < 2:
            # Need at least 2 points to define a direction.
            centerline = np.vstack([centerline, centerline[0] + [1.0, 0.0]])
        width_left = np.resize(np.asarray(width_left, dtype=float), centerline.shape[0])
        width_right = np.resize(np.asarray(width_right, dtype=float), centerline.shape[0])
        diffs = np.gradient(centerline, axis=0)
        tangent = np.arctan2(diffs[:, 1], diffs[:, 0])
        return cls(centerline, width_left, width_right, tangent)

    @classmethod
    def fallback_straight(cls, width: float = 1.0) -> "LaneContext":
        """Straight-line lane used before the map is loaded."""
        xs = np.linspace(-20.0, 20.0, 200)
        centerline = np.column_stack([xs, np.zeros_like(xs)])
        hw = np.full(xs.shape, width / 2.0)
        return cls.from_centerline(centerline, hw, hw)

    def query(self, px: float, py: float) -> LaneSample:
        """Project (px, py) onto the nearest centerline segment."""
        p = np.array([px, py], dtype=float)
        center, tangent_val, idx = self._project(p)
        tv = np.array([np.cos(tangent_val), np.sin(tangent_val)])
        left_normal = np.array([-tv[1], tv[0]])           # 90° CCW = left
        signed_lateral_error = float(np.dot(p - center, left_normal))
        return LaneSample(
            point=center,
            tangent=tangent_val,
            signed_lateral_error=signed_lateral_error,
            width_left=float(self.width_left[idx]),
            width_right=float(self.width_right[idx]),
        )

    def _project(self, p: np.ndarray) -> Tuple[np.ndarray, float, int]:
        segs = self.centerline[1:] - self.centerline[:-1]
        seg_len_sq = np.maximum(np.sum(segs * segs, axis=1), 1e-12)
        t = np.clip(np.sum((p - self.centerline[:-1]) * segs, axis=1) / seg_len_sq, 0.0, 1.0)
        projs = self.centerline[:-1] + t[:, None] * segs
        dists = np.linalg.norm(projs - p, axis=1)
        idx = int(np.argmin(dists))
        tangent_val = float(np.arctan2(segs[idx, 1], segs[idx, 0]))
        width_idx = idx if t[idx] < 0.5 else min(idx + 1, len(self.centerline) - 1)
        return projs[idx], tangent_val, width_idx


def heading_error_to_lane(x: np.ndarray, lane: LaneContext) -> Tuple[float, LaneSample]:
    """Return (heading_error_rad, LaneSample) for state x."""
    sample = lane.query(float(x[0]), float(x[1]))
    return wrap_angle(sample.tangent - float(x[3])), sample


class LaneletContextBuilder:
    """Wraps the racecar_routing LaneletWrapper into a LaneContext."""

    def __init__(self, map_file: str, node: Optional[object] = None, lane_change_cost: float = 1.0):
        self.map_file = map_file
        self.node = node
        self.lane_change_cost = lane_change_cost
        self._wrapper = None

    def _load(self):
        if self._wrapper is not None:
            return self._wrapper
        from routing.routing.lanelet_wrapper import LaneletWrapper
        if self.node is not None:
            self.node.lane_change_cost = self.lane_change_cost
        self._wrapper = LaneletWrapper(self.map_file, self.node)
        return self._wrapper

    def build_near(self, state: np.ndarray, distance_m: float = 8.0) -> LaneContext:
        wrapper = self._load()
        pose = state[:3].tolist() if len(state) >= 3 else [state[0], state[1], 0.0]
        lanelet, arc = wrapper.get_closest_lanelet(pose, check_psi=True)
        L = max(wrapper.get_lanelet_length(lanelet), 1e-6)
        start_s = getattr(arc, "length", 0.0) / L
        routes = wrapper.get_reachable_path(lanelet, start_s, distance_m, allow_lane_change=True)

        centerline = self._pick_route(routes, state)
        wl, wr = [], []
        for x, y in centerline:
            nl, _ = wrapper.get_closest_lanelet([x, y], check_psi=False)
            pt = type("P", (), {"x": float(x), "y": float(y)})()
            left, right = wrapper.get_lane_width(pt, nl, allow_lane_change=True)
            wl.append(left)
            wr.append(right)

        return LaneContext.from_centerline(centerline, np.array(wl), np.array(wr))

    @staticmethod
    def _pick_route(routes, state: np.ndarray) -> np.ndarray:
        if not routes:
            return np.array([[state[0], state[1]], [state[0] + 1.0, state[1]]], dtype=float)
        heading = float(state[3]) if len(state) >= 4 else 0.0

        def score(r):
            if len(r) < 2:
                return float("inf")
            seg = np.asarray(r[1]) - np.asarray(r[0])
            return abs(wrap_angle(float(np.arctan2(seg[1], seg[0])) - heading))

        return np.asarray(min(routes, key=score), dtype=float)
