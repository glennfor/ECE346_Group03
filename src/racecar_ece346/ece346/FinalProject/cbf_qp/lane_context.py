from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .dynamics import wrap_angle


def _route_fingerprint(route: Any) -> Optional[Tuple]:
    route_array = np.asarray(route, dtype=float)
    if route_array.shape[0] < 2:
        return None
    return (
        int(route_array.shape[0]),
        float(route_array[0, 0]),
        float(route_array[0, 1]),
        float(route_array[1, 0]),
        float(route_array[1, 1]),
    )


@dataclass
class LaneSample:
    point: np.ndarray
    tangent: float
    signed_lateral_error: float
    width_left: float
    width_right: float


@dataclass
class LaneContext:
    centerline: np.ndarray
    width_left: np.ndarray
    width_right: np.ndarray
    tangent: np.ndarray
    is_fallback: bool = False

    @classmethod
    def from_centerline(
        cls,
        centerline: np.ndarray,
        width_left: np.ndarray,
        width_right: np.ndarray,
        is_fallback: bool = False,
    ) -> "LaneContext":
        centerline = np.asarray(centerline, dtype=float)
        if centerline.shape[0] < 2:
            centerline = np.vstack([centerline, centerline[0] + np.array([1.0, 0.0])])
        width_left = np.asarray(width_left, dtype=float)
        width_right = np.asarray(width_right, dtype=float)
        if width_left.shape[0] < centerline.shape[0]:
            width_left = np.resize(width_left, centerline.shape[0])
        if width_right.shape[0] < centerline.shape[0]:
            width_right = np.resize(width_right, centerline.shape[0])
        diffs = np.gradient(centerline, axis=0)
        tangent = np.arctan2(diffs[:, 1], diffs[:, 0])
        return cls(centerline, width_left, width_right, tangent, is_fallback)

    @classmethod
    def fallback_straight(cls, width: float = 100.0) -> "LaneContext":
        xs = np.linspace(-20.0, 20.0, 200)
        centerline = np.column_stack([xs, np.zeros_like(xs)])
        half_width = np.full(xs.shape, width / 2.0)
        return cls.from_centerline(centerline, half_width, half_width, is_fallback=True)

    @classmethod
    def from_path_msg(cls, msg) -> "LaneContext":
        centerline = []
        width_left = []
        width_right = []
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            centerline.append([pose.position.x, pose.position.y])
            width_left.append(max(float(pose.orientation.x), 0.0))
            width_right.append(max(float(pose.orientation.y), 0.0))

        if len(centerline) < 2:
            return cls.fallback_straight()
        return cls.from_centerline(
            np.asarray(centerline, dtype=float),
            np.asarray(width_left, dtype=float),
            np.asarray(width_right, dtype=float),
        )

    def query(self, px: float, py: float) -> LaneSample:
        p = np.array([px, py], dtype=float)
        center, tangent, idx = self._project_to_centerline(p)
        tangent_vec = np.array([np.cos(tangent), np.sin(tangent)])
        left_normal = np.array([-tangent_vec[1], tangent_vec[0]])
        signed_lateral_error = float(np.dot(p - center, left_normal))

        return LaneSample(
            point=center,
            tangent=tangent,
            signed_lateral_error=signed_lateral_error,
            width_left=float(self.width_left[idx]),
            width_right=float(self.width_right[idx]),
        )

    def _project_to_centerline(self, p: np.ndarray) -> Tuple[np.ndarray, float, int]:
        segments = self.centerline[1:] - self.centerline[:-1]
        seg_len_sq = np.sum(segments * segments, axis=1)
        seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)

        rel = p - self.centerline[:-1]
        t = np.sum(rel * segments, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        projections = self.centerline[:-1] + t[:, None] * segments
        distances = np.linalg.norm(projections - p, axis=1)
        seg_idx = int(np.argmin(distances))

        projected = projections[seg_idx]
        tangent = float(np.arctan2(segments[seg_idx, 1], segments[seg_idx, 0]))
        width_idx = seg_idx if t[seg_idx] < 0.5 else min(seg_idx + 1, len(self.centerline) - 1)
        return projected, tangent, width_idx


class LaneletContextBuilder:
    def __init__(
        self,
        map_file: str,
        node: Optional[object] = None,
        lane_change_cost: float = 1.0,
        allow_lane_change: bool = True,
        route_hysteresis_rad: float = 0.25,
    ):
        self.map_file = map_file
        self.node = node
        self.lane_change_cost = lane_change_cost
        self.allow_lane_change = allow_lane_change
        self.route_hysteresis_rad = route_hysteresis_rad
        self._wrapper = None
        self._last_route_fingerprint: Optional[Tuple] = None

    def _load_wrapper(self):
        if self._wrapper is not None:
            return self._wrapper

        from routing.routing.lanelet_wrapper import LaneletWrapper

        if self.node is not None:
            self.node.lane_change_cost = self.lane_change_cost
        self._wrapper = LaneletWrapper(self.map_file, self.node)
        return self._wrapper

    def build_near(self, state: np.ndarray, distance_m: float = 8.0) -> LaneContext:
        wrapper = self._load_wrapper()
        lane_pose = self._state_to_lane_pose(state)
        lanelet, arc = wrapper.get_closest_lanelet(lane_pose, check_psi=True)
        start_s = getattr(arc, "length", 0.0) / max(wrapper.get_lanelet_length(lanelet), 1e-6)
        allow_lane_change = self.allow_lane_change
        routes = wrapper.get_reachable_path(
            lanelet,
            start_s,
            distance_m,
            allow_lane_change=allow_lane_change,
        )
        width_allow_lane_change = allow_lane_change
        if not routes:
            routes = wrapper.get_reachable_path(
                lanelet,
                start_s,
                distance_m,
                allow_lane_change=True,
            )
            width_allow_lane_change = True
        if not routes:
            raise RuntimeError("get_reachable_path returned no routes")

        centerline = self._select_route(routes, lane_pose)
        width_left = []
        width_right = []
        for x, y in centerline:
            nearest_lanelet, _ = wrapper.get_closest_lanelet([x, y], check_psi=False)
            point = type("Point", (), {"x": float(x), "y": float(y)})()
            left, right = wrapper.get_lane_width(
                point,
                nearest_lanelet,
                allow_lane_change=width_allow_lane_change,
            )
            width_left.append(left)
            width_right.append(right)

        return LaneContext.from_centerline(centerline, np.array(width_left), np.array(width_right))

    @staticmethod
    def _state_to_lane_pose(state: np.ndarray) -> np.ndarray:
        if len(state) >= 4:
            return np.array([state[0], state[1], state[3]], dtype=float)
        return np.asarray(state[:3], dtype=float)

    def _select_route(self, routes: List, lane_pose: np.ndarray) -> np.ndarray:
        if not routes:
            raise ValueError("empty routes in _select_route")

        heading = float(lane_pose[2]) if len(lane_pose) >= 3 else 0.0

        def route_score(route):
            if len(route) < 2:
                return float("inf")
            segment = route[1] - route[0]
            tangent = np.arctan2(segment[1], segment[0])
            heading_error = abs(wrap_angle(tangent - heading))
            start_distance = np.linalg.norm(route[0] - lane_pose[:2])
            return heading_error + 0.1 * start_distance

        scored = [(route_score(route), route) for route in routes if len(route) >= 2]
        if not scored:
            raise ValueError("no route with >= 2 centerline points")

        best_score, best_route = min(scored, key=lambda item: item[0])
        chosen_route = best_route
        if self._last_route_fingerprint is not None and self.route_hysteresis_rad > 0.0:
            for score, route in scored:
                if score > best_score + self.route_hysteresis_rad:
                    continue
                if _route_fingerprint(route) == self._last_route_fingerprint:
                    chosen_route = route
                    break

        route_array = np.asarray(chosen_route, dtype=float)
        self._last_route_fingerprint = _route_fingerprint(route_array)
        return route_array


def heading_error_to_lane(x: np.ndarray, lane: LaneContext) -> Tuple[float, LaneSample]:
    sample = lane.query(float(x[0]), float(x[1]))
    return wrap_angle(sample.tangent - float(x[3])), sample
