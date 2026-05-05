from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class Obstacle:
    tag_id: int
    position: np.ndarray
    radius: float


class ObstacleMemory:
    def __init__(self, ttl_s: float, growth_mps: float, default_radius: float):
        self.ttl_s = ttl_s
        self.growth_mps = growth_mps
        self.default_radius = default_radius
        self._last_seen: Dict[int, Tuple[float, np.ndarray, float]] = {}

    def update(self, t_now: float, detections: Iterable[Obstacle]) -> None:
        for obs in detections:
            self._last_seen[int(obs.tag_id)] = (
                float(t_now),
                np.asarray(obs.position, dtype=float),
                float(obs.radius),
            )
        self._evict(t_now)

    def get(self, t_now: float) -> List[Obstacle]:
        self._evict(t_now)
        obstacles = []
        for tag_id, (t_seen, position, radius) in self._last_seen.items():
            age = max(0.0, t_now - t_seen)
            obstacles.append(
                Obstacle(
                    tag_id=tag_id,
                    position=position.copy(),
                    radius=radius + age * self.growth_mps,
                )
            )
        return obstacles

    def _evict(self, t_now: float) -> None:
        self._last_seen = {
            tag_id: value
            for tag_id, value in self._last_seen.items()
            if t_now - value[0] <= self.ttl_s
        }
