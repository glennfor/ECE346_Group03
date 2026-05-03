"""
ObstacleMemory — remembers the last-seen position of each AprilTag obstacle.

AprilTag detection drops frames. Without memory, a single missed frame
would make the filter think "no obstacles" and potentially pass a dangerous
human command through. This class keeps each obstacle alive for ttl_s seconds
after its last detection, and inflates its radius as it ages (to account for
uncertainty in where the truck might be if we lose sight of it).
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class Obstacle:
    tag_id: int
    position: np.ndarray     # (2,) xy in world frame
    radius: float


class ObstacleMemory:
    def __init__(self, ttl_s: float, growth_mps: float, base_radius: float):
        self.ttl_s = ttl_s           # seconds before a stale obstacle is dropped
        self.growth_mps = growth_mps # radius grows this many m per second of age
        self.base_radius = base_radius

        # tag_id → (time_seen, position, base_radius_at_detection)
        self._store: Dict[int, Tuple[float, np.ndarray, float]] = {}

    def update(self, t_now: float, detections: Iterable[Obstacle]) -> None:
        """Record fresh detections and evict expired obstacles."""
        for obs in detections:
            self._store[obs.tag_id] = (t_now, obs.position.copy(), obs.radius)
        # Drop anything older than ttl_s.
        self._store = {
            k: v for k, v in self._store.items()
            if t_now - v[0] < self.ttl_s
        }

    def get(self, t_now: float) -> List[Obstacle]:
        """Return all remembered obstacles with age-inflated radii."""
        result = []
        for tag_id, (t_seen, pos, r_base) in self._store.items():
            age = t_now - t_seen
            inflated_r = r_base + age * self.growth_mps
            result.append(Obstacle(tag_id=tag_id, position=pos, radius=inflated_r))
        return result
