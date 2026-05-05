"""Lightweight CBF-QP projection for Ackermann commands."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import CbfQpConfig
from .geometry import (
    Obstacle,
    PathPoint,
    VehicleState,
    clamp,
    min_lane_margin,
    min_obstacle_margin,
    rollout,
)


@dataclass(frozen=True)
class FilterCommand:
    speed: float
    steering_angle: float
    is_override: bool
    reason: str
    lane_margin: Optional[float]
    obstacle_margin: Optional[float]
    cost: float


class CbfQpFilter:
    def __init__(self, config: CbfQpConfig):
        self.config = config

    def filter_command(
        self,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> FilterCommand:
        human_speed = self._clip_speed(human_speed)
        human_steering = self._clip_steering(human_steering)

        candidates = self._candidate_commands(human_speed, human_steering)
        evaluated = [
            self._evaluate_candidate(
                speed=speed,
                steering=steering,
                human_speed=human_speed,
                human_steering=human_steering,
                state=state,
                path=path,
                obstacles=obstacles,
            )
            for speed, steering in candidates
        ]

        feasible = [
            command for command in evaluated
            if (command.lane_margin is None or command.lane_margin >= 0.0)
            and (command.obstacle_margin is None or command.obstacle_margin >= 0.0)
        ]
        if feasible:
            best = min(feasible, key=lambda command: command.cost)
            if abs(best.speed - human_speed) < 1e-6 and abs(
                best.steering_angle - human_steering
            ) < 1e-6:
                return FilterCommand(
                    speed=best.speed,
                    steering_angle=best.steering_angle,
                    is_override=False,
                    reason="pass",
                    lane_margin=best.lane_margin,
                    obstacle_margin=best.obstacle_margin,
                    cost=best.cost,
                )
            return best

        if evaluated:
            least_bad = min(evaluated, key=lambda command: command.cost)
            steering = (
                least_bad.steering_angle
                if self.config.no_solution_keep_steering
                else 0.0
            )
        else:
            steering = human_steering

        return FilterCommand(
            speed=self._clip_speed(self.config.no_solution_speed),
            steering_angle=self._clip_steering(steering),
            is_override=True,
            reason="stop_no_solution",
            lane_margin=None,
            obstacle_margin=None,
            cost=float("inf"),
        )

    def _evaluate_candidate(
        self,
        speed: float,
        steering: float,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> FilterCommand:
        cfg = self.config
        projected_speed = max(speed, cfg.min_projection_speed if speed > 0.0 else 0.0)
        states = rollout(
            state=state,
            speed_cmd=projected_speed,
            steering_cmd=steering,
            wheelbase=cfg.wheelbase,
            dt=cfg.dt,
            horizon_sec=cfg.horizon_sec,
            response_delay_sec=cfg.response_delay_sec,
        )
        lane_buffer = cfg.vehicle_radius + cfg.lane_margin + cfg.localization_buffer
        obs_buffer = cfg.vehicle_radius + cfg.localization_buffer
        stopping = speed * speed / (2.0 * max(cfg.max_decel, 1e-3))
        obs_buffer += cfg.stopping_buffer + 0.25 * stopping

        lane = min_lane_margin(states, path, lane_buffer) if path else None
        obstacle = min_obstacle_margin(states, obstacles, obs_buffer)
        if cfg.require_path_for_lane_filter and lane is None:
            lane = -cfg.lane_margin

        lane_violation = max(0.0, -(lane or 0.0))
        obstacle_violation = max(0.0, -(obstacle or 0.0))
        speed_delta = speed - human_speed
        steering_delta = steering - human_steering
        brake_delta = max(0.0, human_speed - speed)

        cost = (
            cfg.speed_weight * speed_delta * speed_delta
            + cfg.steering_weight * steering_delta * steering_delta
            + cfg.brake_weight * brake_delta * brake_delta
            + cfg.lane_violation_weight * lane_violation * lane_violation
            + cfg.obstacle_violation_weight * obstacle_violation * obstacle_violation
            - cfg.progress_weight * max(0.0, speed)
        )

        reason = self._reason(lane, obstacle)
        return FilterCommand(
            speed=speed,
            steering_angle=steering,
            is_override=reason != "pass",
            reason=reason,
            lane_margin=lane,
            obstacle_margin=obstacle,
            cost=cost,
        )

    def _candidate_commands(
        self,
        human_speed: float,
        human_steering: float,
    ) -> List[tuple]:
        candidates = []
        max_abs_speed = max(abs(human_speed), self.config.max_speed)
        for scale in self.config.speed_samples:
            scaled_speed = human_speed * float(scale)
            if human_speed >= 0.0:
                speed = clamp(scaled_speed, self.config.min_speed, max_abs_speed)
            elif self.config.allow_reverse:
                speed = clamp(scaled_speed, -max_abs_speed, self.config.max_speed)
            else:
                speed = 0.0
            for offset in self.config.steering_offsets:
                steering = self._clip_steering(human_steering + float(offset))
                candidates.append((self._clip_speed(speed), steering))

        candidates.append((0.0, self._clip_steering(human_steering)))
        candidates.append((0.0, 0.0))
        return list(dict.fromkeys(candidates))

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return clamp(float(speed), lower, self.config.max_speed)

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return clamp(float(steering), -limit, limit)

    @staticmethod
    def _reason(
        lane_margin_value: Optional[float],
        obstacle_margin_value: Optional[float],
    ) -> str:
        lane_bad = lane_margin_value is not None and lane_margin_value < 0.0
        obstacle_bad = (
            obstacle_margin_value is not None and obstacle_margin_value < 0.0
        )
        if lane_bad and obstacle_bad:
            return "lane_obstacle"
        if lane_bad:
            return "lane"
        if obstacle_bad:
            return "obstacle"
        return "pass"

