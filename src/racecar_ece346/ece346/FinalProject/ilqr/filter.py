"""ILQR monitor plus QP-style command arbiter."""

from dataclasses import dataclass
from typing import Optional, Sequence

from .config import IlqrQpConfig
from .geometry import (
    Margins,
    Obstacle,
    PathPoint,
    VehicleState,
    clamp,
    rollout_constant_command,
    trajectory_margins,
)
from .solver import IlqrLocalPlanner


@dataclass(frozen=True)
class FilterCommand:
    speed: float
    steering_angle: float
    is_override: bool
    reason: str
    lane_margin: Optional[float]
    obstacle_margin: Optional[float]
    cost: float


class IlqrQpFilter:
    def __init__(self, config: IlqrQpConfig):
        self.config = config
        self.planner = IlqrLocalPlanner(config)

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

        human_margins = self._monitor_human_command(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
        )
        if human_margins.is_safe(self.config.soft_margin):
            return FilterCommand(
                speed=human_speed,
                steering_angle=human_steering,
                is_override=False,
                reason="pass",
                lane_margin=human_margins.lane,
                obstacle_margin=human_margins.obstacle,
                cost=0.0,
            )

        plan = self.planner.plan(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
        )
        speed, steering = plan.first_command
        if not plan.is_feasible and not human_margins.is_safe(self.config.hard_margin):
            speed = self._clip_speed(self.config.no_solution_speed)
            if not self.config.no_solution_keep_steering:
                steering = 0.0

        return FilterCommand(
            speed=self._clip_speed(speed),
            steering_angle=self._clip_steering(steering),
            is_override=True,
            reason=plan.reason if plan.is_feasible else "stop_no_solution",
            lane_margin=plan.margins.lane,
            obstacle_margin=plan.margins.obstacle,
            cost=plan.cost,
        )

    def _monitor_human_command(
        self,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> Margins:
        cfg = self.config
        projected_speed = max(
            human_speed,
            cfg.min_projection_speed if human_speed > 0.0 else 0.0,
        )
        states = rollout_constant_command(
            state=state,
            speed_cmd=projected_speed,
            steering_cmd=human_steering,
            wheelbase=cfg.wheelbase,
            dt=cfg.dt,
            horizon_sec=cfg.horizon_sec,
            response_delay_sec=cfg.response_delay_sec,
            min_projection_speed=cfg.min_projection_speed,
        )
        stopping = human_speed * human_speed / (2.0 * max(cfg.max_decel, 1e-3))
        lane_buffer = cfg.vehicle_radius + cfg.lane_margin + cfg.localization_buffer
        obstacle_buffer = (
            cfg.vehicle_radius
            + cfg.localization_buffer
            + cfg.stopping_buffer
            + 0.25 * stopping
        )
        return trajectory_margins(
            states=states,
            path=path,
            obstacles=obstacles,
            lane_buffer=lane_buffer,
            obstacle_buffer=obstacle_buffer,
            require_path=cfg.require_path_for_lane_filter,
        )

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return clamp(float(speed), lower, self.config.max_speed)

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return clamp(float(steering), -limit, limit)
