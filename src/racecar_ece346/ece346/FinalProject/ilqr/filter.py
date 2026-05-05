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
    fallback_path_ahead,
    forward_obstacle_margin,
    obstacle_time_to_collision,
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
        path = self._effective_path(state, human_speed, path)

        if self.config.require_path_for_lane_filter and len(path) < 2:
            return FilterCommand(
                speed=self._clip_speed(self.config.no_solution_speed),
                steering_angle=self._clip_steering(
                    human_steering if self.config.no_solution_keep_steering else 0.0
                ),
                is_override=True,
                reason="missing_path",
                lane_margin=None,
                obstacle_margin=None,
                cost=float("inf"),
            )

        human_margins = self._monitor_human_command(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
        )
        forward_margin = self._forward_margin(
            human_speed=human_speed,
            state=state,
            obstacles=obstacles,
        )
        ttc = self._time_to_collision(
            human_speed=human_speed,
            state=state,
            obstacles=obstacles,
        )
        hard_forward_stop = (
            forward_margin is not None
            and forward_margin <= self.config.emergency_brake_margin
        ) or (ttc is not None and ttc <= self.config.ttc_hard_sec)
        soft_forward_stop = ttc is not None and ttc <= self.config.ttc_soft_sec

        if hard_forward_stop:
            return FilterCommand(
                speed=self._clip_speed(self.config.no_solution_speed),
                steering_angle=self._clip_steering(human_steering),
                is_override=True,
                reason="obstacle_brake",
                lane_margin=human_margins.lane,
                obstacle_margin=human_margins.obstacle,
                cost=float("inf"),
            )

        path_follow_command = self._path_follow_command(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
            human_margins=human_margins,
        )
        if path_follow_command is not None and not soft_forward_stop:
            return path_follow_command

        if human_margins.is_safe(self.config.soft_margin) and not soft_forward_stop:
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

    def _effective_path(
        self,
        state: VehicleState,
        human_speed: float,
        path: Sequence[PathPoint],
    ) -> Sequence[PathPoint]:
        if len(path) >= 2 or not self.config.fallback_path_enabled:
            return path
        return fallback_path_ahead(
            state=state,
            length=self.config.fallback_path_length,
            width=self.config.fallback_path_width,
            speed_limit=max(human_speed, self.config.min_projection_speed),
            samples=self.config.fallback_path_samples,
        )

    def _path_follow_command(
        self,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
        human_margins: Margins,
    ) -> Optional[FilterCommand]:
        if len(path) < 2 or human_speed <= 1e-3:
            return None

        horizon = self.planner._horizon_steps()
        sequence = self.planner._path_follow_sequence(
            state=state,
            path=path,
            speed=human_speed,
            horizon=horizon,
        )
        if not sequence:
            return None

        plan = self.planner._evaluate(
            commands=sequence,
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
        )
        speed, steering = plan.first_command
        steering_delta = abs(steering - human_steering)
        lane_needs_help = (
            human_margins.lane is not None
            and human_margins.lane < self.config.soft_margin
        )
        if not plan.is_feasible or (not lane_needs_help and steering_delta < 0.08):
            return None

        reason = "lane" if lane_needs_help else "corner"
        return FilterCommand(
            speed=self._clip_speed(speed),
            steering_angle=self._clip_steering(steering),
            is_override=True,
            reason=reason,
            lane_margin=plan.margins.lane,
            obstacle_margin=plan.margins.obstacle,
            cost=plan.cost,
        )

    def _forward_margin(
        self,
        human_speed: float,
        state: VehicleState,
        obstacles: Sequence[Obstacle],
    ) -> Optional[float]:
        cfg = self.config
        stopping = human_speed * human_speed / (2.0 * max(cfg.max_decel, 1e-3))
        obstacle_buffer = (
            cfg.vehicle_radius
            + cfg.localization_buffer
            + cfg.stopping_buffer
            + 0.5 * stopping
        )
        projected_state = VehicleState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            speed=max(state.speed, human_speed),
            steering_angle=state.steering_angle,
        )
        return forward_obstacle_margin(
            state=projected_state,
            obstacles=obstacles,
            safety_buffer=obstacle_buffer,
            forward_width=cfg.forward_obstacle_width,
            max_distance=cfg.forward_obstacle_distance,
        )

    def _time_to_collision(
        self,
        human_speed: float,
        state: VehicleState,
        obstacles: Sequence[Obstacle],
    ) -> Optional[float]:
        cfg = self.config
        projected_state = VehicleState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            speed=max(state.speed, human_speed),
            steering_angle=state.steering_angle,
        )
        return obstacle_time_to_collision(
            state=projected_state,
            obstacles=obstacles,
            safety_buffer=cfg.vehicle_radius + cfg.localization_buffer,
            forward_width=cfg.forward_obstacle_width,
            max_distance=cfg.forward_obstacle_distance,
        )

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return clamp(float(speed), lower, self.config.max_speed)

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return clamp(float(steering), -limit, limit)
