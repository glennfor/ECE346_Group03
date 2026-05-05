"""Local ILQR-style trajectory search used by the ILQR-QP filter."""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

from .config import IlqrQpConfig
from .geometry import (
    Margins,
    Obstacle,
    PathPoint,
    VehicleState,
    clamp,
    closest_path_projection_full,
    forward_obstacle_margin,
    heading_error,
    rollout_command_sequence,
    step_state,
    trajectory_path_tracking_error,
    trajectory_margins,
    wrap_angle,
)


Command = Tuple[float, float]


@dataclass(frozen=True)
class CandidatePlan:
    commands: Tuple[Command, ...]
    states: Tuple[VehicleState, ...]
    margins: Margins
    cost: float
    reason: str

    @property
    def first_command(self) -> Command:
        return self.commands[0] if self.commands else (0.0, 0.0)

    @property
    def is_feasible(self) -> bool:
        return self.margins.is_safe(0.0)


class IlqrLocalPlanner:
    """Finite-horizon planner with ILQR-style warm starts and local refinement."""

    def __init__(self, config: IlqrQpConfig):
        self.config = config
        self._previous_commands: Optional[Tuple[Command, ...]] = None

    def plan(
        self,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> CandidatePlan:
        seeds = self._seed_sequences(human_speed, human_steering, state, path, obstacles)
        seed_plans = [
            self._evaluate(seed, human_speed, human_steering, state, path, obstacles)
            for seed in seeds
        ]
        seed_plans.sort(key=lambda candidate: candidate.cost)

        best: Optional[CandidatePlan] = None
        best_feasible: Optional[CandidatePlan] = None

        for seed_plan in seed_plans[:8]:
            seed = seed_plan.commands
            refined = self._refine(seed, human_speed, human_steering, state, path, obstacles)
            candidate = self._evaluate(refined, human_speed, human_steering, state, path, obstacles)
            if best is None or candidate.cost < best.cost:
                best = candidate
            if candidate.is_feasible and (
                best_feasible is None or candidate.cost < best_feasible.cost
            ):
                best_feasible = candidate

        if best is None and seed_plans:
            best = seed_plans[0]
        feasible_seeds = [candidate for candidate in seed_plans if candidate.is_feasible]
        if best_feasible is None and feasible_seeds:
            best_feasible = min(feasible_seeds, key=lambda candidate: candidate.cost)

        selected = best_feasible or best
        if selected is None:
            selected = self._failsafe_plan(human_steering, state, path, obstacles)
        self._previous_commands = selected.commands
        return selected

    def _seed_sequences(
        self,
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> List[Tuple[Command, ...]]:
        horizon = self._horizon_steps()
        seeds: List[Tuple[Command, ...]] = []

        for speed_scale in self.config.speed_scales:
            speed = self._clip_speed(human_speed * float(speed_scale))
            for steering_offset in self.config.steering_offsets:
                steering = self._clip_steering(human_steering + float(steering_offset))
                seeds.append(tuple((speed, steering) for _ in range(horizon)))

        for speed_scale in self.config.path_follow_speed_scales:
            path_seed = self._path_follow_sequence(
                state=state,
                path=path,
                speed=self._clip_speed(human_speed * float(speed_scale)),
                horizon=horizon,
            )
            if path_seed:
                seeds.append(path_seed)

        for steering in self.config.recovery_steering_offsets:
            recovery = self._clip_steering(float(steering))
            mid = max(1, int(0.7 * horizon))
            speed = self._clip_speed(0.55 * max(human_speed, self.config.min_projection_speed))
            sequence = [
                (speed, recovery if idx < mid else self._clip_steering(human_steering))
                for idx in range(horizon)
            ]
            seeds.append(tuple(sequence))

        for dodge_steering in self.config.dodge_steering_offsets:
            dodge = self._dodge_sequence(
                human_speed=human_speed,
                human_steering=human_steering,
                dodge_steering=float(dodge_steering),
                horizon=horizon,
            )
            seeds.append(dodge)

        seeds.extend(self._brake_sequences(human_speed, human_steering, horizon))

        if self._previous_commands:
            shifted = list(self._previous_commands[1:])
            shifted.append((self._clip_speed(human_speed), self._clip_steering(human_steering)))
            if len(shifted) == horizon:
                seeds.append(tuple(shifted))

        return list(dict.fromkeys(seeds))

    def _path_follow_sequence(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
        speed: float,
        horizon: int,
    ) -> Tuple[Command, ...]:
        if len(path) < 2:
            return tuple()

        commands: List[Command] = []
        current = state
        for _ in range(horizon):
            steering = self._path_follow_steering(current, path, max(speed, current.speed))
            command_speed = speed
            if abs(steering) > 0.22:
                command_speed = min(speed, self.config.path_corner_speed_scale * max(speed, 0.1))
            command = (self._clip_speed(command_speed), self._clip_steering(steering))
            commands.append(command)
            current = step_state(current, command[0], command[1], self.config.wheelbase, self.config.dt)
        return tuple(commands)

    def _path_follow_steering(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
        speed: float,
    ) -> float:
        projection = closest_path_projection_full(state.x, state.y, path)
        if projection is None:
            return state.steering_angle

        lookahead = self.config.path_lookahead_base + self.config.path_lookahead_time * max(speed, 0.0)
        target = self._lookahead_point(path, projection, lookahead)
        target_yaw = math.atan2(target.y - state.y, target.x - state.x)
        alpha = wrap_angle(target_yaw - state.yaw)
        pure_pursuit = math.atan2(
            2.0 * self.config.wheelbase * math.sin(alpha),
            max(lookahead, 1e-3),
        )
        heading_correction = -self.config.path_heading_gain * wrap_angle(
            state.yaw - projection.tangent_yaw
        )
        lateral_correction = -self.config.path_lateral_gain * projection.lateral
        return self._clip_steering(pure_pursuit + heading_correction + lateral_correction)

    def _lookahead_point(
        self,
        path: Sequence[PathPoint],
        projection,
        lookahead: float,
    ) -> PathPoint:
        remaining = max(0.0, lookahead)
        idx = projection.segment_index
        t = projection.segment_t

        while idx < len(path) - 1:
            p0 = path[idx]
            p1 = path[idx + 1]
            seg_len = math.hypot(p1.x - p0.x, p1.y - p0.y)
            available = (1.0 - t) * seg_len
            if seg_len > 1e-9 and remaining <= available:
                ratio = t + remaining / seg_len
                return PathPoint(
                    x=p0.x + ratio * (p1.x - p0.x),
                    y=p0.y + ratio * (p1.y - p0.y),
                    left_width=(1.0 - ratio) * p0.left_width + ratio * p1.left_width,
                    right_width=(1.0 - ratio) * p0.right_width + ratio * p1.right_width,
                    speed_limit=(1.0 - ratio) * p0.speed_limit + ratio * p1.speed_limit,
                )
            remaining -= max(0.0, available)
            idx += 1
            t = 0.0

        return path[-1]

    def _dodge_sequence(
        self,
        human_speed: float,
        human_steering: float,
        dodge_steering: float,
        horizon: int,
    ) -> Tuple[Command, ...]:
        dodge_steps = max(1, int(0.35 * horizon))
        recover_steps = max(1, int(0.35 * horizon))
        speed = self._clip_speed(0.55 * max(human_speed, self.config.min_projection_speed))
        commands: List[Command] = []
        for idx in range(horizon):
            if idx < dodge_steps:
                steering = human_steering + dodge_steering
            elif idx < dodge_steps + recover_steps:
                steering = human_steering - 0.55 * dodge_steering
            else:
                steering = human_steering
            commands.append((speed, self._clip_steering(steering)))
        return tuple(commands)

    def _brake_sequences(
        self,
        human_speed: float,
        human_steering: float,
        horizon: int,
    ) -> List[Tuple[Command, ...]]:
        sequences: List[Tuple[Command, ...]] = []
        for keep_steering in (True, False):
            commands: List[Command] = []
            for idx in range(horizon):
                ratio = max(0.0, 1.0 - (idx + 1) / max(1, int(0.45 * horizon)))
                speed = self._clip_speed(human_speed * ratio)
                steering = human_steering if keep_steering else 0.0
                commands.append((speed, self._clip_steering(steering)))
            sequences.append(tuple(commands))
        return sequences

    def _refine(
        self,
        seed: Tuple[Command, ...],
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> Tuple[Command, ...]:
        current = seed
        current_plan = self._evaluate(current, human_speed, human_steering, state, path, obstacles)
        for iteration in range(max(0, int(self.config.ilqr_iterations))):
            scale = 0.5 ** iteration
            speed_step = 0.18 * scale
            steer_step = 0.08 * scale
            alternatives = []
            for speed_delta, steer_delta in (
                (-speed_step, 0.0),
                (speed_step, 0.0),
                (0.0, -steer_step),
                (0.0, steer_step),
                (-speed_step, -steer_step),
                (-speed_step, steer_step),
            ):
                alternatives.append(
                    tuple(
                        (
                            self._clip_speed(speed + speed_delta),
                            self._clip_steering(steering + steer_delta),
                        )
                        for speed, steering in current
                    )
                )
            for alternative in alternatives:
                plan = self._evaluate(alternative, human_speed, human_steering, state, path, obstacles)
                if plan.cost < current_plan.cost:
                    current = alternative
                    current_plan = plan
        return current

    def _evaluate(
        self,
        commands: Tuple[Command, ...],
        human_speed: float,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> CandidatePlan:
        cfg = self.config
        states = tuple(
            rollout_command_sequence(
                state=state,
                commands=commands,
                wheelbase=cfg.wheelbase,
                dt=cfg.dt,
            )
        )
        first_speed, first_steering = commands[0] if commands else (0.0, 0.0)
        stopping = first_speed * first_speed / (2.0 * max(cfg.max_decel, 1e-3))
        lane_buffer = cfg.vehicle_radius + cfg.lane_margin + cfg.localization_buffer
        obstacle_buffer = (
            cfg.vehicle_radius
            + cfg.localization_buffer
            + cfg.stopping_buffer
            + 0.25 * stopping
        )
        margins = trajectory_margins(
            states=states,
            path=path,
            obstacles=obstacles,
            lane_buffer=lane_buffer,
            obstacle_buffer=obstacle_buffer,
            require_path=cfg.require_path_for_lane_filter,
        )
        lane_margin_value = cfg.soft_margin if margins.lane is None else margins.lane
        obstacle_margin_value = (
            cfg.soft_margin if margins.obstacle is None else margins.obstacle
        )
        lane_violation = max(0.0, cfg.soft_margin - lane_margin_value)
        obstacle_violation = max(0.0, cfg.soft_margin - obstacle_margin_value)
        forward_margin = forward_obstacle_margin(
            state=states[0],
            obstacles=obstacles,
            safety_buffer=obstacle_buffer,
            forward_width=cfg.forward_obstacle_width,
            max_distance=cfg.forward_obstacle_distance,
        )
        forward_violation = max(
            0.0,
            cfg.emergency_brake_margin
            - (
                cfg.emergency_brake_margin
                if forward_margin is None
                else forward_margin
            ),
        )
        if forward_margin is not None and first_speed > 1e-3:
            ttc = max(0.0, forward_margin) / first_speed
            forward_violation = max(
                forward_violation,
                max(0.0, cfg.ttc_soft_sec - ttc) / max(cfg.ttc_soft_sec, 1e-3),
            )
        tracking = trajectory_path_tracking_error(states, path)
        speed_delta = first_speed - human_speed
        steering_delta = first_steering - human_steering
        brake_delta = max(0.0, human_speed - first_speed)
        sequence_speed_error = (
            sum((command[0] - human_speed) ** 2 for command in commands)
            / max(1, len(commands))
        )
        sequence_brake_error = (
            sum(max(0.0, human_speed - command[0]) ** 2 for command in commands)
            / max(1, len(commands))
        )
        steering_rate = abs(first_steering - state.steering_angle) / max(cfg.dt, 1e-3)
        end_heading = abs(heading_error(states[-1], path)) if path else 0.0
        progress = sum(max(0.0, command[0]) * cfg.dt for command in commands)
        expected_path_progress = (
            0.6 * max(0.0, human_speed) * cfg.dt * max(1, len(commands))
        )
        path_progress_error = (
            max(0.0, expected_path_progress - progress)
            if path and margins.obstacle is None and forward_margin is None
            else 0.0
        )

        cost = (
            cfg.speed_weight * speed_delta * speed_delta
            + cfg.speed_weight * sequence_speed_error
            + cfg.steering_weight * steering_delta * steering_delta
            + cfg.brake_weight * brake_delta * brake_delta
            + cfg.brake_weight * sequence_brake_error
            + cfg.steering_rate_weight * steering_rate * steering_rate
            + cfg.lane_violation_weight * lane_violation * lane_violation
            + cfg.obstacle_violation_weight * obstacle_violation * obstacle_violation
            + cfg.forward_obstacle_weight * forward_violation * forward_violation
            + cfg.path_lateral_weight * tracking.mean_abs_lateral * tracking.mean_abs_lateral
            + cfg.path_heading_weight * tracking.mean_abs_heading * tracking.mean_abs_heading
            + cfg.path_progress_weight * path_progress_error * path_progress_error
            + cfg.heading_weight * end_heading * end_heading
            - cfg.progress_weight * progress
        )
        return CandidatePlan(
            commands=commands,
            states=states,
            margins=margins,
            cost=cost,
            reason=self._reason(margins),
        )

    def _failsafe_plan(
        self,
        human_steering: float,
        state: VehicleState,
        path: Sequence[PathPoint],
        obstacles: Sequence[Obstacle],
    ) -> CandidatePlan:
        steering = human_steering if self.config.no_solution_keep_steering else 0.0
        command = (self._clip_speed(self.config.no_solution_speed), self._clip_steering(steering))
        commands = tuple(command for _ in range(self._horizon_steps()))
        return self._evaluate(commands, 0.0, human_steering, state, path, obstacles)

    def _horizon_steps(self) -> int:
        return max(1, int(round(self.config.horizon_sec / max(self.config.dt, 1e-3))))

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return clamp(float(speed), lower, self.config.max_speed)

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return clamp(float(steering), -limit, limit)

    @staticmethod
    def _reason(margins: Margins) -> str:
        lane_bad = margins.lane is not None and margins.lane < 0.0
        obstacle_bad = margins.obstacle is not None and margins.obstacle < 0.0
        if lane_bad and obstacle_bad:
            return "lane_obstacle"
        if lane_bad:
            return "lane"
        if obstacle_bad:
            return "obstacle"
        return "soft_margin" if margins.minimum is not None else "pass"
