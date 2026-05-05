"""Local ILQR-style trajectory search used by the ILQR-QP filter."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import IlqrQpConfig
from .geometry import (
    Margins,
    Obstacle,
    PathPoint,
    VehicleState,
    clamp,
    heading_error,
    rollout_command_sequence,
    trajectory_margins,
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
        seeds = self._seed_sequences(human_speed, human_steering)
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

    def _seed_sequences(self, human_speed: float, human_steering: float) -> List[Tuple[Command, ...]]:
        horizon = self._horizon_steps()
        seeds: List[Tuple[Command, ...]] = []

        for speed_scale in self.config.speed_scales:
            speed = self._clip_speed(human_speed * float(speed_scale))
            for steering_offset in self.config.steering_offsets:
                steering = self._clip_steering(human_steering + float(steering_offset))
                seeds.append(tuple((speed, steering) for _ in range(horizon)))

        for steering in self.config.recovery_steering_offsets:
            recovery = self._clip_steering(float(steering))
            mid = max(1, int(0.7 * horizon))
            speed = self._clip_speed(0.55 * max(human_speed, self.config.min_projection_speed))
            sequence = [
                (speed, recovery if idx < mid else self._clip_steering(human_steering))
                for idx in range(horizon)
            ]
            seeds.append(tuple(sequence))

        if self._previous_commands:
            shifted = list(self._previous_commands[1:])
            shifted.append((self._clip_speed(human_speed), self._clip_steering(human_steering)))
            if len(shifted) == horizon:
                seeds.append(tuple(shifted))

        return list(dict.fromkeys(seeds))

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
        lane_violation = max(0.0, cfg.soft_margin - (margins.lane or cfg.soft_margin))
        obstacle_violation = max(0.0, cfg.soft_margin - (margins.obstacle or cfg.soft_margin))
        speed_delta = first_speed - human_speed
        steering_delta = first_steering - human_steering
        brake_delta = max(0.0, human_speed - first_speed)
        steering_rate = abs(first_steering - state.steering_angle) / max(cfg.dt, 1e-3)
        end_heading = abs(heading_error(states[-1], path)) if path else 0.0
        progress = sum(max(0.0, command[0]) * cfg.dt for command in commands)

        cost = (
            cfg.speed_weight * speed_delta * speed_delta
            + cfg.steering_weight * steering_delta * steering_delta
            + cfg.brake_weight * brake_delta * brake_delta
            + cfg.steering_rate_weight * steering_rate * steering_rate
            + cfg.lane_violation_weight * lane_violation * lane_violation
            + cfg.obstacle_violation_weight * obstacle_violation * obstacle_violation
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
