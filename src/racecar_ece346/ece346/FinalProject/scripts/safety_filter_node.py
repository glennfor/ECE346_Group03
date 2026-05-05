#!/usr/bin/env python3
"""
Final Project — Safety Filter (STUDENT SKELETON)

You will build a ROS2 node that sits between a human driver (PS4 joystick)
and the vehicle, intervening only when necessary to keep the car safe.

Pipeline:

    /joy --> joy_to_ackermann --> /teleop ┐
                                           │
                             /SLAM/Pose ───┤---> [safety_filter_node] ---> /drive
                                           │
                       /Obstacles/Static ──┘

Topics you will work with:

    /teleop            ackermann_msgs/AckermannDriveStamped  (human command in)
    <odom_topic>       nav_msgs/Odometry                     (vehicle state in)
    /Obstacles/Static  visualization_msgs/MarkerArray        (obstacles in)
    /drive             ackermann_msgs/AckermannDriveStamped  (safe command out)

------------------------------------------------------------------------------
Task 1: Build the node plumbing (subscribers, publisher, timer).
Task 2: Implement the safety filter logic.
------------------------------------------------------------------------------

You may:
  - Add imports, helper methods, and ROS parameters to THIS file.
  - Add your own files anywhere under FinalProject/ — e.g. an `ilqr/`
    subfolder with your ILQR solver, an `ilqr_params.yaml` with cost
    weights, utility modules, etc. Load yaml configs from within this
    node using `open()` + `yaml.safe_load()`, or declare a ROS param
    for the config path and set it in `final_project_*.yaml`.
  - Edit any yaml under `FinalProject/config/` — add parameters, tune
    thresholds, change topic names. All existing yaml values are
    documented and intended to be tunable.

You should NOT need to rewrite the launch files, the plumbing nodes
(joy_to_ackermann, drive_to_servo, etc.), or the CMakeLists top-level
install lists. Only touch those if you are adding a new standalone
executable — in which case ask first.
"""

import copy
import math
from dataclasses import fields, replace
from typing import Optional

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
# from ece346.FinalProject.cbf_heuristic.node import main as cbf_heuristic_main
# from ece346.FinalProject.cbf_qp.node import main as cbf_qp_main
# from ece346.FinalProject.ilqr.node import main as ilqr_main
from ece346.FinalProject.ilqr.config import DEFAULT_CONFIG_PATH, IlqrQpConfig, load_config
from ece346.FinalProject.ilqr.geometry import (
    fallback_path_ahead,
    forward_obstacle_margin,
    obstacles_from_msg,
    obstacle_time_to_collision,
    path_from_msg,
    state_from_odom,
)
from ece346.FinalProject.ilqr.solver import CandidatePlan, IlqrLocalPlanner
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray


def yaw_from_quat(qx, qy, qz, qw):
    """Extract yaw (heading, rad) from a quaternion. Useful for Task 2."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SafetyFilterNode(Node):

    # =========================================================================
    # TASK 1 — Node setup (subscribers, publisher, timer)
    # =========================================================================
    #
    # Fill in __init__ below so that the node:
    #   1. Declares ROS parameters for each topic name and for the publish rate.
    #      Hint: use self.declare_parameter('name', default_value). The yaml
    #      file (final_project_*.yaml) will override these at launch time.
    #      Required parameter names (match the yaml):
    #          teleop_topic, drive_topic, odom_topic, static_obs_topic,
    #          publish_rate
    #
    #   2. Creates three subscribers that cache the latest message each:
    #          /teleop            -> AckermannDriveStamped
    #          <odom_topic>       -> Odometry
    #          /Obstacles/Static  -> MarkerArray
    #      Hint: self.create_subscription(MsgType, topic, callback, queue_size)
    #      Each callback can be a one-liner that stores msg into an instance
    #      attribute (e.g. self._latest_teleop = msg).
    #
    #   3. Creates one publisher:
    #          /drive             -> AckermannDriveStamped
    #      Hint: self.create_publisher(MsgType, topic, queue_size)
    #
    #   4. Creates a timer at `publish_rate` Hz that calls a method which
    #      invokes self.safety_filter(...) and publishes the result.
    #      Hint: self.create_timer(period_sec, callback)
    #
    # For reference, open any other node in this repo (e.g.
    # FinalProject/scripts/joy_to_ackermann_node.py) to see the same pattern.
    # =========================================================================

    def __init__(self):
        super().__init__('safety_filter_node')

        # ---- TODO(Task 1.1): declare ROS parameters ----
        self.declare_parameter("ilqr_config", str(DEFAULT_CONFIG_PATH))
        config_path = self.get_parameter("ilqr_config").value

        # ---- TODO(Task 1.2): read parameter values ----
        self.config = self._declare_and_read_config(load_config(config_path))

        # Use two independent ILQR planners: one scores the current human command,
        # and one searches for the best safe override when the human command is bad.
        self.safety_planner = IlqrLocalPlanner(self.config)
        self.optimal_planner = IlqrLocalPlanner(self.config)

        self._latest_teleop: Optional[AckermannDriveStamped] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_obs: Optional[MarkerArray] = None
        self._latest_path: Optional[Path] = None
        self._teleop_time: Optional[float] = None
        self._odom_time: Optional[float] = None
        self._obstacle_time: Optional[float] = None
        self._path_time: Optional[float] = None
        self._last_log_time = 0.0
        self._last_steering = 0.0

        # ---- TODO(Task 1.3): create subscribers ----
        self.create_subscription(
            AckermannDriveStamped,
            self.config.teleop_topic,
            self._teleop_cb,
            1,
        )
        self.create_subscription(Odometry, self.config.odom_topic, self._odom_cb, 1)
        self.create_subscription(
            MarkerArray,
            self.config.static_obs_topic,
            self._obstacles_cb,
            1,
        )
        self.create_subscription(Path, self.config.routing_path_topic, self._path_cb, 1)

        # ---- TODO(Task 1.4): create the publisher ----
        self.pub = self.create_publisher(AckermannDriveStamped, self.config.drive_topic, 1)

        # ---- TODO(Task 1.5): create a timer at publish_rate Hz ----
        self.create_timer(1.0 / max(self.config.publish_rate, 1e-3), self._publish_filtered)

        # self.get_logger().info(
        #     f"safety_filter_node ready: {teleop_topic} + {odom_topic} "
        #     f"+ {obs_topic} -> {drive_topic}"
        # )
        self.get_logger().info(
            "ilqr safety_filter_node ready: "
            f"{self.config.teleop_topic} + {self.config.odom_topic} + "
            f"{self.config.static_obs_topic} + {self.config.routing_path_topic} "
            f"-> {self.config.drive_topic}"
        )

    def _declare_and_read_config(self, config: IlqrQpConfig) -> IlqrQpConfig:
        values = {}
        for field in fields(IlqrQpConfig):
            default = getattr(config, field.name)
            self.declare_parameter(
                field.name,
                list(default) if isinstance(default, tuple) else default,
            )
            value = self.get_parameter(field.name).value
            if isinstance(default, tuple) and isinstance(value, list):
                value = tuple(value)
            values[field.name] = value
        return replace(config, **values)

    # ---- TODO(Task 1.6): implement callbacks ----

    def _teleop_cb(self, msg: AckermannDriveStamped):
        self._latest_teleop = msg
        self._teleop_time = self._now_sec()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_time = self._now_sec()

    def _obstacles_cb(self, msg: MarkerArray):
        self._latest_obs = msg
        self._obstacle_time = self._now_sec()

    def _path_cb(self, msg: Path):
        self._latest_path = msg
        self._path_time = self._now_sec()

    # ---- TODO(Task 1.7): the timer callback ----
    # Should:
    #   - return early if no teleop has arrived yet (self._latest_teleop is None)
    #   - call self.safety_filter(teleop=..., odom=..., obstacles=...)
    #   - if the returned command is not None, update its header.stamp to now
    #     and publish it on /drive
    #
    # def _publish_filtered(self):
    #
    def _publish_filtered(self):
        if self._latest_teleop is None:
            return

        command = self.safety_filter(
            teleop=self._latest_teleop,
            odom=self._latest_odom,
            obstacles=self._latest_obs,
        )
        if command is None:
            return

        command.header.stamp = self.get_clock().now().to_msg()
        self._last_steering = float(command.drive.steering_angle)
        self.pub.publish(command)

    # =========================================================================
    # TASK 2 — Safety filter implementation
    # =========================================================================
    #
    # Start as a passthrough, then add real safety logic.
    #
    # You are free to pick any approach (or your own). Add helper methods,
    # extra parameters, even a sub-folder of modules — this skeleton will
    # get out of the way.
    # =========================================================================

    def safety_filter(self, teleop, odom, obstacles):
        """
        Args
        ----
        teleop : ackermann_msgs.msg.AckermannDriveStamped   (or None)
            Human's desired command. Useful fields:
                teleop.drive.speed           float   m/s     target forward speed
                teleop.drive.steering_angle  float   rad     target steering angle
                teleop.drive.acceleration    float   m/s^2   usually 0 from the joy
                teleop.header.stamp          Time            when the command was issued

        odom : nav_msgs.msg.Odometry                        (or None)
            Vehicle state. Useful fields:
                odom.pose.pose.position.x       float   m       map-frame x
                odom.pose.pose.position.y       float   m       map-frame y
                odom.pose.pose.position.z       float   m       usually 0
                odom.pose.pose.orientation      Quaternion (.x .y .z .w)
                    → use yaw_from_quat(..) above to get heading in rad
                odom.twist.twist.linear.x       float   m/s     forward velocity
                odom.twist.twist.angular.z      float   rad/s   yaw rate

        obstacles : visualization_msgs.msg.MarkerArray      (or None)
            Static obstacles (cubes). Useful fields:
                obstacles.markers               list[Marker]
                for m in obstacles.markers:
                    m.pose.position.x / .y / .z float   m       obstacle center
                    m.scale.x / .y / .z         float   m       cube size (x=y=z typically)
                    m.id                        int             obstacle id
                    m.ns                        str             namespace

        Returns
        -------
        ackermann_msgs.msg.AckermannDriveStamped
            The command to publish on /drive. Set `.drive.speed` (m/s) and
            `.drive.steering_angle` (rad). Header.stamp is overwritten for you.
            Return None to skip publishing this tick.
        """
        # ---- TODO(Task 2): replace this passthrough ----
        if teleop is None:
            return None

        now = self._now_sec()
        human_speed = self._clip_speed(teleop.drive.speed)
        human_steering = self._clip_steering(teleop.drive.steering_angle)

        if self._is_stale(self._teleop_time, self.config.teleop_timeout_sec):
            self._log_periodic(now, "stale teleop; publishing stop")
            return self._make_command(teleop, 0.0, 0.0)

        if odom is None or self._is_stale(self._odom_time, self.config.odom_timeout_sec):
            self._log_periodic(now, "missing/stale odom; publishing failsafe")
            return self._make_command(
                teleop,
                self.config.stale_odom_speed,
                self.config.stale_odom_steering,
            )

        obstacle_msg = (
            None
            if self._is_stale(self._obstacle_time, self.config.obstacle_timeout_sec)
            else obstacles
        )
        path_msg = (
            None
            if self._is_stale(self._path_time, self.config.path_timeout_sec)
            else self._latest_path
        )

        state = state_from_odom(odom, self._last_steering)
        obstacle_list = obstacles_from_msg(
            obstacle_msg,
            self.config.obstacle_radius_buffer,
        )
        path = self._effective_path(state, human_speed, path_from_msg(path_msg))

        if self.config.require_path_for_lane_filter and len(path) < 2:
            self._log_periodic(now, "missing path; publishing failsafe")
            return self._make_command(
                teleop,
                self.config.no_solution_speed,
                human_steering if self.config.no_solution_keep_steering else 0.0,
            )

        safety_plan = self._check_current_control(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacle_list,
        )
        hard_forward_stop = self._has_hard_forward_stop(human_speed, state, obstacle_list)
        soft_forward_stop = self._has_soft_forward_stop(human_speed, state, obstacle_list)

        if safety_plan.margins.is_safe(self.config.soft_margin) and not soft_forward_stop:
            return self._make_command(teleop, human_speed, human_steering)

        optimal_plan = self.optimal_planner.plan(
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacle_list,
        )

        speed, steering = optimal_plan.first_command
        reason = optimal_plan.reason
        if hard_forward_stop or not optimal_plan.is_feasible:
            speed = self.config.no_solution_speed
            if not self.config.no_solution_keep_steering:
                steering = 0.0
            reason = "obstacle_brake" if hard_forward_stop else "stop_no_solution"

        self._log_periodic(
            now,
            "ilqr override "
            f"reason={reason} speed={speed:.2f} steer={steering:.2f} "
            f"lane={self._fmt_margin(optimal_plan.margins.lane)} "
            f"obs={self._fmt_margin(optimal_plan.margins.obstacle)} "
            f"cost={optimal_plan.cost:.2f}",
        )
        return self._make_command(teleop, speed, steering)

    def _check_current_control(
        self,
        human_speed,
        human_steering,
        state,
        path,
        obstacles,
    ) -> CandidatePlan:
        horizon = self.safety_planner._horizon_steps()
        commands = tuple((human_speed, human_steering) for _ in range(horizon))
        return self.safety_planner._evaluate(
            commands=commands,
            human_speed=human_speed,
            human_steering=human_steering,
            state=state,
            path=path,
            obstacles=obstacles,
        )

    def _effective_path(self, state, human_speed, path):
        if len(path) >= 2 or not self.config.fallback_path_enabled:
            return path
        return fallback_path_ahead(
            state=state,
            length=self.config.fallback_path_length,
            width=self.config.fallback_path_width,
            speed_limit=max(human_speed, self.config.min_projection_speed),
            samples=self.config.fallback_path_samples,
        )

    def _has_hard_forward_stop(self, human_speed, state, obstacles) -> bool:
        forward_margin = self._forward_margin(human_speed, state, obstacles)
        ttc = self._time_to_collision(human_speed, state, obstacles)
        return (
            forward_margin is not None
            and forward_margin <= self.config.emergency_brake_margin
        ) or (ttc is not None and ttc <= self.config.ttc_hard_sec)

    def _has_soft_forward_stop(self, human_speed, state, obstacles) -> bool:
        ttc = self._time_to_collision(human_speed, state, obstacles)
        return ttc is not None and ttc <= self.config.ttc_soft_sec

    def _forward_margin(self, human_speed, state, obstacles) -> Optional[float]:
        stopping = human_speed * human_speed / (2.0 * max(self.config.max_decel, 1e-3))
        obstacle_buffer = (
            self.config.vehicle_radius
            + self.config.localization_buffer
            + self.config.stopping_buffer
            + 0.5 * stopping
        )
        return forward_obstacle_margin(
            state=state,
            obstacles=obstacles,
            safety_buffer=obstacle_buffer,
            forward_width=self.config.forward_obstacle_width,
            max_distance=self.config.forward_obstacle_distance,
        )

    def _time_to_collision(self, human_speed, state, obstacles) -> Optional[float]:
        projected_state = replace(state, speed=max(state.speed, human_speed))
        return obstacle_time_to_collision(
            state=projected_state,
            obstacles=obstacles,
            safety_buffer=self.config.vehicle_radius + self.config.localization_buffer,
            forward_width=self.config.forward_obstacle_width,
            max_distance=self.config.forward_obstacle_distance,
        )

    def _make_command(self, source, speed, steering):
        command = copy.deepcopy(source)
        command.drive.speed = self._clip_speed(speed)
        command.drive.steering_angle = self._clip_steering(steering)
        return command

    def _clip_speed(self, speed: float) -> float:
        lower = -self.config.max_speed if self.config.allow_reverse else self.config.min_speed
        return max(lower, min(self.config.max_speed, float(speed)))

    def _clip_steering(self, steering: float) -> float:
        limit = self.config.max_steering_angle
        return max(-limit, min(limit, float(steering)))

    def _is_stale(self, stamp_sec: Optional[float], timeout_sec: float) -> bool:
        if stamp_sec is None:
            return True
        return self._now_sec() - stamp_sec > timeout_sec

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_periodic(self, now: float, message: str):
        if now - self._last_log_time >= self.config.log_period_sec:
            self.get_logger().info(message)
            self._last_log_time = now

    @staticmethod
    def _fmt_margin(margin: Optional[float]) -> str:
        return "none" if margin is None else f"{margin:.2f}"


def main(args=None):
    # rclpy.init(args=args)
    # node = SafetyFilterNode()
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()

    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


    # ===========================
    # Test any other safety filter here
    # ===========================

    # ILQR Safety Filter
    # ilqr_main(args=args)

    # # CBF-QP Safety Filter
    # cbf_qp_main(args=args)

    # CBF-Heuristic Safety Filter
    # cbf_heuristic_main(args=args)


if __name__ == '__main__':
    main()


# implement a simple heuristic based CBF safety filter
# such that it steers to center of the lane when the vehicle is going to going to go off the lane (with adjusted speed based on the distance to the lane)

# stop if obstcale is directly in front of the vehicle
# if dynamic obstacles are in the way, stop 

# simple implementation
