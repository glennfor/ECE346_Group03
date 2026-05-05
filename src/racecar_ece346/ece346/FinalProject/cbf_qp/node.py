"""ROS2 node for the CBF-QP safety filter."""

from dataclasses import fields, replace
from typing import Optional

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from .config import DEFAULT_CONFIG_PATH, CbfQpConfig, load_config
from .filter import CbfQpFilter, FilterCommand
from .geometry import obstacles_from_msg, path_from_msg, state_from_odom


class CbfQpSafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_node")

        self.declare_parameter("cbf_qp_config", str(DEFAULT_CONFIG_PATH))
        config_path = self.get_parameter("cbf_qp_config").value
        self.config = self._declare_and_read_config(load_config(config_path))
        self.filter = CbfQpFilter(self.config)

        self._latest_teleop: Optional[AckermannDriveStamped] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_obstacles: Optional[MarkerArray] = None
        self._latest_path: Optional[Path] = None
        self._teleop_time: Optional[float] = None
        self._odom_time: Optional[float] = None
        self._obstacle_time: Optional[float] = None
        self._path_time: Optional[float] = None
        self._last_log_time = 0.0

        self.pub = self.create_publisher(
            AckermannDriveStamped, self.config.drive_topic, 1
        )
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
        self.create_timer(1.0 / max(self.config.publish_rate, 1e-3), self._tick)

        self.get_logger().info(
            "cbf_qp safety_filter_node ready: "
            f"{self.config.teleop_topic} + {self.config.odom_topic} + "
            f"{self.config.static_obs_topic} + {self.config.routing_path_topic} "
            f"-> {self.config.drive_topic}"
        )

    def _declare_and_read_config(self, config: CbfQpConfig) -> CbfQpConfig:
        values = {}
        for field in fields(CbfQpConfig):
            default = getattr(config, field.name)
            self.declare_parameter(field.name, list(default) if isinstance(default, tuple) else default)
            value = self.get_parameter(field.name).value
            if isinstance(default, tuple) and isinstance(value, list):
                value = tuple(value)
            values[field.name] = value
        return replace(config, **values)

    def _teleop_cb(self, msg: AckermannDriveStamped):
        self._latest_teleop = msg
        self._teleop_time = self._now_sec()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_time = self._now_sec()

    def _obstacles_cb(self, msg: MarkerArray):
        self._latest_obstacles = msg
        self._obstacle_time = self._now_sec()

    def _path_cb(self, msg: Path):
        self._latest_path = msg
        self._path_time = self._now_sec()

    def _tick(self):
        if self._latest_teleop is None:
            return

        now = self._now_sec()
        if self._is_stale(self._teleop_time, self.config.teleop_timeout_sec):
            self._publish_drive(
                speed=0.0,
                steering=0.0,
                source=self._latest_teleop,
            )
            self._log_periodic(now, "stale teleop; publishing stop")
            return

        if self._latest_odom is None or self._is_stale(
            self._odom_time, self.config.odom_timeout_sec
        ):
            self._publish_drive(
                speed=self.config.stale_odom_speed,
                steering=self.config.stale_odom_steering,
                source=self._latest_teleop,
            )
            self._log_periodic(now, "missing/stale odom; publishing failsafe")
            return

        obstacle_msg = (
            None
            if self._is_stale(self._obstacle_time, self.config.obstacle_timeout_sec)
            else self._latest_obstacles
        )
        path_msg = (
            None
            if self._is_stale(self._path_time, self.config.path_timeout_sec)
            else self._latest_path
        )

        command = self.filter.filter_command(
            human_speed=self._latest_teleop.drive.speed,
            human_steering=self._latest_teleop.drive.steering_angle,
            state=state_from_odom(self._latest_odom),
            path=path_from_msg(path_msg),
            obstacles=obstacles_from_msg(
                obstacle_msg,
                self.config.obstacle_radius_buffer,
            ),
        )
        self._publish_drive(
            speed=command.speed,
            steering=command.steering_angle,
            source=self._latest_teleop,
        )
        if command.is_override:
            self._log_command(now, command)

    def _publish_drive(
        self,
        speed: float,
        steering: float,
        source: AckermannDriveStamped,
    ):
        out = AckermannDriveStamped()
        out.header = source.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive = source.drive
        out.drive.speed = float(speed)
        out.drive.steering_angle = float(steering)
        self.pub.publish(out)

    def _log_command(self, now: float, command: FilterCommand):
        lane = "none" if command.lane_margin is None else f"{command.lane_margin:.2f}"
        obs = (
            "none"
            if command.obstacle_margin is None
            else f"{command.obstacle_margin:.2f}"
        )
        self._log_periodic(
            now,
            "cbf override "
            f"reason={command.reason} speed={command.speed:.2f} "
            f"steer={command.steering_angle:.2f} lane={lane} obs={obs}",
        )

    def _log_periodic(self, now: float, message: str):
        if now - self._last_log_time >= self.config.log_period_sec:
            self.get_logger().info(message)
            self._last_log_time = now

    def _is_stale(self, stamp_sec: Optional[float], timeout_sec: float) -> bool:
        if stamp_sec is None:
            return True
        return self._now_sec() - stamp_sec > timeout_sec

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = CbfQpSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

