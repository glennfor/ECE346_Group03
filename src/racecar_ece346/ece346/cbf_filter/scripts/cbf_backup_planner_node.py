#!/usr/bin/env python3
"""
backup_planner_node — rolls out the brake-and-recenter backup policy from the
current truck state and publishes the resulting trajectory.

Role in the 3-node CBF filter: FALLBACK
  "If we commit to the backup policy right now, where does the truck go?"

Consumers:
  safety_monitor_node  — reads /safety/backup_traj to evaluate safety margins
  safety_filter_qp_node — reads /safety/backup_u0 as the infeasibility fallback

Published topics:
  /safety/backup_traj   Float64MultiArray  — (H+1)*5 floats, row-major states
  /safety/backup_u0     Float64MultiArray  — [a, omega], first backup action
  /safety/backup_path   nav_msgs/Path      — for rviz visualization
"""
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ece346.cbf_filter.cbf_filter.backup_policy import brake_and_recenter
from ece346.cbf_filter.cbf_filter.config import CbfParams, declare_and_load
from ece346.cbf_filter.cbf_filter.dynamics import rollout
from ece346.cbf_filter.cbf_filter.lane_context import LaneContext, LaneletContextBuilder
from ece346.cbf_filter.cbf_filter.ros_utils import odom_to_state


class BackupPlannerNode(Node):
    def __init__(self):
        super().__init__("backup_planner_node")
        self.params: CbfParams = declare_and_load(self)

        if not self.params.map_file:
            try:
                self.params.map_file = (
                    get_package_share_directory("racecar_routing") + "/maps/track.osm"
                )
            except Exception:
                pass  # stays as ""; fallback straight lane will be used

        self.delta_estimate = 0.0
        self.last_state: np.ndarray = None
        self.last_lane_center: np.ndarray = None
        self.last_lane_yaw: float = None

        self.lane_context: LaneContext = LaneContext.fallback_straight()
        self.lane_builder = LaneletContextBuilder(
            self.params.map_file, self, self.params.lane_change_cost
        )

        self.odom_sub = self.create_subscription(
            Odometry, self.params.odom_topic, self._odom_cb, 10
        )
        self.traj_pub = self.create_publisher(Float64MultiArray, "/safety/backup_traj", 1)
        self.u0_pub = self.create_publisher(Float64MultiArray, "/safety/backup_u0", 1)
        self.path_pub = self.create_publisher(Path, "/safety/backup_path", 1)

        self.timer = self.create_timer(1.0 / self.params.control_rate_hz, self._step)
        self.get_logger().info("backup_planner_node ready")

    def _odom_cb(self, msg: Odometry):
        self.last_state = odom_to_state(msg, self.delta_estimate, self.params)
        self._maybe_rebuild_lane(self.last_state)

    def _step(self):
        if self.last_state is None:
            return

        state = self.last_state.copy()
        lane = self.lane_context

        try:
            traj = rollout(state, lambda x: brake_and_recenter(x, lane, self.params), self.params)
            u0 = brake_and_recenter(state, lane, self.params)
            self._pub_traj(traj)
            self._pub_u0(u0)
            self._pub_path(traj)
        except Exception:
            self.get_logger().error(f"backup_planner fault:\n{traceback.format_exc()}")

    def _maybe_rebuild_lane(self, state: np.ndarray):
        p = state[:2]
        if self.last_lane_center is not None:
            if (np.linalg.norm(p - self.last_lane_center) < self.params.lane_context_rebuild_distance_m
                    and abs((state[3] - self.last_lane_yaw + np.pi) % (2 * np.pi) - np.pi)
                    < self.params.lane_context_rebuild_yaw_rad):
                return
        try:
            self.lane_context = self.lane_builder.build_near(state)
            self.last_lane_center = p.copy()
            self.last_lane_yaw = float(state[3])
        except Exception as e:
            self.get_logger().warn(f"lane rebuild failed: {e}")

    def _pub_traj(self, traj: np.ndarray):
        msg = Float64MultiArray()
        msg.data = list(traj.flatten())
        self.traj_pub.publish(msg)

    def _pub_u0(self, u0: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(u0[0]), float(u0[1])]
        self.u0_pub.publish(msg)

    def _pub_path(self, traj: np.ndarray):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for s in traj:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(s[0])
            pose.pose.position.y = float(s[1])
            pose.pose.orientation.z = float(np.sin(s[3] / 2.0))
            pose.pose.orientation.w = float(np.cos(s[3] / 2.0))
            msg.poses.append(pose)
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BackupPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
