#!/usr/bin/env python3
"""
Sim-side static obstacle publisher.

Owns the /Obstacles/Static topic in simulation. Obstacles come from two
sources, both accumulated into the same MarkerArray:

  1. A yaml file listing (x, y) coordinates, loaded at startup.
  2. RViz "Publish Point" clicks (/clicked_point), appended on the fly.

Republishes the combined list at a fixed rate so markers don't expire.

The traffic_simulator's built-in random static obstacles should be disabled
(~num_static_obs: 0) so this node is the single source of truth.
"""

import os

import rclpy
from rclpy.node import Node
import yaml

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker


class StaticObstaclePublisher(Node):
    def __init__(self):
        super().__init__('static_obstacle_publisher_node')

        self.declare_parameter('obstacle_yaml', '')
        self.declare_parameter('static_obs_topic', '/Obstacles/Static')
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        self.declare_parameter('static_obs_size', 0.2)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('frame_id', 'map')

        # Ground plane is published on its OWN topic so the student's
        # safety filter can subscribe to /Obstacles/Static without seeing
        # a 30x30m "obstacle" spanning the whole map.
        self.declare_parameter('click_target_topic', '/fp_click_target')
        self.declare_parameter('ground_plane_enabled', True)
        self.declare_parameter('ground_plane_center_x', 3.0)
        self.declare_parameter('ground_plane_center_y', 3.0)
        self.declare_parameter('ground_plane_size', 30.0)

        yaml_path = self.get_parameter('obstacle_yaml').value
        obs_topic = self.get_parameter('static_obs_topic').value
        click_topic = self.get_parameter('clicked_point_topic').value
        click_target_topic = self.get_parameter('click_target_topic').value
        self.size = self.get_parameter('static_obs_size').value
        self.frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('publish_rate').value

        self.ground_plane_enabled = self.get_parameter('ground_plane_enabled').value
        self.gp_cx = self.get_parameter('ground_plane_center_x').value
        self.gp_cy = self.get_parameter('ground_plane_center_y').value
        self.gp_size = self.get_parameter('ground_plane_size').value

        self.obstacles = []  # list of (x, y)
        self._load_from_yaml(yaml_path)

        self.pub = self.create_publisher(MarkerArray, obs_topic, 1)
        self.click_target_pub = self.create_publisher(MarkerArray, click_target_topic, 1)
        self.create_subscription(PointStamped, click_topic, self._click_cb, 1)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"static_obstacle_publisher ready: "
            f"{len(self.obstacles)} preset + clicks on {click_topic} "
            f"-> {obs_topic}"
        )

    def _load_from_yaml(self, path):
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            for xy in data.get('obstacles', []):
                if len(xy) >= 2:
                    self.obstacles.append((float(xy[0]), float(xy[1])))
        except Exception as e:
            self.get_logger().warn(f"Failed to load {path}: {e}")

    def _click_cb(self, msg: PointStamped):
        self.obstacles.append((msg.point.x, msg.point.y))
        self.get_logger().info(
            f"Added obstacle #{len(self.obstacles)} at "
            f"({msg.point.x:.2f}, {msg.point.y:.2f})"
        )

    def _publish(self):
        now = self.get_clock().now().to_msg()

        if self.ground_plane_enabled:
            ct_arr = MarkerArray()
            gp = Marker()
            gp.header.frame_id = self.frame_id
            gp.header.stamp = now
            gp.ns = 'fp_click_target'
            gp.id = 0
            gp.type = Marker.CUBE
            gp.action = Marker.ADD
            gp.pose.position.x = float(self.gp_cx)
            gp.pose.position.y = float(self.gp_cy)
            gp.pose.position.z = 0.01   # top face at z=0.02 → above lane
            gp.pose.orientation.w = 1.0  # geometry so TopDownOrtho hits it
            gp.scale.x = float(self.gp_size)
            gp.scale.y = float(self.gp_size)
            gp.scale.z = 0.02
            gp.color.r = 0.3
            gp.color.g = 0.3
            gp.color.b = 0.3
            gp.color.a = 0.05
            ct_arr.markers.append(gp)
            self.click_target_pub.publish(ct_arr)

        arr = MarkerArray()
        for i, (x, y) in enumerate(self.obstacles):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = 'fp_static_obs'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = self.size / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = self.size
            m.scale.y = self.size
            m.scale.z = self.size
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.6
            m.color.a = 0.8
            arr.markers.append(m)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
