#!/usr/bin/env python3
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray


class SafetyVizNode(Node):
    def __init__(self):
        super().__init__("safety_viz_node")
        self.declare_parameter("backup_traj_topic", "/safety/backup_traj")
        self.declare_parameter("value_topic", "/safety/value")
        self.declare_parameter("override_topic", "/safety/override")
        self.declare_parameter("marker_topic", "/safety/markers")

        self.backup_traj_topic = self.get_parameter("backup_traj_topic").value
        self.value_topic = self.get_parameter("value_topic").value
        self.override_topic = self.get_parameter("override_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value

        self.latest_path = None
        self.latest_value = 0.0
        self.latest_override = False

        self.create_subscription(Path, self.backup_traj_topic, self.path_callback, 10)
        self.create_subscription(Float32, self.value_topic, self.value_callback, 10)
        self.create_subscription(Bool, self.override_topic, self.override_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.timer = self.create_timer(0.1, self.publish_markers)

    def path_callback(self, msg: Path):
        self.latest_path = msg

    def value_callback(self, msg: Float32):
        self.latest_value = msg.data

    def override_callback(self, msg: Bool):
        self.latest_override = msg.data

    def publish_markers(self):
        markers = MarkerArray()
        if self.latest_path is not None:
            markers.markers.append(self._trajectory_marker())
            markers.markers.append(self._text_marker())
        self.marker_pub.publish(markers)

    def _trajectory_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_backup"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color.a = 1.0
        marker.color.r = 1.0 if self.latest_override else 0.1
        marker.color.g = 0.1 if self.latest_override else 0.8
        marker.color.b = 0.1
        marker.points = [pose.pose.position for pose in self.latest_path.poses]
        return marker

    def _text_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_status"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        if self.latest_path.poses:
            pose = self.latest_path.poses[0].pose.position
            marker.pose.position.x = pose.x
            marker.pose.position.y = pose.y
            marker.pose.position.z = 0.5
        marker.text = f"h={self.latest_value:.3f} override={self.latest_override}"
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = SafetyVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

