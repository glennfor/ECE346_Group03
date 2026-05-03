#!/usr/bin/env python3
"""
ILQR Safety Filter Visualization Node.

Subscribes to:
  /safety/plan_path     (nav_msgs/Path)   — ILQR plan trajectory
  /safety/V_hat         (std_msgs/Float32) — safety certificate value
  /safety/override      (std_msgs/Bool)    — arbiter override flag
  /safety/binding_constraint (std_msgs/String)

Publishes:
  /safety/markers  (visualization_msgs/MarkerArray) — plan + status text
"""
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


class ILQRSafetyVizNode(Node):
    def __init__(self):
        super().__init__("ilqr_safety_viz_node")
        self.declare_parameter("plan_path_topic", "/safety/plan_path")
        self.declare_parameter("vhat_topic", "/safety/V_hat")
        self.declare_parameter("override_topic", "/safety/override")
        self.declare_parameter("binding_topic", "/safety/binding_constraint")
        self.declare_parameter("marker_topic", "/safety/markers")

        plan_topic = self.get_parameter("plan_path_topic").value
        vhat_topic = self.get_parameter("vhat_topic").value
        override_topic = self.get_parameter("override_topic").value
        binding_topic = self.get_parameter("binding_topic").value
        marker_topic = self.get_parameter("marker_topic").value

        self.latest_path: Path = None
        self.latest_vhat: float = 0.0
        self.latest_override: bool = False
        self.latest_binding: str = "—"

        self.create_subscription(Path, plan_topic, self.path_callback, 10)
        self.create_subscription(Float32, vhat_topic, self.vhat_callback, 10)
        self.create_subscription(Bool, override_topic, self.override_callback, 10)
        self.create_subscription(String, binding_topic, self.binding_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 1)
        self.create_timer(0.1, self.publish_markers)

    def path_callback(self, msg: Path):
        self.latest_path = msg

    def vhat_callback(self, msg: Float32):
        self.latest_vhat = float(msg.data)

    def override_callback(self, msg: Bool):
        self.latest_override = bool(msg.data)

    def binding_callback(self, msg: String):
        self.latest_binding = str(msg.data)

    def publish_markers(self):
        markers = MarkerArray()
        if self.latest_path is not None and len(self.latest_path.poses) > 1:
            markers.markers.append(self._trajectory_marker())
            markers.markers.append(self._text_marker())
        self.marker_pub.publish(markers)

    def _trajectory_marker(self) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ilqr_plan"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.a = 0.9
        # Red when overriding, green when safe
        if self.latest_override:
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
        else:
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.2
        m.points = [pose.pose.position for pose in self.latest_path.poses]
        return m

    def _text_marker(self) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ilqr_status"
        m.id = 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.z = 0.20
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        if self.latest_path.poses:
            p = self.latest_path.poses[0].pose.position
            m.pose.position.x = p.x
            m.pose.position.y = p.y
            m.pose.position.z = 0.4
        status = "OVERRIDE" if self.latest_override else "safe"
        m.text = f"V̂={self.latest_vhat:.3f} [{status}] bind={self.latest_binding}"
        return m


def main(args=None):
    rclpy.init(args=args)
    node = ILQRSafetyVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
