#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from builtin_interfaces.msg import Time as msg_Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from racecar_msgs.msg import OdometryArray
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.executors import ExternalShutdownException

from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
"""
Publishes all of the visuals to RVIZ
"""
class Truckvis(Node):
    def __init__(self):

        #calling prent node constructor Node
        super().__init__('visualization_node')


        #Parameters
        self.declare_parameter('odom_topic', '/slam_pose') 
        self.declare_parameter('dyn_obs_topic', '/Obstacles/Dynamic')

        self.odom_topic = self.get_parameter('odom_topic').value
        self.dyn_obs_topic = self.get_parameter('dyn_obs_topic').value



        #subscribers
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.odometry_callback, 10)
        self.dyn_obs_sub = self.create_subscription(OdometryArray, self.dyn_obs_topic, self.dyn_obs_callback, 1)

        #publishers
        self.car_pub = self.create_publisher(MarkerArray, 'vis/truck', 10)
        self.origin_pub = self.create_publisher(PoseStamped, 'vis/origin', 1)
        self.playground_pub = self.create_publisher(Marker, 'vis/playground', 1)
        self.dyn_obs_pub = self.create_publisher(MarkerArray, '/vis/dyn_obs', 1)


    def dyn_obs_callback(self, msg):
        color = [204/255.0, 51/255.0, 0/255.0,  0.5]
        msg_to_pub = MarkerArray()
        for i, obs in enumerate(msg.odometry_array):
            self.visualize_car(obs, 'obs', i, color, msg_to_pub)
        self.dyn_obs_pub.publish(msg_to_pub)

    def odometry_callback(self, msg):
        #x_pos = msg.pose.pose.position.x 
        #y_pos = msg.pose.pose.position.y 
        #self.get_logger().info(f"x: {x_pos}, y: {y_pos}")
        color = [255/255.0, 165/255.0, 15/255.0,  0.5]
        msg_to_pub = MarkerArray()
        self.visualize_car(msg, 'ego', 0, color, msg_to_pub)
        self.car_pub.publish(msg_to_pub)
        
        self.visualize_origin()
        self.visualize_playground()


    def visualize_origin(self):
        marker = PoseStamped()
        marker.header.frame_id = 'map'
        time =  self.get_clock().now().seconds_nanoseconds()
        msg_time = msg_Time()
        msg_time.sec = time[0]
        msg_time.nanosec = time[1]
        marker.header.stamp = msg_time
        if rclpy.ok():
            self.origin_pub.publish(marker)

    #Publish the borders of the map
    def visualize_playground(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        time =  self.get_clock().now().seconds_nanoseconds()
        msg_time = msg_Time()
        msg_time.sec = time[0]
        msg_time.nanosec = time[1]
        marker.header.stamp = msg_time
        
        marker.type = 4 # Line Strip
        marker.ns = 'playground'
        marker.id = 0
        marker.action = 0 # ADD/modify
        marker.scale.x = 0.02
        
        pt1 = Point()
        pt1.x = 0.0
        pt1.y = 0.0
        pt1.z = 0.0
        
        pt2 = Point()
        pt2.x = 6.1
        pt2.y = 0.0
        pt2.z = 0.0
        
        pt3 = Point()
        pt3.x = 6.1
        pt3.y = 6.1
        pt3.z = 0.0
        
        pt4 = Point()
        pt4.x = 0.0
        pt4.y = 6.1
        pt4.z = 0.0
        
        marker.points = [pt1, pt2, pt3, pt4, pt1]
        
        marker.color.r = 204/255.0
        marker.color.g = 0/255.0
        marker.color.b = 0/255.0
        marker.color.a = 1.0
        if rclpy.ok():
            self.playground_pub.publish(marker)
        


    def visualize_car(self, msg, ns, id, color, marker_array):
        cuboid = Marker()
        cuboid.header = msg.header
        cuboid.ns = ns
        cuboid.id = id
        cuboid.type = 1 # CUBE
        cuboid.action = 0 # ADD/modify existing marker
        cuboid.scale.x = 0.42
        cuboid.scale.y = 0.19
        cuboid.scale.z = 0.188

        cuboid.pose = copy.deepcopy(msg.pose.pose)
    
        q = [cuboid.pose.orientation.x, cuboid.pose.orientation.y,
                cuboid.pose.orientation.z, cuboid.pose.orientation.w]
        

        yaw = R.from_quat(q).as_euler('xyz', degrees=False)[-1]
        cuboid.pose.position.x += 0.1285*np.cos(yaw)
        cuboid.pose.position.y += 0.1285*np.sin(yaw)
        cuboid.pose.position.z = 0.0 
        
        # ORANGE
        cuboid.color.r = color[0]
        cuboid.color.g = color[1]
        cuboid.color.b = color[2]
        cuboid.color.a = 0.5
        cuboid.lifetime = Duration(sec = 0, nanosec = 0)
        marker_array.markers.append(cuboid)
        
        # Create the arrow marker
        arrow = Marker()
        arrow.header = msg.header 
        arrow.ns =  ns+"_arrow"
        arrow.id = id
        arrow.type = 0 # ARROW
        arrow.action = 0 # ADD/modify
        arrow.pose = copy.deepcopy(msg.pose.pose)
        
        arrow.scale.x = 0.3
        arrow.scale.y = 0.02
        arrow.scale.z = 0.02
        
        arrow.color.r = 255/255.0
        arrow.color.g = 255/255.0
        arrow.color.b = 255/255.0
        arrow.color.a = 1.0
        
        arrow.lifetime = Duration(sec = 0, nanosec = 0)
        marker_array.markers.append(arrow)



def main(args = None):
    rclpy.init(args=args)

    visualization_node = Truckvis()

    visualization_node.get_logger().info("Started Visualization Node")

    try:
        rclpy.spin(visualization_node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException: #context shutdown from outside, ros2 launch
        pass
    except Exception as e:
        #if shutdown race
        if not rclpy.ok():
            pass
        else:
            raise
    finally:
        visualization_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()






