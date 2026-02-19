import rclpy

#ROS Related Imports
import rclpy.service
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as routePath
from .lanelet_wrapper import LaneletWrapper
from .utils import map_to_markerarray
import message_filters
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path as FsPath
from rcl_interfaces.msg import SetParametersResult
from scipy.spatial.transform import Rotation as R

from racecar_routing.srv import Plan

import numpy as np
import os

"""
Node that will initializes lanelets, and give paths given a pose and a goal
"""

class Routing(Node):
    def __init__(self):
        super().__init__('routing_node')

        #declaring parameters
        try:
            share_dir = FsPath(get_package_share_directory('racecar_routing'))
        except Exception as e:
            self.get_logger().error(f"Could not locate package 'Routing' share dir: {e}")
            share_dir = FsPath('.')  # dummy to avoid NameError
        
        default_map = share_dir / 'maps' / 'track.osm'

        self.declare_parameter('map_file', str(default_map)) 
        self.declare_parameter('odom_topic', '/slam_pose')  
        self.declare_parameter('lane_change_cost', 1.0)  

        #we do not take into account goal_heading yet, TBD
        self.declare_parameter('goal_with_heading', False)
        self.declare_parameter('clicked_goal', True)

        self.declare_parameter('routing_path_topic', '/Routing/Path')
        self.declare_parameter('routing_pathL_topic', '/Routing/PathL')
        self.declare_parameter('routing_pathR_topic', '/Routing/PathR')

        self.declare_parameter('goal_topic', '/goal_pose')


        #initializing parameters
        map_file = self.get_parameter('map_file').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.lane_change_cost = self.get_parameter('lane_change_cost').value
    
        self.goal_with_heading = self.get_parameter('goal_with_heading').value
        self.click_goal = self.get_parameter('clicked_goal').value

        self.routing_path_topic = self.get_parameter('routing_path_topic').value
        self.routing_pathL_topic = self.get_parameter('routing_pathL_topic').value
        self.routing_pathR_topic = self.get_parameter('routing_pathR_topic').value

        self.goal_topic = self.get_parameter('goal_topic').value

        #set up pub / sub
        self.setup_pubsub()

        #service for replanning manually
        self.setup_services()

        #callback for when parameters are updated (mainly goal_with_heading)
        self.param_callback_handler = self.add_on_set_parameters_callback(self.param_callback)

        #setup wrapper class
        self.lanelet_wrapper = LaneletWrapper(map_file, self)



    def setup_pubsub(self):
        if self.click_goal:
            self.path_pub = self.create_publisher(routePath, self.routing_path_topic, 10)
            self.pathL_pub = self.create_publisher(routePath, self.routing_pathL_topic, 10)
            self.pathR_pub = self.create_publisher(routePath, self.routing_pathR_topic, 10)
            self.odom_msg = None
            self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
            #default for location for 2d navGoals
            self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.replan_callback, 1)

    def setup_services(self):
        self.plan_srv = self.create_service(Plan, '/routing/plan', self.plan_srv_cb)


    #only changes goal_with_heading for now
    #doesn't really do anything right now 
    def param_callback(self, params):
        for p in params:
            if p.name == 'goal_with_heading':
                self.goal_with_heading = bool(p.value)
                self.get_logger().info(f"Set goal_with_heading to {self.goal_with_heading}")
        return SetParametersResult(successful=True) 

    def odom_callback(self, odom_msg):
        self.odom_msg = odom_msg
    

    #receive a request with start and goal, create a path
    def plan_srv_cb(self, input, response):
        response.path = self.plan_route(input.start, input.goal)
        return response
    
    #receive new goal, create new plan
    def replan_callback(self, goal_msg):
        if self.odom_msg is None:
            return
        cur_odom_msg = self.odom_msg

        start_x = cur_odom_msg.pose.pose.position.x
        start_y = cur_odom_msg.pose.pose.position.y

        q = [cur_odom_msg.pose.pose.orientation.x, cur_odom_msg.pose.pose.orientation.y, 
                cur_odom_msg.pose.pose.orientation.z, cur_odom_msg.pose.pose.orientation.w]
        
        start_yaw = R.from_quat(q).as_euler('xyz', degrees=False)[-1]
        start_pose = np.array([start_x, start_y, start_yaw])

        goal_x = goal_msg.pose.position.x
        goal_y = goal_msg.pose.position.y

        q = [goal_msg.pose.orientation.x, goal_msg.pose.orientation.y, 
                goal_msg.pose.orientation.z, goal_msg.pose.orientation.w]
        goal_yaw = R.from_quat(q).as_euler('xyz', degrees=False)[-1]
        goal_pose = np.array([goal_x, goal_y, goal_yaw])

        self.get_logger().info(f"Planning route from [{start_x:.2f}, {start_y:.2f}, {start_yaw:.2f}] to [{goal_x:.2f}, {goal_y:.2f}, {goal_yaw:.2f}]")

        planned_path, plannedL_path, plannedR_path = self.plan_route(start_pose, goal_pose, True, self.goal_with_heading, goal_msg.header)

        if planned_path is not None:
            self.path_pub.publish(planned_path)
            self.pathL_pub.publish(plannedL_path)
            self.pathR_pub.publish(plannedR_path)
        


    def plan_route(self, start_pose, goal_pose, start_with_heading = False, goal_with_heading = False, header = None):
        """
        Note: pose are [x, y, psi]
        """


        path = self.lanelet_wrapper.get_shortest_path(start_pose, goal_pose,start_with_heading, goal_with_heading, allow_lane_change = True)

        if path is None:
            self.get_logger().info("No Path Found")
            return None
        
        path_msg = routePath()
        pathL_msg = routePath()
        pathR_msg = routePath()
        if header is not None:
            path_msg.header = header
            pathL_msg.header = header
            pathR_msg.header = header
        else:
            path_msg.header.frame_id = 'map'
            pathL_msg.header.frame_id = 'map'
            pathR_msg.header.frame_id = 'map'
            time =  self.get_clock().now().to_msg()
            path_msg.header.stamp = time
            pathL_msg.header.stamp = time
            pathR_msg.header.stamp = time
        
        #waypoint is a list of [x, y, width_left, width_right, speed_limit]
        #note PoseStamped is usually position(x, y, z) and orientation x, y, z
        #but we are using the orientation as left_width, right_width, and speed limit
        for i, waypoint in enumerate(path):
            x_c = waypoint[0]
            y_c = waypoint[1]
            w_left  = waypoint[2]   # left width
            w_right = waypoint[3]   # right width
            speed   = waypoint[4]

            # Center pose
            temp = PoseStamped()
            temp.header = path_msg.header
            temp.pose.position.x = x_c
            temp.pose.position.y = y_c
            temp.pose.orientation.x = w_left   # storing metadata
            temp.pose.orientation.y = w_right
            temp.pose.orientation.z = speed

            # Compute tangent using raw waypoints
            if i == 0:
                # use next point to define forward direction
                x_next, y_next = path[i + 1][0], path[i + 1][1]
                dx = x_next - x_c
                dy = y_next - y_c
            else:
                dx = x_c - path[i - 1][0]
                dy = y_c - path[i - 1][1]
            dr = (dx * dx + dy * dy) ** 0.5
            if dr == 0.0:
                #default
                dx_n, dy_n = 1.0, 0.0
            else:
                dx_n, dy_n = -dy / dr, dx / dr

            # Left boundary
            tempL = PoseStamped()
            tempL.header = pathL_msg.header
            tempL.pose.position.x = x_c + dx_n * w_left
            tempL.pose.position.y = y_c + dy_n * w_left

            # Right boundary
            tempR = PoseStamped()
            tempR.header = pathR_msg.header
            tempR.pose.position.x = x_c - dx_n * w_right
            tempR.pose.position.y = y_c - dy_n * w_right

            # Append to paths
            path_msg.poses.append(temp)
            pathL_msg.poses.append(tempL)
            pathR_msg.poses.append(tempR)

            
        return path_msg, pathL_msg, pathR_msg
    



    


    
        



        