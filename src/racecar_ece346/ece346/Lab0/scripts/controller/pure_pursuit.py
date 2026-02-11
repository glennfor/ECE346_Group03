#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
import numpy as np

from ece346.Lab0.scripts.controller.utils.generate_pwm import GeneratePwm
from ece346.Lab0.scripts.controller.utils.realtime_buffer import RealtimeBuffer
from ece346.Lab0.scripts.controller.utils.state_2d import State2D

from racecar_msgs.msg import ServoMsg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult

from rclpy.time import Time
from rclpy.clock import Clock


class PurePursuitController(Node):
    '''
    Main class for the controller
    '''
    def __init__(self):
        super().__init__("pure_pursuit_controller_node")
        '''
        Constructor for the PurePursuitController class
        '''
        
        # Read Parameters from the parameter server
        self.declare_read_parameters()

        # Initialize the PWM converter
        self.pwm_converter = GeneratePwm([self.min_throttle, self.max_throttle])

        # Initialize the real-time buffer for the state and goal
        self.state_buffer = RealtimeBuffer()
        self.goal_buffer = RealtimeBuffer()

        

        # Allow dynamic reconfigure to update parameters at runtime
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Setup the publisher and subscriber
        self.setup_publisher()
        self.setup_subscriber()

        # start planning thread
        # We use a thread to run the planning loop so that we can continuously read the odometry
        # and goal message in the callback function without being blocked by the planning loop
        threading.Thread(target=self.planning_thread, daemon = True).start()

    def declare_read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''

        #max / min throttle
        self.declare_parameter('max_throttle', 0.5)
        self.declare_parameter('min_throttle', -0.35)
        self.max_throttle = self.get_parameter('max_throttle').value
        self.min_throttle = self.get_parameter('min_throttle').value

        # ROS topic name to subscribe odometry 
        self.declare_parameter('odom_topic', '/slam_pose')
        self.odom_topic = self.get_parameter('odom_topic').value
        
        # Read ROS topic name to publish control command
        self.declare_parameter('control_topic', '/control')
        self.control_topic = self.get_parameter('control_topic').value
        
        # Proportional gain for the throttle
        self.declare_parameter('throttle_gain', 0.05)
        self.throttle_gain = self.get_parameter('throttle_gain').value
        
        # maximum look ahead distance
        self.declare_parameter('ld_max', 2)
        self.ld_max = self.get_parameter('ld_max').value
        
        # Read controller thresholds
        self.declare_parameter('max_steer', 0.35)
        self.declare_parameter('max_vel', 0.5)
        self.max_steer = self.get_parameter('max_steer').value
        self.max_vel = self.get_parameter('max_vel').value
        
        # Distance threshold to stop
        self.declare_parameter('stop_distance', 0.5)
        self.stop_distance = self.get_parameter('stop_distance').value
        
        # Read the wheel base of the robot
        self.declare_parameter('wheel_base', 0.257)
        self.wheel_base = self.get_parameter('wheel_base').value
        
        # Read the simulation flag, 
        # if the flag is true, we are in simulation 
        # and no need to convert the throttle and steering angle to PWM
        self.declare_parameter('simulation', True)
        self.simulation = self.get_parameter('simulation').value

    def parameter_callback(self, params):
        '''
        Callback for dynamic reconfigure — updates parameters at runtime
        '''
        for param in params:
            if param.name == 'max_throttle':
                self.max_throttle = param.value
            elif param.name == 'min_throttle':
                self.min_throttle = param.value
            elif param.name == 'throttle_gain':
                self.throttle_gain = param.value
            elif param.name == 'max_vel':
                self.max_vel = param.value
            elif param.name == 'max_steer':
                self.max_steer = param.value
            elif param.name == 'ld_max':
                self.ld_max = param.value
            elif param.name == 'stop_distance':
                self.stop_distance = param.value
            elif param.name == 'wheel_base':
                self.wheel_base = param.value
        self.get_logger().info(f"Parameters updated: {[p.name for p in params]}")
        return SetParametersResult(successful=True)

    def setup_publisher(self):
        '''
        This function sets up the publisher for the control command
        '''
        ################## TODO: 1. Set up a publisher for the ServoMsg message###################
        # Create a publisher - self.control_pub:
        #   - subscribes to the topic <self.control_topic>
        #   - has message type <ServoMsg> (racecar_msgs.msg.Odometry) 
        #   - with queue size 1
        self.control_pub = self.create_publisher(ServoMsg, self.control_topic, 1) # TO BE FILLED
        ########################### END OF TODO 1#################################
        
            
    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and goal message
        '''
        # This set up a subscriber for the goal you click on the rviz
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 1)
        
        ################## TODO: 2. Set up a subscriber for the odometry message###################
        # Create a subscriber:
        #   - subscribes to the topic <self.odom_topic>
        #   - has message type <Odometry> (nav_msgs.msg.Odometry) 
        #   - with callback function <self.odometry_callback>, which has already been implemented
        #   - with queue size 1
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.odometry_callback, 1) # TO BE FILLED
        ########################### END OF TODO 2#################################
        
    def odometry_callback(self, odom_msg: Odometry):
        """
        Subscriber callback function for the robot odometry message
        Parameters:
            odom_msg: Odometry message with type nav_msgs.msg.Odometry
        """
        # Retrieve state needed for planning from the odometry message
        # [x, y, v, w, delta]
        state_cur = State2D(odom_msg = odom_msg)

        # Add the current state to the buffer
        # Planning thread will read from the buffer
        self.state_buffer.writeFromNonRT(state_cur)

    def goal_callback(self, goal_msg: PoseStamped):
        """
        Subscriber callback function of the robot goal point
        Parameters:
            goal_msg: goal message with type geometry_msgs.msg.PoseStamped
        """
        ############## TODO: 3. Fill in the subscriber callback function ###################
        # 1. Inspect the data structure of the goal message <geometry_msgs.msg.PoseStamped>
        #   Hint: check the message data structure using the command
        #       ros2 interface show geometry_msgs/msg/PoseStamped
        # 2. Retrieve the goal from the goal message 
        #   and create a 3-dim numpy array [x,y,1]
        # 3. add the goal to the buffer (self.goal_buffer)
        
        goal_x = np.float64(goal_msg.position.x) # TO BE FILLED
        goal_y = np.float64(goal_msg.position.y) # TO BE FILLED
        goal = np.array([goal_x, goal_y,  1])
        self.goal_buffer.writeFromNonRT(goal)
        
        ########################### END OF TODO 3 #################################
        # Log the goal to the console using "self.get_logger().info()"
        self.get_logger().info(f"Received a new goal [{np.round(goal_x, 3)}, {np.round(goal_y,3)}]")

        
    def publish_control(self, accel, steer, state):
        '''
        Helper function to publish the control command
        Parameters:
            accel: float, linear acceleration of the robot [m/s^2]
            steer: float, steering angle of the robot [rad]
            state: State2D, current state of the robot
        '''
        # If we are in simulation,
        # the throttle and steering angle are acceleration and steering angle
        if self.simulation:
            throttle = accel
        else:
            # If we are using robot,
            # the throttle and steering angle needs to convert to PWM signal
            throttle, steer = self.pwm_converter.convert(accel, steer, state)
            
        ########################## TODO: 4. Construct and publish a ROS message ###################
        # 1. Create an empty ServoMsg message
        #   Hint: check the message data structure using the command
        #       ros2 interface show racecar_msgs/msg/ServoMsg
        # 2. Set the header time to the current time
        #   Hint: self.get_clock().now().to_msg()
        # 3. Set the throttle and steering angle to the servo message
        #   Hint: throttle and steer must be Python floats, use float() to cast
        
        servo_msg = ServoMsg()
        servo_msg.header.stamp = self.get_clock().now().to_msg()
        servo_msg.throttle = float(throttle)
        servo_msg.steer = float(steer)
        self.control_pub.publish(servo_msg)

        ########################### END OF TODO 4 #################################

    def planning_thread(self):
        '''
        Main thread for the planning
        '''
        self.get_logger().info("Planning thread started waiting for ROS service calls...")
        while rclpy.ok():
            # determine if we need to replan
            # if the current state is not None and a new state is available
            # we need to re-calculate the control command
            if self.state_buffer.new_data_available:

                # read the current state and goal from the buffer
                state_cur = self.state_buffer.readFromRT()
                goal_cur = self.goal_buffer.readFromRT()
                
                # current longitudinal velocity
                vel_cur = state_cur.v_long 

                # check if a goal is available
                if goal_cur is not None:
                    # First, transform the goal to the robot frame
                    goal_robot = np.linalg.inv(state_cur.transformation_matrix()).dot(goal_cur)
                    
                    # relative heading angle of the goal wrt the car
                    alpha = np.arctan2(goal_robot[1], goal_robot[0])
                    
                    # relative distance between the car and the goal
                    dis2goal = np.sqrt(goal_robot[0]**2 + goal_robot[1]**2)

                    ########################## TODO: 5. Finish the pure pursuit controlle ###################
                    # 1. Check if the goal is close enough
                    #
                    # 2. if that is the case, stop the car by apply a negative acceleration (eg: -1 m/s^2)
                    #   and zero steering angle. Then, continue to the next iteration
                    #
                    # 3. If the target is behind the car, apply maximum steering angle 
                    #   and set the reference_velocity to vel_max
                    # 
                    # 4. Otherwise, 
                    #   - apply a pure pursuit controller for steering by assuming 
                    #       the reference path a straight line between the car and the goal
                    #   - Hint: look-ahead distance l_d is min(self.ld_max, dis2goal)
                    #   - set the reference_velocity to the minimum of vel_max and (dis2goal-self.stop_distance)
                    #   Detail Explanation: 
                    #   https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
                    #
                    # 5. clip the steering angle between "-self.steer_max" and "self.steer_max"
                    # 6. apply the simple proportional controller for the acceleration to track the reference_velocity

                    if dis2goal < self.stop_distance: # too close
                        accel = -1.0 # TO BE FILLED
                        steer = 0.0 # TO BE FILLED
                    elif abs(alpha) > (np.pi/2): # check behind here
                        sign = -1 if alpha > 0 else 1
                        steer = sign * self.steer_max
                        vel_ref = self.max_vel
                    else: #continue normally
                        l_d = min(self.ld_max, dis2goal)
                        vel_ref = min(self.max_vel,(dis2goal-self.stop_distance))
                        accel = self.throttle_gain * vel_ref
                        steer = np.arctan(2 * self.wheel_base * np.sin(alpha)/ l_d)
                    steer = np.clip(-self.steer_max, self.steer_max)


                    ########################### END OF TODO 5 ###########################################
                    
                    # publish the control
                    self.publish_control(accel, steer, state_cur)