#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
import queue

from .utils.generate_pwm import GeneratePwm
from .utils.policy import Policy
from .utils.realtime_buffer import RealtimeBuffer

from .ILQR.ref_path import RefPath
from .ILQR.ilqr import ILQR

from rclpy.time import Time
from rclpy.clock import Clock

from racecar_msgs.msg import ServoMsg

#for packages
from ament_index_python.packages import get_package_share_directory


from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg # used to display the trajectory on RVIZ
from std_srvs.srv import Empty

#dynamic reconfigure imports
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from rcl_interfaces.msg import SetParametersResult

from ece346.Lab2.scripts.ILQR.config import Config

# You will use the imports below for lab3   
from racecar_msgs.msg import OdometryArray
from ece346.Lab3.scripts.utils.dyn_obstacle import frs_to_obstacle, frs_to_msg
from ece346.Lab3.scripts.utils.static_obstacle import get_obstacle_vertices
from visualization_msgs.msg import MarkerArray
from racecar_ece346.srv import GetFRS  

class TrajectoryPlanner(Node):
    '''
    Main class for the Receding Horizon trajectory planner
    '''

    def __init__(self):
        super().__init__("traj_planning_node")
        # Indicate if the planner is used to generate a new trajectory
        self.update_lock = threading.Lock()

        self.package = get_package_share_directory("racecar_ece346")
        
        self.declare_read_parameters()
        
        # Initialize the PWM converter
        self.pwm_converter = GeneratePwm(self.min_throttle, self.max_throttle, self.pwm_model)
        
        # set up the optimal control solver
        self.setup_planner()
        
        self.setup_publisher()
        
        self.setup_subscriber()

        self.setup_service()

        # start planning and control thread
        threading.Thread(target=self.control_thread, daemon=True).start()
        if not self.receding_horizon:
            threading.Thread(target=self.policy_planning_thread, daemon=True).start()
        else:
            threading.Thread(target=self.receding_horizon_planning_thread, daemon=True).start()
    
    def declare_read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        # declaring parameters:

        #can be changed by dynamic reconfig
        self.declare_parameter('truck_latency', 0.0, descriptor = 
                               ParameterDescriptor(
                                   name = 'truck_latency',
                                   type = 3,
                                   description = 'trucks latency to receive control, send control, and send position',
                                   read_only = False,
                                   floating_point_range = [FloatingPointRange(
                                       from_value = 0.0,
                                       to_value= 3.0,
                                       step= 0.0
                                   )]
                               )) 

        self.declare_parameter('odom_topic', 'slam_pose')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('static_obstacles_topic', '/Obstacles/Static')
        self.declare_parameter('traj_topic', '/Planning/Trajectory')
        self.declare_parameter('planning_start', '/Planning/Start')
        self.declare_parameter('planning_stop', '/Planning/Stop')
        self.declare_parameter('routing_path_topic', '/Routing/Path')
        self.declare_parameter('package_path', self.package)
        self.declare_parameter('simulation', True)
        self.declare_parameter('receding_horizon', False)
        self.declare_parameter('replan_dt', 0.1)
        self.declare_parameter('ilqr_params_file', os.path.join(self.package, "config", "lab3_ilqr.yaml"))
        self.declare_parameter('PWM_model', os.path.join(self.package, "config", "mlp_model.sav"))
        self.declare_parameter('max_throttle', 0.5)
        self.declare_parameter('min_throttle', -0.35)
   

        #reading parameters
        self.latency = self.get_parameter('truck_latency').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.min_throttle = self.get_parameter('min_throttle').value
        self.package_path = self.get_parameter('package_path').value
        self.receding_horizon = self.get_parameter('receding_horizon').value

        # Read ROS topic names to subscribe 
        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('routing_path_topic').value

        
        # Read ROS topic names to publish
        self.control_topic = self.get_parameter('control_topic').value
        self.traj_topic = self.get_parameter('traj_topic').value
        #Lab 3 Task 1.1 Comment if Lab 2
        self.static_obs_topic = self.get_parameter('static_obstacles_topic').value
        
        # Read the simulation flag, 
        # if the flag is true, we are in simulation 
        # and no need to convert the throttle and steering angle to PWM
        self.simulation = self.get_parameter('simulation').value

        #read service topics to start and stop plannig
        self.start_srv_name = self.get_parameter('planning_start').value
        self.stop_srv_name = self.get_parameter('planning_stop').value
        
        # Read Planning parameters
        # if true, the planner will load a path from a file rather than subscribing to a path topic           
        self.replan_dt = self.get_parameter('replan_dt').value
        
        # Read the ILQR parameters file, if empty, use the default parameters
        self.ilqr_params_abs_path = self.get_parameter('ilqr_params_file').value
 
        self.pwm_model = self.get_parameter('PWM_model').value
    def setup_planner(self):
        '''
        This function setup the ILQR solver
        '''
        # Initialize ILQR solver
        self.planner = ILQR(logger = self.get_logger(), config_file = self.ilqr_params_abs_path)

        # create buffers to handle multi-threading
        self.plan_state_buffer = RealtimeBuffer()
        self.control_state_buffer = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()

        #Lab 3 Task 1.3 Comment if Lab 2
        self.static_obstacle_dict = {}

        # Indicate if the planner is ready to generate a new trajectory
        self.planner_ready = True

    def setup_publisher(self):
        '''
        This function sets up the publisher for the trajectory
        '''
        # Publisher for the planned nominal trajectory for visualization
        self.trajectory_pub = self.create_publisher(PathMsg, self.traj_topic, 1)

        # Publisher for the control command
        self.control_pub = self.create_publisher(ServoMsg, self.control_topic, 1)

        #Lab 3, Publishes for FRS Visualization
        self.frs_pub = self.create_publisher(MarkerArray, 'vis/FRS', 1)


    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.odometry_callback, 10)
        self.path_sub = self.create_subscription(PathMsg, self.path_topic, self.path_callback, 10)

        #Lab 3 Task 1.2 Comment if Lab 2
        self.static_obs_sub = self.create_subscription(MarkerArray, self.static_obs_topic, self.static_obstacle_callback, 10)

    #Lab 3 Task 1.4 Comment if Lab 2
    def static_obstacle_callback(self, msg):
        '''
        Static obstacle callback function
        '''
        if self.simulation:
            self.static_obstacle_dict.clear()

        for obs in msg.markers:
            id, vertices = get_obstacle_vertices(obs)
            self.static_obstacle_dict[id] = vertices


    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = self.create_service(Empty, self.start_srv_name, self.start_planning_cb)
        self.stop_srv = self.create_service(Empty, self.stop_srv_name, self.stop_planning_cb)

        #change parameters we designate (i.e truck_latency)
        self.add_on_set_parameters_callback(self._on_params)

        #lab 3 Task 3 TBD
        self.frs_client = self.create_client(GetFRS, '/obstacles/get_frs')

    def get_frs(self, t_list):
        req = GetFRS.Request()
        req.t_list = list(t_list)
        try:
            response = self.frs_client.call(req)   # synchronous call
            return response
        except Exception as e:
            self.get_logger().warn(f"FRS service call failed inside get_frs(): {e}")
            return None

    def frs_available(self):
        return self.frs_client.wait_for_service(timeout_sec=0.0)

    def start_planning_cb(self, req, res):
        '''
        ros service callback function for start planning
        '''
        self.get_logger().info('start_planning!')
        self.planner_ready = True
        return Empty.Response()

    def stop_planning_cb(self, req):
        '''
        ros service callback function for stop planning
        '''
        self.get_logger().info('Stop planning!')
        self.planner_ready = False
        self.policy_buffer.reset()
        return Empty.Response()
    
    def _on_params(self, params):
        self.update_lock.acquire(blocking=True)
        for p in params:
            if p.name == 'truck_latency':
                self.latency = p.value

        self.get_logger().info(f"Latency Updated to {self.latency} s")
        #shouldn't be possible due to our parameter constraints
        if self.latency < 0.0:
            self.get_logger().info(f"Negative latency compensation {self.latency} is not a good idea!")
        self.update_lock.release()

        return SetParametersResult(successful = True)
    
    def odometry_callback(self, odom_msg):
        '''
        Subscriber callback function of the robot pose
        '''
        # Add the current state to the buffer
        # Controller thread will read from the buffer
        # Then it will be processed and add to the planner buffer 
        # inside the controller thread
        self.control_state_buffer.writeFromNonRT(odom_msg)
    
    def path_callback(self, path_msg):
        x = []
        y = []
        width_L = []
        width_R = []
        speed_limit = []
        
        for waypoint in path_msg.poses:
            x.append(waypoint.pose.position.x)
            y.append(waypoint.pose.position.y)
            width_L.append(waypoint.pose.orientation.x)
            width_R.append(waypoint.pose.orientation.y)
            speed_limit.append(waypoint.pose.orientation.z)
                    
        centerline = np.array([x, y])
        
        try:
            ref_path = RefPath(centerline, width_L, width_R, speed_limit, loop=False)
            self.path_buffer.writeFromNonRT(ref_path)
            self.get_logger().info('Path received!')
        except:
            self.get_logger().warn('Invalid path received! Move your robot and retry!')

    @staticmethod
    def compute_control(x, x_ref, u_ref, K_closed_loop):
        '''
        Given the current state, reference trajectory, control command 
        and closed loop gain, compute the control command
        
        Args:
            x: np.ndarray, [dim_x] current state
            x_ref: np.ndarray, [dim_x] reference trajectory
            u_ref: np.ndarray, [dim_u] reference control command
            K_closed_loop: np.ndarray, [dim_u, dim_x] closed loop gain

        Returns:
            accel: float, acceleration command [m/s^2]
            steer_rate: float, steering rate command [rad/s]
        '''

        ###############################
        #### TODO: Task 2 #############
        ###############################
        # Implement your control law here using ILQR policy
        # Hint: make sure that the difference in heading is between [-pi, pi]
        # but make sure that the angle is still preserved (e.g. do something
        # with np.mod() to make sure x_diff[3] is in the right range)
        
        dx = x - x_ref
        dx[3] = np.mod(dx[3] + np.pi, 2 * np.pi) - np.pi
        u = u_ref + K_closed_loop @ dx
        accel = u[0]
        steer_rate = u[1]

        ##### END OF TODO ##############

        return accel, steer_rate
  
    def control_thread(self):
        '''
        Main control thread to publish control command
        '''
        rate = self.create_rate(40) #40 Hz
        u_queue = queue.Queue()
        
        # values to keep track of the previous control command
        prev_state = None #[x, y, v, psi, delta]
        prev_u = np.zeros(3) # [accel, steer, t]
        
        # helper function to compute the next state
        def dyn_step(x, u, dt):
            dx = np.array([x[2]*np.cos(x[3]),
                        x[2]*np.sin(x[3]),
                        u[0],
                        x[2]*np.tan(u[1]*1.1)/0.257,
                        0
                        ])
            x_new = x + dx*dt
            x_new[2] = max(0, x_new[2]) # do not allow negative velocity
            x_new[3] = np.mod(x_new[3] + np.pi, 2 * np.pi) - np.pi
            x_new[-1] = u[1]
            return x_new
        
        while rclpy.ok():
            # initialize the control command
            accel = -5.0
            steer = 0.0
            state_cur = None
            policy = self.policy_buffer.readFromRT()
            
            # take the latency of publish into the account
            if self.simulation:
                t_act = self.get_clock().now().nanoseconds * 1e-9
            else:
                self.update_lock.acquire()
                t_act = (self.get_clock().now().nanoseconds * 1e-9) + self.latency 
                self.update_lock.release()
            
            # check if there is new state available
            if self.control_state_buffer.new_data_available:
                odom_msg = self.control_state_buffer.readFromRT()
                slam_time = odom_msg.header.stamp
                t_slam = slam_time.sec +  slam_time.nanosec * 1e-9
                
                u = np.zeros(3)
                u[-1] = t_slam
                while not u_queue.empty() and u_queue.queue[0][-1] < t_slam:
                    u = u_queue.get() # remove old control commands
                
                # get the state from the odometry message
                q = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, 
                        odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
                # get the heading angle from the quaternion
                psi = R.from_quat(q).as_euler('xyz', degrees=False)[-1]
                
                state_cur = np.array([
                            odom_msg.pose.pose.position.x,
                            odom_msg.pose.pose.position.y,
                            odom_msg.twist.twist.linear.x,
                            psi,
                            u[1]
                        ])
               
                # predict the current state use past control command
                for i in range(u_queue.qsize()):
                    u_next = u_queue.queue[i]
                    dt = u_next[-1] - u[-1]
                    state_cur = dyn_step(state_cur, u, dt)
                    u = u_next
                    
                # predict the cur state with the most recent control command
                state_cur = dyn_step(state_cur, u, t_act - u[-1])
                
                # update the state buffer for the planning thread
                plan_state = np.append(state_cur, t_act)
                self.plan_state_buffer.writeFromNonRT(plan_state)
                
            # if there is no new state available, we do one step forward integration to predict the state
            elif prev_state is not None:
                t_prev = prev_u[-1]
                dt = t_act - t_prev
                # predict the state using the last control command is executed
                state_cur = dyn_step(prev_state, prev_u, dt)
            
            # Generate control command from the policy
            if policy is not None:
                # get policy
                if not self.receding_horizon:
                    state_ref, u_ref, K = policy.get_policy_by_state(state_cur)
                else:
                    state_ref, u_ref, K = policy.get_policy(t_act)

                if state_ref is not None:
                    accel, steer_rate = self.compute_control(state_cur, state_ref, u_ref, K)
                    steer = max(-0.37, min(0.37, prev_u[1] + steer_rate*dt))
                else:
                    # reset the policy buffer if the policy is not valid
                    self.get_logger().warn("Try to retrieve a policy beyond the horizon! Reset the policy buffer!")
                    self.policy_buffer.reset()
                        
            # generate control command
            if not self.simulation and state_cur is not None:
                # If we are using robot,
                # the throttle and steering angle needs to convert to PWM signal
                throttle_pwm, steer_pwm = self.pwm_converter.convert(accel, steer, state_cur[2])
            else:
                throttle_pwm = accel
                steer_pwm = steer                
            
            # publish control command
            servo_msg = ServoMsg()
            control_time = self.get_clock().now().to_msg()
            servo_msg.header.stamp = control_time
            servo_msg.throttle = throttle_pwm
            servo_msg.steer = steer_pwm
            self.control_pub.publish(servo_msg)
            
            # Record the control command and state for next iteration
            u_record = np.array([accel, steer, t_act])
            u_queue.put(u_record)            
            prev_u = u_record
            prev_state = state_cur

            # end of while loop
            rate.sleep()

    def policy_planning_thread(self):
        '''
        This function is the main thread for open loop planning
        We plan entire trajectory (policy) everytime when a new reference path is available
        '''
        self.get_logger().info('Policy Planning thread started waiting for ROS service calls...')
        while rclpy.ok():
            # determine if we need to replan
            if self.path_buffer.new_data_available and self.planner_ready:
                new_path = self.path_buffer.readFromRT()
                self.planner.update_ref_path(new_path)
                
                # check if there is an existing policy
                original_policy = self.policy_buffer.readFromRT()
                if original_policy is not None:
                    # reset the buffer, which will stop the car from moving
                    self.policy_buffer.reset() 
                    time.sleep(2) # wait for 2 seconds to make sure the car is stopped
                self.get_logger().info('Planning a new policy...')
                # Get current state
                state = self.plan_state_buffer.readFromRT()[:-1] # the last element is the time
                prev_progress = -np.inf
                _, _, progress = new_path.get_closest_pts(state[:2])
                progress = progress[0]
                
                nominal_trajectory = []
                nominal_controls = []
                K_closed_loop = []
                
                # stop when the progress is not increasing
                while (progress - prev_progress)*new_path.length > 1e-3: # stop when the progress is not increasing
                    nominal_trajectory.append(state)
                    new_plan = self.planner.plan(state, None)
                    nominal_controls.append(new_plan['controls'][:,0])
                    K_closed_loop.append(new_plan['K_closed_loop'][:,:,0])
                    
                    # get the next state and its progress
                    state = new_plan['trajectory'][:,1]
                    prev_progress = progress
                    _, _, progress = new_path.get_closest_pts(state[:2])
                    progress = progress[0]
                    print('Planning progress %.4f' % progress, end='\r')

                nominal_trajectory = np.array(nominal_trajectory).T # (dim_x, N)
                nominal_controls = np.array(nominal_controls).T # (dim_u, N)
                K_closed_loop = np.transpose(np.array(K_closed_loop), (1,2,0)) # (dim_u, dim_x, N)
                
                T = nominal_trajectory.shape[-1] # number of time steps
                t0 = self.get_clock().now().nanoseconds * 1e-9

                # If stop planning is called, we will not write to the buffer
                new_policy = Policy(X = nominal_trajectory, 
                                    U = nominal_controls,
                                    K = K_closed_loop, 
                                    t0 = t0, 
                                    dt = self.planner.dt,
                                    T = T)
                
                self.policy_buffer.writeFromNonRT(new_policy)
                
                self.get_logger().info('Finish planning a new policy...')
                
                # publish the new policy for RVIZ visualization
                self.trajectory_pub.publish(new_policy.to_msg())        

    def receding_horizon_planning_thread(self):
        '''
        This function is the main thread for receding horizon planning
        We repeatedly call ILQR to replan the trajectory (policy) once the new state is available
        '''
        
        self.get_logger().info('Receding Horizon Planning thread started waiting for ROS service calls...')
        t_last_replan = 0
        while rclpy.ok():
            # determine if we need to replan
            if self.plan_state_buffer.new_data_available:
                state_cur = self.plan_state_buffer.readFromRT()
                
                t_cur = state_cur[-1] # the last element is the time
                dt = t_cur - t_last_replan
                
                # Do replanning
                if dt >= self.replan_dt:
                    # Get the initial controls for hot start
                    init_controls = None

                    original_policy = self.policy_buffer.readFromRT()
                    if original_policy is not None:
                        init_controls = original_policy.get_ref_controls(t_cur)

                    # Update the path
                    if self.path_buffer.new_data_available:
                        self.get_logger().info("setting new reference path")
                        new_path = self.path_buffer.readFromRT()
                        self.planner.update_ref_path(new_path)
                    
                    #Lab 3 task 2.1 - 2.3
                    # # Update the static obstacles
                    obstacles_list = []
                    for vertices in self.static_obstacle_dict.values():
                        obstacles_list.append(vertices)


                    #Lab 3
                    # update dynamic obstacles
                    # Check if the service is up
                    if not self.frs_available():
                        self.get_logger().warn("FRS server not available!")
                        frs_respond = None
                    else:
                        try:
                            t_list = t_cur + np.arange(self.planner.T) * self.planner.dt
                            frs_respond = self.get_frs(t_list)
                            if frs_respond is None:
                                raise RuntimeError("FRS returned None.")

                            obstacles_list.extend(frs_to_obstacle(frs_respond))
                            self.frs_pub.publish(frs_to_msg(frs_respond))
                        except Exception as e:
                            self.get_logger().warn(f"FRS service call failed: {e}")
                            frs_respond = None

                        
                    self.planner.update_obstacles(obstacles_list)
                    
                    # Replan use ilqr
                    new_plan = self.planner.plan(state_cur[:-1], init_controls)
                    
                    plan_status = new_plan['status']
                    if plan_status == -1:
                        #self.get_logger().info('No path specified!')
                        continue
                    
                    
                    if self.planner_ready:
                        #self.get_logger().info('NEW_PLAN: %s' % new_plan)
                        # If stop planning is called, we will not write to the buffer
                        new_policy = Policy(X = new_plan['trajectory'], 
                                            U = new_plan['controls'],
                                            K = new_plan['K_closed_loop'], 
                                            t0 = t_cur, 
                                            dt = self.planner.dt,
                                            T = self.planner.T)
                        
                        self.policy_buffer.writeFromNonRT(new_policy)
                        
                        # publish the new policy for RVIZ visualization
                        self.trajectory_pub.publish(new_policy.to_msg())        
                        t_last_replan = t_cur
            ###############################
            #### TODO: Task 3 #############
            ###############################

            '''
            Implement the receding horizon planning thread
            Hint: Make sure you are familiar with the <Policy> class in utils/policy.py
            1. Determine if we need to replan by
                - checking if there is new data in the plan_state_buffer using 
                    <self.plan_state_buffer.new_data_available>
                - checking if the time since <t_last_replan> is larger than <self.replan_dt>
                - checking if <self.planner_ready> is True
            2. If we need to replan, 
                - Get the current state from the plan_state_buffer using <self.plan_state_buffer.readFromRT>
                - Get the previous policy from the policy_buffer using <self.policy_buffer.readFromRT>
                - Get the initial controls for hot start if there is a previous policy
                    you can use helper function <get_ref_controls> in the <Policy> class
                - Check if there is a new path in the path_buffer using <self.path_buffer.new_data_available>.
                    if true, Update the reference path in ILQR using <self.planner.update_ref_path(new path)>
                - Replan using ILQR 
            3. If the replan is successful,
                - Create a new <Policy> object using your new plan
                - Write the new policy to the policy buffer using <self.policy_buffer.writeFromNonRT>
                - Publish the new policy for RVIZ visualization
                    for example: self.trajectory_pub.publish(new_policy.to_msg())       
            '''
            ###############################
            #### END OF TODO #############
            ###############################
            time.sleep(0.01)