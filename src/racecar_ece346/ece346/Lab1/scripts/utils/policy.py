import numpy as np
import rclpy
from nav_msgs.msg import Path as TrajMsg # used to display the trajectory on RVIZ
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from rclpy.clock import Clock
from rclpy.time import Time
'''
    A container class to store the feedback policy
'''
class Policy():
    def __init__(self, X, U, K, t0, dt, T) -> None:
        self.nominal_X = X
        self.nominal_U = U
        self.K = K
        self.t0 = t0
        self.dt = dt
        self.T = T
        self.clock = Clock()
    
    def get_policy(self, t):
        '''
        Return the policy at time t
        '''
        i = self.get_index(t)
        if i>= (self.T-1):
            return None, None, None
        else:
            x_i = self.nominal_X[:,i]
            u_i = self.nominal_U[:,i]
            K_i = self.K[:,:,i]

        return x_i, u_i, K_i
    
    def get_policy_by_state(self, x):
        '''
        Return the policy at the closest state (Eculidean distance in x-y plane)
        '''
        distance = np.linalg.norm(self.nominal_X[:2,:].T-x[:2], axis=1)
        i = np.argmin(distance)
        
        if distance[i] > 0.2:
            return None, None, None
        
        x_i = self.nominal_X[:,i]
        u_i = self.nominal_U[:,i]
        K_i = self.K[:,:,i]
        
        return x_i, u_i, K_i
        
    
    def get_index(self,t):
        return int(np.ceil((t-self.t0)/self.dt))

    def get_ref_controls(self, t):
        '''
        Return the nominal control at time t and forward
        Args:
            t: float time in seconds
        Returns:
            ref_u: np.array of shape (dim_u, T) reference controls at time t and forward
        '''
        i = self.get_index(t)
        if i>= self.T:
            return None
        else:
            ref_u = np.zeros_like(self.nominal_U)
            ref_u[:,:self.T-i] = self.nominal_U[:,i:]

            return ref_u
        
    def to_msg(self, frame_id='map'):
        traj_msg = TrajMsg()
        traj_msg.header.frame_id = frame_id
        time =  self.clock.now().to_msg()
        traj_msg.header.stamp = time
        
        trajectory = self.nominal_X
        for i in range(self.T):
            t = self.t0 + i * self.dt
            pose = self.state_to_pose_stamped(trajectory[:,i], t, frame_id)
            traj_msg.poses.append(pose)

        return traj_msg
    
    @staticmethod
    def state_to_pose_stamped(state, t, frame_id='map'):
        '''
        Convert a State Vector object to a PoseStamped message
        state: [x,y,v,yaw,delta]
        t: float time in seconds
        frame_id: string
        '''

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = Time(seconds = t).to_msg()
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.position.z = 0.0
        
        q = R.from_euler('xyz', [0.0, 0.0, state[3]]).as_quat() 
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose
    
    def __str__(self) -> str:
        return f"Policy: t0: {self.t0}, dt: {self.dt}, N: {self.T}\n"+\
                f"nominal_X: {self.nominal_X}\n"+\
                f"nominal_U: {self.nominal_U}\n"