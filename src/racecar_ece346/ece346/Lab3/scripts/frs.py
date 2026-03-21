from ece346.Lab3.scripts.quickzonoreach.zono import onestep_zonotope_reachset, zono_from_box
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
import pickle
from routing.routing.lanelet_wrapper import LaneletWrapper
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from pathlib import Path as FsPath



def multistep_zonotope_reachset(init_box, a_mat, b_mat, input_box, dt_list, quick=True):
    '''
    Generator for multiple step reachable set for a zonotope at dt
    params:
        init_box: bounding box of the initial state, 
            np.ndarray [N,2] for N dimensions state space, 
            each row is [min, max] value of the state space, 
        a_mat: a matrices for dynamics, [N,N] np.ndarray
        b_mat: b matrices for dynamics, [N,M] np.ndarray for M-dimensional input space
        input_box: list of input boxes, [M,2] np.ndarray for M-dimensional input space
            each row is [min, max] value of the input space 
        dt_list: list of time steps. 
            Note: i-th dt in this list is the time difference i and i-1 steps of reachable set
    return: 
        reachable_set_list: a list of reachable sets, each reachable set is a zonotope
    '''
    reachable_set_list = []
    # Create a zonotope to represent the initial state
    init_z = zono_from_box(init_box)

    ############################
    #### TODO ##################
    # Use helper function <onestep_zonotope_reachset> to get the reachable set for each time step
    # This function have following parameters:
    # Input:
    #     init_z: initial set represented as a zonotope
    #     a_mat: a matrices for dynamics, [N,N] np.ndarray for N-dimensional state space
    #     b_mat: b matricesfor dynamics, [N,M] np.ndarray for M-dimensional input space
    #     input_box: list of input boxes, [M,2] np.ndarray for M-dimensional input space
    #     dt: time step
    #     quick: if True, use the quick method, otherwise use the Kamenev method. Default: False
    # Output:
    #     z: the reachable set as a zonotope
    ############################

    z_cur = init_z
    for dt in dt_list:
        # use only + valyes since - values don't make sense for dt 
        dt = max(float(dt), 0.0)
        z_cur = onestep_zonotope_reachset(
            z_cur, a_mat, b_mat, input_box, dt, quick=quick
        )
        reachable_set_list.append(z_cur)

    return reachable_set_list
        

class FRS():
    def __init__(self, node, map_file = None):
        # State-space model  
        # State: [x,y, vx, vy, vx_ref]]    
        self.a_mat = np.array([[0, 0, 1, 0, 0], 
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])
        self.b_mat = np.array([[0, 0],
                [0, 0],
                [1,0],
                [0,1],
                [0,0]])
        
        # Get 
        share_dir = FsPath(get_package_share_directory('racecar_routing'))

        if map_file is None:
            self.map_file = str(share_dir / 'maps' / 'track.osm')
        self.lane_change_cost = 1.0
        self.lanelet_wrapper = LaneletWrapper(self.map_file, node)
    
    def route_length(self, route):
        diffs = np.diff(route, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(seg_lengths)

    def sample_route(self, route, s_arr):
        # normalize s_arr from 0→1 → convert to arclength
        diffs = np.diff(route, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)

        # Replace zero-length segments with epsilon
        seg_lens = np.where(seg_lens < 1e-6, 1e-6, seg_lens)

        cum = np.hstack([[0], np.cumsum(seg_lens)])
        total = cum[-1]

        arc_positions = s_arr * total  # convert normalized

        pts = []
        for s in arc_positions:
            idx = np.searchsorted(cum, s) - 1
            idx = np.clip(idx, 0, len(diffs) - 1)

            # Safe interpolation
            t = (s - cum[idx]) / seg_lens[idx]
            t = np.clip(t, 0, 1)

            pt = route[idx] * (1 - t) + route[idx + 1] * t
            pts.append(pt)

        return np.array(pts)


    
    def route_derivative(self, route, s_arr):
        diffs = np.diff(route, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.hstack([[0], np.cumsum(seg_lens)])
        total = cum[-1]

        slopes = []
        for s in s_arr * total:
            idx = np.searchsorted(cum, s) - 1
            idx = np.clip(idx, 0, len(diffs) - 1)
            dx, dy = diffs[idx]
            slopes.append([dx, dy])
        return np.array(slopes)

    def get(self, state: Odometry, t_list, K_vx, K_vy, K_y, dx, dy, v_ref = None, allow_lane_change=True):
        '''
        Get the zonotope reachable set for given time steps in cartesian coordinates
        Parameters:
            t_list: [N,] np.ndarray, time to calculate the reachable set [s] 
                Note: this is the absolute wall time in your ROS system
            K_vx: float, gain for longitudinal velocity
            K_vy: float, gain for lateral velocity
            K_y: float, gain for lateral state
            v_ref: float, reference velocity
            dx: float, disturbance in x direction [m/s^2]
            dy: float, disturbance in y direction [m/s^2]
            allow_lane_change: bool, whether to allow lane change
                Note: if True, the reachable set will be calculated for 
                    1) the closest lanelet
                    2) the neighboring lanelets
                    If False, the reachable set will be calculated for
                    1) the closest lanelet
                    2) the neighboring lanelets if the vehicle cross the lanelet boundary
        Returns:
            time_varying_vertices: list of N reachable sets,
                    each reachable set is list of T polygons. Each polygon is a (P,2) np.ndarray
        '''
        # Initialize
        if not isinstance(t_list, np.ndarray):
            t_list = np.array(t_list)
            
        reachable_sets_list = []
        
        # get pose of the vehicle
        x = state.pose.pose.position.x
        y = state.pose.pose.position.y
        
        q = [state.pose.pose.orientation.x, state.pose.pose.orientation.y, 
                state.pose.pose.orientation.z, state.pose.pose.orientation.w]
        
        r = R.from_quat(q)
        psi = r.as_euler('xyz', degrees=False)[-1]
        pose = np.array([x, y, psi])

        # add reference lane into the list
        ref_lanelet_list = []
        
        # get lateral distance to the lane
        closest_lanelet, arc = self.lanelet_wrapper.get_closest_lanelet(pose, check_psi=True)

        # get the normalized s ∈ [0,1]
        lane_length = LaneletWrapper.get_lanelet_length(closest_lanelet)

        s_start = arc.length / lane_length if lane_length > 0 else 0.0
        s_start = min(max(s_start, 0.0), 0.999999)

        d_lateral = self.lanelet_wrapper.distance_to_centerline(closest_lanelet, pose)
        ref_lanelet_list.append((closest_lanelet.id, s_start, d_lateral))

        if allow_lane_change:
            for left_lanelet in self.lanelet_wrapper.routing_graph.lefts(closest_lanelet):
                d_lateral, s_start = self.lanelet_wrapper.normalized_arc_distance(left_lanelet, pose[:2])
                ref_lanelet_list.append((left_lanelet.id, s_start, d_lateral))

            for right_lanelet in self.lanelet_wrapper.routing_graph.rights(closest_lanelet):
                d_lateral, s_start = self.lanelet_wrapper.normalized_arc_distance(right_lanelet, pose[:2])
                ref_lanelet_list.append((right_lanelet.id, s_start, d_lateral))

        elif d_lateral > 0.05: # if vehicle overleap the left lane boundary
            for right_lanelet in self.lanelet_wrapper.routing_graph.rights(closest_lanelet):
                d_lateral, s_start  = self.lanelet_wrapper.normalized_arc_distance(right_lanelet, pose[:2])
                ref_lanelet_list.append((right_lanelet.id, s_start, d_lateral))

        elif d_lateral < -0.05: #if vehicle overleap the left lane boundary
            for left_lanelet in self.lanelet_wrapper.routing_graph.lefts(closest_lanelet):
                d_lateral, s_start = self.lanelet_wrapper.normalized_arc_distance(left_lanelet, pose[:2])
                ref_lanelet_list.append((left_lanelet.id, s_start, d_lateral))


        for (start_id, s_start, d_lateral) in ref_lanelet_list:
            cur_lanelet = self.lanelet_wrapper.lanelet_layer[start_id]

            # construct the closed loop state space model
            k_mat = np.array([[0, 0, K_vx, 0, -K_vx],
                    [0, K_y, 0, K_vy, 0]])

            a_hat_mat = self.a_mat - self.b_mat@k_mat

            # construct the disturbance matrix
            input_box = [[-dx, dx], [-dy, dy]]

            # Decompose body-frame velocity into Frenet frame (along-lane / cross-lane).
            # The odometry twist is in the body frame, but the FRS dynamics
            # operate in the Frenet frame of the lanelet.
            centerline = np.array([(pt.x, pt.y) for pt in cur_lanelet.centerline])
            tangent = self.route_derivative(centerline, np.array([s_start]))
            lane_heading = np.arctan2(tangent[0, 1], tangent[0, 0])
            alpha = (psi - lane_heading + np.pi) % (2 * np.pi) - np.pi

            vx_body = state.twist.twist.linear.x
            vy_body = state.twist.twist.linear.y
            vx_cur = vx_body * np.cos(alpha) - vy_body * np.sin(alpha)
            vy_cur = vx_body * np.sin(alpha) + vy_body * np.cos(alpha)
            if v_ref is None:# if no reference velocity is given, use current velocity
                v_ref = vx_cur 
            init_box = [[-1e-3, 1e-3], 
                        [d_lateral-1e-3, d_lateral+1e-3], 
                        [vx_cur,vx_cur], 
                        [vy_cur,vy_cur], 
                        [v_ref, v_ref]]
            
            a_mat_list = []
            b_mat_list = []
            input_box_list = []

            stamp_time = state.header.stamp.sec + state.header.stamp.nanosec * 1e-9
            dt_list = (t_list - stamp_time).tolist()
            dt_list[1:] = np.diff(dt_list)

            # print("dt_list: ", dt_list)
            for _ in range(len(dt_list)):
                a_mat_list.append(a_hat_mat)
                b_mat_list.append(self.b_mat)
                input_box_list.append(input_box)
            
            # get the zonotope in frenet frame
            zonotopes = multistep_zonotope_reachset(init_box, a_hat_mat, self.b_mat, input_box, dt_list, quick=True)
            
            # max in x
            d_max = np.max(np.array(zonotopes[-1].verts())[:,0])+0.3
            reachable_routes = self.lanelet_wrapper.get_reachable_path(cur_lanelet, s_start, d_max, False)
            
            # Map back to the global frame

            for route in reachable_routes:
                reachable_sets = []
                for zonotope in zonotopes:
                    # Reachable set is rectangular
                    x_min = np.min(np.array(zonotope.verts())[:,0])-0.1
                    x_max = np.max(np.array(zonotope.verts())[:,0])+0.3
                    y_min = np.min(np.array(zonotope.verts())[:,1])-0.1
                    y_max = np.max(np.array(zonotope.verts())[:,1])+0.1
                

                    L = self.route_length(route)
                    s_min = x_min / L
                    s_max = x_max / L
                    
                    s_samples = np.linspace(s_min, s_max, 10)
                    
                    sampled_pt = self.sample_route(route, s_samples) #[N,2]
                    deri = self.route_derivative(route, s_samples)
                    slope = np.arctan2(deri[:,1], deri[:,0])
                    
                    right_bound = np.array([sampled_pt[:,0] + np.sin(slope)*y_min,
                            sampled_pt[:,1] - np.cos(slope)*y_min]).T #[N,2]
                    
                    left_bound = np.flip(np.array([sampled_pt[:,0] + np.sin(slope)*y_max,
                                        sampled_pt[:,1] - np.cos(slope)*y_max]).T, axis = 0) #[N,2]
                    
                    reachable_sets.append(np.concatenate([right_bound, left_bound], axis = 0))
                reachable_sets_list.append(reachable_sets)
            
        return reachable_sets_list
    