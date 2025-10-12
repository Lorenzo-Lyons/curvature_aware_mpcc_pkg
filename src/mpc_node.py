#!/usr/bin/env python3

# MPCC Control Node
#This node
# 1. It is subscribing to the following topics:
#   - optitrack_state, safety_value
# 2. It is publishing the following topics:
#   - des_roll, des_pitch, des_yaw, thrust, delta_thrust, slack, s_dot, comptime

# The node is doing the following:
# 1. It is computing the control inputs for the drone
# 2. It is publishing the control inputs to the drone
# 3. Updating the extra state for the thrust
# 4. It is publishing the computation time of the solver

import traceback
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rospy

import roslib.packages
import sys

pkg_path = roslib.packages.get_pkg_dir('curvature_aware_mpcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))     # or whatever sub‑folder holds the module
from reference_path_handeling_functions import generate_path_data


#from MPCC_ROS_workspace.src.curvature_aware_mpcc_pkg.src.reference_path_handeling_functions import generate_path_data

from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
import math
import tf

# import main solver handler functions
from solvers_setup.solver_manager_classes import MPC_solver_handler

# import acados (you must have it installed first)
from acados_template import AcadosOcpSolver

# for dynamic paramters reconfigure (setting param values from rqt_reconfigure GUI)
from dynamic_reconfigure.server import Server
from dynamic_reconfigure_pkg.cfg import Drone_MPCC_dynamic_reconfigureConfig
import copy
from scipy.spatial.transform import Rotation as R
from solvers_setup.drone_dynamic_model import evaluate_w

#for dynamic parameter change using rqt_reconfigure GUI
class MPC_GUI_manager:
    def __init__(self,drone_mpc_obj):
        self.drone_mpc_obj = drone_mpc_obj
        self.MPC_algorithm_options = drone_mpc_obj.MPC_algorithm_options #['MPCC', 'CAMPCC','MPCCPP','CAMPCC_EA']
        self.software_choice_options = drone_mpc_obj.software_choice_options #['acados', 'forcespro']

        # as a last thing creat the server because it will be locked executing here
        srv = Server(Drone_MPCC_dynamic_reconfigureConfig, self.reconfig_callback)

        
    def reconfig_callback(self, config, level):
        print('_________________________________________________')
        #print('  reconfiguring parameters from dynamic_reconfig ')
        #print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

        # Map of parameter names to current values
        param_map = {
            'q_sdot':       'q_sdot',
            'q_cont':       'q_cont',
            'q_lag':        'q_lag',
            'q_thrust':     'q_thrust',
            'q_roll_pitch':       'q_roll_pitch',
            'q_yaw':        'q_yaw',
            'qt_pos':       'qt_pos',
            'qt_v':       'qt_v',
            'qt_s':       'qt_s',
            'lane_radius':       'lane_radius',
            'local_path_length': 'local_path_length',
            'controller_type': 'controller_type',
            'software_choice': 'software_choice'
        }


        #solver_changed = False

        for cfg_key, attr in param_map.items():
            current_value = getattr(self.drone_mpc_obj, attr)
            new_value = config[cfg_key]

            if cfg_key == 'controller_type':
                new_value = self.MPC_algorithm_options[new_value]
            if cfg_key == 'software_choice':
                new_value = self.software_choice_options[new_value]
            
            if current_value != new_value:
                print(f"  {attr} changed: {current_value} → {new_value}")
                setattr(self.drone_mpc_obj, attr, new_value)
                

        #self.drone_mpc_obj.assemble_fixed_parameters()
        self.drone_mpc_obj.send_static_rviz_markers()


        print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
        return config





class path_handeling_utilities_class():
    def __init__(self):
        # this class collects the functions that relate to the reference path handleing for the MPC node
        pass
            

    def find_s_of_closest_point_on_global_path_3d(self,
                                                xyz_state, 
                                                s_vals_global_path, 
                                                x_vals_global_path, 
                                                y_vals_global_path, 
                                                z_vals_global_path,
                                                previous_index, 
                                                estimated_ds):
        
        min_ds = np.min(np.diff(s_vals_global_path))
        estimated_index_jumps = math.ceil(estimated_ds / min_ds)
        minimum_index_jumps = math.ceil(0.1 / min_ds)

        if estimated_index_jumps < minimum_index_jumps:
            estimated_index_jumps = minimum_index_jumps

        Delta_indexes = estimated_index_jumps * 3
        start_i = previous_index - Delta_indexes
        finish_i = previous_index + Delta_indexes

        # Wrap-around indexing logic
        if start_i < 0:
            s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[:finish_i]), axis=0)
            x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[:finish_i]), axis=0)
            y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[:finish_i]), axis=0)
            z_search_vector = np.concatenate((z_vals_global_path[start_i:], z_vals_global_path[:finish_i]), axis=0)
        elif finish_i > s_vals_global_path.size:
            s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[:finish_i - s_vals_global_path.size]), axis=0)
            x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[:finish_i - s_vals_global_path.size]), axis=0)
            y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[:finish_i - s_vals_global_path.size]), axis=0)
            z_search_vector = np.concatenate((z_vals_global_path[start_i:], z_vals_global_path[:finish_i - s_vals_global_path.size]), axis=0)
        else:
            s_search_vector = s_vals_global_path[start_i:finish_i]
            x_search_vector = x_vals_global_path[start_i:finish_i]
            y_search_vector = y_vals_global_path[start_i:finish_i]
            z_search_vector = z_vals_global_path[start_i:finish_i]

        # Distance in 3D
        distances = np.array([
            math.dist([x_search_vector[ii], y_search_vector[ii], z_search_vector[ii]], xyz_state)
            for ii in range(s_search_vector.size)
        ])

        local_index = np.argmin(distances)
        dist_to_state = math.dist([x_search_vector[local_index], y_search_vector[local_index], z_search_vector[local_index]], xyz_state)

        if local_index == 0 or local_index == s_search_vector.size - 1 or dist_to_state > 1:
            distances_full = np.array([
                math.dist([x_vals_global_path[ii], y_vals_global_path[ii], z_vals_global_path[ii]], xyz_state)
                for ii in range(s_vals_global_path.size)
            ])
            index = np.argmin(distances_full)
        else:
            index = np.where(s_vals_global_path == s_search_vector[local_index])[0][0]

        s = float(s_vals_global_path[index])
        dist_to_centerline = math.dist([x_vals_global_path[index], y_vals_global_path[index], z_vals_global_path[index]], xyz_state)

        xyz_closest_point = np.array([x_vals_global_path[index], y_vals_global_path[index], z_vals_global_path[index]])
        return s, index, dist_to_centerline, xyz_closest_point



    def relative_xyyaw_to_current_path(self, drone_xyz_roll_pitch_yaw, s):
        # produce vector of the current path in the point s
        # current_path_index_on_4_local_path = np.argmin(np.abs(self.s_4_local_path - s))
        # x_ref_path = self.x_4_local_path[current_path_index_on_4_local_path]
        # y_ref_path = self.y_4_local_path[current_path_index_on_4_local_path]
        # z_ref_path = self.z_4_local_path[current_path_index_on_4_local_path]
        # dx_ds_ref_path = self.dx_ds[current_path_index_on_4_local_path]
        # dy_ds_ref_path = self.dy_ds[current_path_index_on_4_local_path]
        # dz_ds_ref_path = self.dz_ds[current_path_index_on_4_local_path]


        # path_xyz_rpy = np.array([x_ref_path, y_ref_path, z_ref_path, dx_ds_ref_path, dy_ds_ref_path, dz_ds_ref_path])

        # Step 1: Get reference path index and position
        current_path_index = np.argmin(np.abs(self.s_4_local_path - s))
        x_ref = self.x_4_local_path[current_path_index]
        y_ref = self.y_4_local_path[current_path_index]
        z_ref = self.z_4_local_path[current_path_index]

        # Step 2: Compute the tangent vector (X-axis of body frame)
        dx_ds = self.dx_ds[current_path_index]
        dy_ds = self.dy_ds[current_path_index]
        dz_ds = self.dz_ds[current_path_index]
        tangent = np.array([dx_ds, dy_ds, dz_ds])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])  # fallback direction
        else:
            tangent /= tangent_norm
        x_axis = tangent

        # Step 3: Compute Z-axis lying in same plane as global Z ([0, 0, 1])
        z_global = np.array([0, 0, 1])
        z_proj = z_global - np.dot(z_global, x_axis) * x_axis
        if np.linalg.norm(z_proj) < 1e-6:
            # x_axis is nearly vertical, choose arbitrary horizontal z_axis
            if not np.allclose(x_axis, [0, 1, 0]):
                z_axis = np.cross(x_axis, [0, 1, 0])
            else:
                z_axis = np.cross(x_axis, [1, 0, 0])
        else:
            z_axis = z_proj / np.linalg.norm(z_proj)

        # Step 4: Compute Y-axis to complete right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Step 5: Assemble rotation matrix
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))  # columns: x, y, z

        # Step 6: Convert to roll, pitch, yaw
        rpy = R.from_matrix(rot_matrix).as_euler('xyz')

        # Step 7: Return pose [x, y, z, roll, pitch, yaw]
        path_xyz_rpy = np.array([x_ref, y_ref, z_ref, rpy[0], rpy[1], rpy[2]])



        # compute the relative position of the dron wrt the current path
        drone_relative_position, drone_relative_orientation = self.relative_pose(path_xyz_rpy, drone_xyz_roll_pitch_yaw)

        # put them together
        #drone_relative_pose = np.concatenate((drone_relative_position, drone_relative_orientation))
        drone_relative_pose = np.concatenate((drone_relative_position, drone_xyz_roll_pitch_yaw[3:]))

        return drone_relative_pose, path_xyz_rpy


    def relative_pose(self, pose_A, pose_B):
        """
        Compute the relative pose of B with respect to A.
        
        Args:
            pose_A: [x, y, z, roll, pitch, yaw] of reference frame A
            pose_B: [x, y, z, roll, pitch, yaw] of target frame B
        
        Returns:
            relative_position: [x, y, z] of B in A's frame
            relative_orientation: [roll, pitch, yaw] of B in A's frame
        """
        # Decompose poses
        pos_A = np.array(pose_A[:3])
        pos_B = np.array(pose_B[:3])
        rpy_A = pose_A[3:]
        rpy_B = pose_B[3:]

        # Rotations
        rot_A = R.from_euler('xyz', rpy_A)
        rot_B = R.from_euler('xyz', rpy_B)

        # Relative position in world frame
        delta_pos = pos_B - pos_A

        # Rotate delta_pos into A's local frame
        relative_position = rot_A.inv().apply(delta_pos)

        # Relative rotation: B relative to A
        relative_rotation = rot_A.inv() * rot_B
        relative_rpy = relative_rotation.as_euler('xyz')

        return relative_position, relative_rpy

    def produce_ylabels_4_local_kernelized_path(self,s,local_path_length,n,xyz_closest_point): # path_xyz_rpy
        #extract indexes of local path
        #Ds_back = 0
        # mask = (self.s_4_local_path >= s - Ds_back) & (self.s_4_local_path <= s + Ds_forward)
        #local_path_length = Ds_back + Ds_forward
        # Extract the indexes where the condition is true
        mask = (self.s_4_local_path >= s) & (self.s_4_local_path <= s + local_path_length)

        indexes = np.where(mask)[0]
        s_data_points =  self.s_4_local_path[indexes]  # This will have the local path parametrized starting from 0
        x_data_points = self.x_4_local_path[indexes] - xyz_closest_point[0]  # Center the path around the closest point
        y_data_points = self.y_4_local_path[indexes] - xyz_closest_point[1]  # Center the path around the closest point
        z_data_points = self.z_4_local_path[indexes] - xyz_closest_point[2]  # Center the path around the closest point
        dx_ds_data_points = self.dx_ds[indexes]
        dy_ds_data_points = self.dy_ds[indexes]
        dz_ds_data_points = self.dz_ds[indexes]
        d2x_ds2_data_points = self.d2x_ds2[indexes]
        d2y_ds2_data_points = self.d2y_ds2[indexes]
        d2z_ds2_data_points = self.d2z_ds2[indexes]
        self.k_data_points = self.k_vec[indexes]  # Curvature at the data points

        roll_ref_path = self.roll_4_local_path[indexes]
        pitch_ref_path = self.pitch_4_local_path[indexes]
        yaw_ref_path = self.yaw_4_local_path[indexes]

        # reference path pose (position and orientation) at the closest point
        wz_ref_data_points = self.wz_4_local_path[indexes]
        wx_ref_data_points = self.wx_4_local_path[indexes]
        
        # now rototranslate the x and y points to have the first point at the origin
        # x_t, y_t, z_t, dx_t, dy_t, dz_t = self.rototranslate_abs_2_path_frame(  x_data_points, y_data_points, z_data_points,
        #                                                                         dx_ds_data_points, dy_ds_data_points, dz_ds_data_points,
        #                                                                         path_xyz_rpy)

        # Normalize s and have the right number of labels
        labels_s = np.linspace(0, 1, n)
        s_interp = (s_data_points - s_data_points[0]) / local_path_length

        # Interpolate transformed quantities
        labels_x     = np.interp(labels_s, s_interp, x_data_points)
        labels_y     = np.interp(labels_s, s_interp, y_data_points)
        labels_z     = np.interp(labels_s, s_interp, z_data_points)
        labels_dxds  = np.interp(labels_s, s_interp, dx_ds_data_points)
        labels_dyds  = np.interp(labels_s, s_interp, dy_ds_data_points)
        labels_dzds  = np.interp(labels_s, s_interp, dz_ds_data_points)
        labels_d2xds2 = np.interp(labels_s, s_interp, d2x_ds2_data_points)
        labels_d2yds2 = np.interp(labels_s, s_interp, d2y_ds2_data_points)
        labels_d2zds2 = np.interp(labels_s, s_interp, d2z_ds2_data_points)
        labels_k = np.interp(labels_s, s_interp, self.k_data_points)  # Curvature at the data points
        labels_wz = np.interp(labels_s, s_interp, wz_ref_data_points)
        labels_wx = np.interp(labels_s, s_interp, wx_ref_data_points)
        labels_roll = np.interp(labels_s, s_interp, roll_ref_path)
        labels_pitch = np.interp(labels_s, s_interp, pitch_ref_path)
        labels_yaw = np.interp(labels_s, s_interp, yaw_ref_path)
        # unwrap yaw to avoid discontinuities
        labels_yaw = labels_yaw % (2*np.pi)


        # note that the lables s are not required because they are always fixed
        # so tehy are burned into the solver (look at generate_fixed_path_quantities in the solver handler class)

        return labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds,\
        labels_d2xds2, labels_d2yds2, labels_d2zds2, labels_k, labels_wz, labels_wx,\
        labels_roll, labels_pitch, labels_yaw

    def rototranslate_abs_2_path_frame(self, x, y, z, vx, vy, vz, path_xyz_rpy):
        """
        Transforms 3D points and orientation vectors into the local frame defined by path_xyz_rpy.

        Args:
            x, y, z: arrays of global positions
            vx, vy, vz: arrays of orientation vectors
            path_xyz_rpy: reference pose [x, y, z, roll, pitch, yaw]

        Returns:
            x_t, y_t, z_t: transformed positions
            vx_t, vy_t, vz_t: transformed direction vectors
        """
        # Stack input into arrays
        points = np.vstack((x, y, z)).T
        directions = np.vstack((vx, vy, vz)).T

        # Extract reference pose
        ref_pos = np.array(path_xyz_rpy[:3])
        ref_rpy = path_xyz_rpy[3:]

        # Compute inverse rotation: world to local
        R_world_to_local = R.from_euler('xyz', ref_rpy).inv()

        # Translate and rotate points
        translated_points = points - ref_pos
        transformed_points = R_world_to_local.apply(translated_points)

        # Rotate direction vectors
        transformed_directions = R_world_to_local.apply(directions)

        # Split into individual components
        x_t, y_t, z_t = translated_points.T
        vx_t, vy_t, vz_t = transformed_directions.T

        return x_t, y_t, z_t, vx, vy, vz
    
    def interpolate_array_given_s(self,s ,s_vec, X):
        x_out = []
        for i in range(X.shape[1]):
            x_out.append(np.interp(s, s_vec, X[:, i]))
        return np.array(x_out)



class MPCC_controller_class(path_handeling_utilities_class):
    def __init__(self):
        # indicate where to compiled solvers are stored
        path_to_this_folder = os.path.dirname(os.path.abspath(__file__))
        self.solvers_folder_path = os.path.join(path_to_this_folder,'solvers_setup/solvers') 


        ## SELECT THE TRACK ##
        track_choice = 'analytic_circle'
        #track_choice = 'spline_circle'
        #track_choice = 'vicon_racetrack'


        # TEMPORARY
        self.origin_frame = 'map'  # "map" this is the frame where the path is defined, it is used to transform the path into the drone frame

        # Loop of the path
        #self.loop_path = True

        # produce track related fixed quantities
        #self.generate_track(track_choice)
        load_optimally_smoothed_path = False

        self.s_vals_global_path, self.x_vals_global_path, self.y_vals_global_path, self.z_vals_global_path, \
        self.roll_global_path, self.pitch_global_path, self.yaw_global_path, \
        self.s_4_local_path, self.x_4_local_path, self.y_4_local_path, self.z_4_local_path, \
        self.dx_ds, self.dy_ds, self.dz_ds, self.d2x_ds2, self.d2y_ds2, self.d2z_ds2, self.k_vec, \
        self.roll_4_local_path, self.pitch_4_local_path, self.yaw_4_local_path, \
        self.wz_4_local_path, self.wx_4_local_path,\
        self.gates_coordinates, self.gates_s , self.time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, load_optimally_smoothed_path)

        self.synchronous_sim = False



        # Define controller and software options
        self.MPC_algorithm_options = ['MPCC', 'CAMPCC', 'MPCCPP','CAMPCC_EA','CAMPCC_EA_2']
        self.software_choice_options = ['acados', 'forcespro']

        # Default selections
        self.controller_type = "CAMPCC"
        self.software_choice = "forcespro"

        # Create a dictionary to hold all solver handlers
        self.solver_objects_dict = {}

        # Populate the dictionary using combinations
        for controller in self.MPC_algorithm_options:
            self.solver_objects_dict[controller] = {}
            for software in self.software_choice_options:
                handler, solver, dt = self.produce_solver_handlers(controller, software)
                self.solver_objects_dict[controller][software] = {
                    "handler": handler,
                    "solver": solver,
                    "dt": dt
                }

        selected_solver_object = self.solver_objects_dict[self.controller_type][self.software_choice]
        self.solver_handler_obj = selected_solver_object["handler"]
        self.solver = selected_solver_object["solver"]
        self.solver_dt = selected_solver_object["dt"]


        # temporary, then remove this since there will be only 1 drone
        vehicle_number = 1
        self.vehicle_number = vehicle_number




        # set up default values for variables (they will be overwritten in the GUI anyway)
        #self.vehicle_number = vehicle_number
        #self.dt = dt
        self.wide_bound = 1.5
        self.tight_bound = 0.8
        self.gates_width = [0.8]
        self.alpha = 5                      # higher priority term to stick on the path in proximity of the gates
        self.max_thrust = 3.6               # [N], max thrust of the drone, 20% of the hover thrust (inequality constraint)
        self.max_delta_thrust = 1.5         # NOT USED, [N], max variation of thrust of the drone (inequality constraint)

        # set up N (prediction horizon)
        #self.N = 10                         # must match the number of stages in the solver


        ### INITIALIZE VARIABLES ###

        self.q_sdot   = 1     # q_sdot:   tracking velocity weight
        self.q_cont   = 1.0     # q_cont:   contouring error weight
        self.q_lag    = 1     # self.q_lag:    lag error weight
        self.q_thrust = 1     # self.q_thrust: control input cost weight
        self.q_roll_pitch   = 1.0     # self.q_roll_pitch:   roll_rate weight
        self.q_yaw    = 1.0     # self.q_yaw:    yaw_rate weight
        self.qt_pos   = 1      # qt_pos:   terminal position weight
        self.qt_v    = 1      # qt_v:     terminal velocity weight
        self.qt_s = 1
        self.lane_radius = 0.5
        self.local_path_length = 5  # local_path_length: length of the local path used for the MPC controller

        # set up the initial s prediction
        # # chose solver handler objects
        # if self.controller_type == "MPCC":
        #     solver_handler_obj = self.solver_handler_obj_MPCC
        #     solver = self.solver_MPCC
        # elif self.controller_type == "CAMPCC":
        #     solver_handler_obj = self.solver_handler_obj_CAMPCC
        #     solver = self.solver_CAMPCC
        # self.s_pred = np.zeros(solver_handler_obj.N)


        # Initialize state
        #                    [pos_x,  pos_y, pos_z, vx, vy, vz, roll, pitch, yaw,  s ]
        self.initial_state = [ -4.0 , -3.0 , 2.0,   0.0, 0.0 , 0.0 , 0.0 ,  0.0 , 0.0, 0.0]


        # place the initial state into the state variable
        self.state = self.initial_state
        # self.previous_yaw = self.state[8]
        # self.yaw_correction = 0

        #
        self.state_traj = []
        self.input_traj = []

        # initialize path relative variable necessary for local search of closest point on path
        self.previous_index = 1

        # define rviz related topics
        self.set_up_topics_for_rviz()

        # varibles to print lap time
        self.lap_start_time = rospy.Time.now() 
        self.previous_s = 0

        # 
        self.safety_value = 0


        # produce the lane boundaries for the global path
        self.global_lane_bounds_vec = self.produce_lane_boundaries(self.s_vals_global_path)

        # set drone control inputs publisher
        self.des_roll_publisher = rospy.Publisher('des_roll', Float32, queue_size=1)
        self.des_pitch_publisher = rospy.Publisher('des_pitch', Float32, queue_size=1)
        self.des_yaw_publisher = rospy.Publisher('des_yaw', Float32, queue_size=1)
        self.thrust_publisher = rospy.Publisher('thrust', Float32, queue_size=1)
        self.slack_publisher = rospy.Publisher('slack', Float32, queue_size=1)
        self.s_publisher = rospy.Publisher('s', Float32, queue_size=1)
        self.distance_from_centerline_publisher = rospy.Publisher('distance_from_centerline', Float32, queue_size=1)

        # set s_dot publisher
        self.s_dot_publisher = rospy.Publisher('s_dot', Float32, queue_size=1)

        # set computation time publisher (time needed for a complete iteration of the MPC control loop)
        self.comptime_publisher = rospy.Publisher('comptime', Float32, queue_size=1)

        # set subscriber for utility variables
        #self.point_optitrack_state_subscriber = rospy.Subscriber('Point_optitrack_data_topic_', pointmass_opti_pose_stamped_msg, self.point_state_subscriber_callback)
        self.drone_state_subscriber = rospy.Subscriber('Drone_state', Float32MultiArray, self.drone_state_subscriber_callback)
        self.safety_value_subscriber = rospy.Subscriber('safety_value', Float32, self.safety_value_subscriber_callback)

        # visualization publishers if also forward simulating by takin gthe next mpc state
        self.rviz_state_publisher = rospy.Publisher('rviz_data', PoseStamped, queue_size=10)
        self.rviz_drone_visualization_publisher = rospy.Publisher('rviz_drone_visual' , MarkerArray, queue_size=10)

        self.rviz_final_state_terminal_contraint_publisher = rospy.Publisher('rviz_final_state_terminal_contraint', Marker, queue_size=10)

        # in case of using drone simulator, set up the publisher for the trajectory
        self.drone_controller_trajectory_publisher = rospy.Publisher('trajectory_for_drone_controller', Float32MultiArray, queue_size=10)

 


    def send_static_rviz_markers(self):
        # this contains all the rviz messages that don't need to be update, like the global path and the gates location.

        rgba = [160, 189, 212, 0.35]
        marker_type = 4
        self.global_path_message = self.produce_marker_array_rviz(self.x_vals_global_path, self.y_vals_global_path, self.z_vals_global_path, rgba, marker_type)
        self.rviz_global_path_publisher.publish(self.global_path_message)

        # produce and send out global lane boundaries message to rviz
        rgba = [100.0, 255.0, 100.0, 0.15]
        # steps (i.e. how often the lane boundaries are plotted)
        bound_step = 3
        bound_width = self.global_lane_bounds_vec
        bound_thickness = 0.0001
        #rviz_global_lane_bound_message = self.produce_and_publish_rviz_gates_global_lane_boundaries(self.s_vals_global_path, rgba, bound_width, bound_step, bound_thickness)
        #self.rviz_global_lane_bound_publisher.publish(rviz_global_lane_bound_message)

        # produce and send out gates message to rviz
        rgba = [255.0, 0.0, 0.0, 0.35]
        gates_width = self.gates_width
        gates_thickness = 0.05
        #rviz_gates_message = self.produce_and_publish_rviz_gates_global_lane_boundaries(self.gates_s, rgba, gates_width, 1,  gates_thickness)
        #self.rviz_gates_publisher.publish(rviz_gates_message)

        # produce start of the track message (useful to understand when starting collecting data)
        rgba = [255, 0, 255, 1]
        marker_type = 2
        self.start_of_track_message = self.produce_marker_rviz(self.x_vals_global_path[0], self.y_vals_global_path[0], self.z_vals_global_path[0], rgba, marker_type)
        self.rviz_start_track_publisher.publish(self.start_of_track_message)




    ### in the following section are present all the callback functions for the subscribed topics mentioned in __init__ of MPC_Controller_class ###

    # callback for the drone optitrack state subscriber
    def drone_state_subscriber_callback(self, msg):
        #rospy.loginfo('Received drone state from subscriber')
        self.state = msg.data

    # callback for the safety value subscriber
    def safety_value_subscriber_callback(self, msg):
        self.safety_value = msg.data


    def produce_solver_handlers(self,controller_type, software_choice):

        # add additional solver options here when they are developed (for now only CAMPCC is available)
        solver_handler_obj = MPC_solver_handler(controller_type,software_choice)

        if software_choice == 'acados':
            solver_path = os.path.join( self.solvers_folder_path,
                                        solver_handler_obj.solver_name,
                                        solver_handler_obj.solver_name + '.json')
            # check if the file exists
            if os.path.isfile(solver_path) == False:
                print('')
                print('Warning! The solver location is invalid:')
                print(solver_path)
                print('')
                dt_controller_rate = 0.1  # default value if the solver is not found
                solver = None

            else:
                ocp = solver_handler_obj.produce_ocp()
                solver = AcadosOcpSolver(ocp, json_file=solver_path, build=False, generate=False)

                print('________________________________________________________________________________________')
                print('Successfully loaded  solver: ' + solver_handler_obj.solver_name)
                print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
                dt_controller_rate = solver_handler_obj.time_horizon/solver_handler_obj.N

        elif software_choice == 'forcespro':
            solver_path = os.path.join(self.solvers_folder_path, solver_handler_obj.solver_name_forcespro)

            # check if the folder exists
            if os.path.isdir(solver_path) == False:
                print('')
                print('Warning! The solver location is invalid:')
                print(solver_path)
                print('')
                dt_controller_rate = 0.1  # default value if the solver is not found
                solver = None

            else:
                # load the solver
                import forcespro 
                solver = forcespro.nlp.Solver.from_directory(solver_path)

                print('________________________________________________________________________________________')
                print('Successfully loaded  solver: ' + solver_handler_obj.solver_name_forcespro)
                print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
                dt_controller_rate = solver_handler_obj.time_horizon/solver_handler_obj.N

        # adjust control rate
        
        #print('control rate (dt):', np.round(dt_controller_rate,3))
        #print('________________________________________________________________________________________')

        

        return solver_handler_obj, solver, dt_controller_rate



    def set_up_topics_for_rviz(self):
        self.rviz_MPC_path_publisher = rospy.Publisher('rviz_MPC_path_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_global_path_publisher = rospy.Publisher('rviz_global_path_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_start_track_publisher = rospy.Publisher('rviz_start_track_' + str(self.vehicle_number), Marker, queue_size=10)
        self.rviz_local_path_publisher = rospy.Publisher('rviz_local_path_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_local_path_CAMPCC_EA_publisher = rospy.Publisher('rviz_local_path_CAMPCC_EA_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_X0_path_publisher = rospy.Publisher('rviz_X0_path_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_global_lane_bound_publisher = rospy.Publisher('rviz_global_lane_bound_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_local_lane_bound_publisher = rospy.Publisher('rviz_local_lane_bound_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_curvature_centre_publisher = rospy.Publisher('rviz_curvature_centre_' + str(self.vehicle_number), MarkerArray, queue_size=10)
        self.rviz_gates_publisher = rospy.Publisher('rviz_gates_' + str(self.vehicle_number), MarkerArray, queue_size=10)

        self.reference_path_pose_publisher = rospy.Publisher('reference_pose', PoseStamped, queue_size=1)
        self.drone_relative_2path_pose_publisher = rospy.Publisher('drone_relative_2path_pose', PoseStamped, queue_size=1)




        # MPC path is the path followed by the vehicle in the open loop prediction (prediction horizon)
        # global path is the overall path structure (i.e. the track)
        # local path is the path evaluated by the chebychev polynomials of extremes [a,b], is the portion of path plotted by rviz
        # global lane bound are the lane boundaries plotted for all the path
        # local lane bound are the lane boundaries plotted for the local path and seen by the forces pro solver (i.e. the ones used for the constraints) 
        # curvature centre is the centre of the circle that best fits the local path
        # gates are the gates of the track



    # according to s parameter produce the value for the lane boundary, condisering if we're far/close to the gates
    # gate position is determined according to the s values related to the global path
    # so the evaluation of the local lane boundaries is valid assuming that the xyz coordinates of the smoothed local path are close to the global ones
    def produce_lane_boundaries(self, s_vals_vec):

        # initialize the lane_bounds_vec, all the points far from the gates have the same bound width (wide one)
        lane_bounds_vec = np.full(len(s_vals_vec), self.wide_bound)

        # define how far from the gate the tight bound should start
        approach_point = 0.5  # [m]

        # produce the vector of indexes of the points close to the gates
        for ii in range(len(self.gates_s)):
            # s value of the gate of interest
            s_gate = self.gates_s[ii]
            
            # define the s_start and s_end values
            s_start = s_gate - approach_point
            s_end = s_gate + approach_point

            # check if the s_start and s_end are within the bounds of the global path, if not wrap them around

            if s_start < self.s_vals_global_path[0]:
                # if s_start is outside the bounds of the global path, we should do the search also on the last part of the track
                s_start = self.s_vals_global_path[-1] - approach_point

                # find the index of the s values between (s_gate - approach_point) and (s_gate + approach_point)
                index1 = np.where((s_vals_vec >= s_start) & (s_vals_vec <= self.s_vals_global_path[-1]))
                index1 = index1[0]

                index2 = np.where((s_vals_vec >= self.s_vals_global_path[0]) & (s_vals_vec <= s_end))
                index2 = index2[0]

                index = np.concatenate((index1, index2)) 

            elif s_end > self.s_vals_global_path[-1]:
                # if s_end is outside the bounds of the global path, we should do the search also on the first part of the track
                s_end = self.s_vals_global_path[0] + approach_point

                # find the index of the s values between (s_gate - approach_point) and (s_gate + approach_point)
                index1 = np.where((s_vals_vec >= s_start) & (s_vals_vec <= self.s_vals_global_path[-1]))
                index1 = index1[0]

                index2 = np.where((s_vals_vec >= self.s_vals_global_path[0]) & (s_vals_vec <= s_end))
                index2 = index2[0]
                
                index = np.concatenate((index1, index2))

            else:

                # find the index of the s values between (s_gate - approach_point) and (s_gate + approach_point)
                index = np.where((s_vals_vec >= s_start) & (s_vals_vec <= s_end))
                index = index[0]

            # now index contains the indexes of all the points inside the s_vals_vec vector that are close to the gate of interest (gate[ii])

            # for every point close to the gate of interest set the tight bound width
            lane_bounds_vec[index] = self.tight_bound

        return lane_bounds_vec



    # produce the marker_array to visualize things in rviz (global path, local path, predictions)
    def produce_marker_array_rviz(self, x, y, z, rgba, marker_type):
        marker_array = MarkerArray()              # definition of an array of markers
        marker = Marker()                         # single marker within the marker_array

        marker.header.frame_id = self.origin_frame            # map frame, used when markers or data should be positioned in relation to a global map.
        marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line_strip: 4
        marker.type = marker_type
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.025

        # Set the color
        marker.color.r = rgba[0] / 256
        marker.color.g = rgba[1] / 256
        marker.color.b = rgba[2] / 256
        marker.color.a = rgba[3]

        # Set the pose of the marker
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        points_list = []
        for i in range(len(x)):
            p = Point()
            p.x = x[i]
            p.y = y[i]
            p.z = z[i]
            points_list = points_list + [p]

        marker.points = points_list

        # append the created marker to marker_array
        marker_array.markers.append(marker)

        return  marker_array



    # produce a single marker for the rviz visualization
    def produce_marker_rviz(self, x, y, z, rgba, marker_type):
        marker = Marker()                         # single marker within the marker_array

        marker.header.frame_id = self.origin_frame            # map frame, used when markers or data should be positioned in relation to a global map.
        marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line_strip: 4
        marker.type = marker_type
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.scale.z = 0.025

        # Set the color
        marker.color.r = rgba[0] / 256
        marker.color.g = rgba[1] / 256
        marker.color.b = rgba[2] / 256
        marker.color.a = rgba[3]

        # Set the pose of the marker
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return  marker



    # prepare and publish the global lane boundaries and gates and for the rviz visualization
    def produce_and_publish_rviz_gates_global_lane_boundaries(self, s_vals_path, rgba, width, step = 3 ,thickness = 0.0001):

        # defining the points for the boundary check
        length = len(s_vals_path)                                            # n points that make the global path
        x_lane, y_lane, z_lane, vec_quat, step_width = [], [], [], [], []    # initialize the lane vectors


        # fill the lane vectors
        for i in range(0,length,step):
            if (i*step < length):
                actual_s = s_vals_path[i*step]

                # if we're producing the lane boundaries we place the proper width, if we're producing the gates we place the gates width
                if len(width) == 1:
                    # if width is a single value, we're producing the gates
                    step_width.append(width[0])
                else:
                    # if width is a vector, we're producing the lane boundaries
                    step_width.append(width[i*step])
                
                # retrieve the position
                actual_x = self.x_of_s(actual_s)
                actual_y = self.y_of_s(actual_s)
                actual_z = self.z_of_s(actual_s)

                # retrieve and normalize the tangent vector
                tangent = [self.x_tan_of_s(actual_s), self.y_tan_of_s(actual_s), self.z_tan_of_s(actual_s)]
                tangent = tangent / ( np.linalg.norm(tangent) + 1e-15 ) 

                # retrieve and normalize the normal vector
                normal = [self.x_nor_of_s(actual_s), self.y_nor_of_s(actual_s), self.z_nor_of_s(actual_s)]
                normal = normal / ( np.linalg.norm(normal) + 1e-15 )

                # compute the binormal
                binormal = np.cross(tangent, normal)
                binormal = binormal / ( np.linalg.norm(binormal) + 1e-15 )

                # create a rotation matrix from the tangent, normal and binormal
                rotation_matrix = np.column_stack((tangent, normal, binormal))

                # to have the cylinder oriented in the useful direction we should rotate on y axis by 90°
                rotation_matrix = np.dot(rotation_matrix, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

                # convert the rotation matrix to a rotation object for easier manipulation and convert in quaternion
                quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

                x_lane.append(actual_x)
                y_lane.append(actual_y)
                z_lane.append(actual_z)
                vec_quat.append(quaternion)




        ## FOR RVIZ ##

        # marker type (circle)
        marker_type = 3

        # produce marker array for the lane boundaries and publish it
        rviz_lane_bound_message = self.produce_tridimensional_lane_bounds(x_lane, y_lane, z_lane, vec_quat, rgba, marker_type, step_width, thickness)

        return rviz_lane_bound_message



    # takes the vectors of positions and orientations for the lane boundaries markers and produces the relative marker object
    def produce_tridimensional_lane_bounds(self, x, y, z, quat, rgba, marker_type, width_vec, thickness = 0.0001):
        marker_array = MarkerArray()                  # definition of an array of markers

        for i in range(0,len(x)):
            marker = Marker()                         # single marker within the marker_array
            marker.header.frame_id = self.origin_frame            # map frame, used when markers or data should be positioned in relation to a global map.
            marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line_strip: 4
            marker.type = marker_type
            marker.id = i

            # Set the scale of the marker
            marker.scale.x = width_vec[i]
            marker.scale.y = width_vec[i]
            marker.scale.z = thickness                # we want a circle, so z is very small

            # Set the color
            marker.color.r = rgba[0] / 256
            marker.color.g = rgba[1] / 256
            marker.color.b = rgba[2] / 256
            marker.color.a = rgba[3]

            # Set the pose of the marker
            marker.pose.position.x = x[i]
            marker.pose.position.y = y[i]
            marker.pose.position.z = z[i]
            marker.pose.orientation.x = quat[i][0]
            marker.pose.orientation.y = quat[i][1]
            marker.pose.orientation.z = quat[i][2]
            marker.pose.orientation.w = quat[i][3]

            # append the created marker to marker_array
            marker_array.markers.append(marker)

        return  marker_array
  




    # The MPCC control loop (this is one iteration only so this need to be called multiple times in a while loop)
    def run_one_mpc_control_loop(self, controller_type, software_choice):

        # select solver related objects
        selected_solver_object = self.solver_objects_dict[controller_type][software_choice]
        solver_handler_obj = selected_solver_object["handler"]
        solver = selected_solver_object["solver"]
        solver_dt = selected_solver_object["dt"]

        # also copy to self to keep track of things in testin envrironment
        self.solver_handler_obj = solver_handler_obj
        self.solver = solver


        # copy the state so it can't chnge during the control loop
        state = copy.deepcopy(self.state)  # copy the state to avoid modifying the original one
        # print('yaw = ', state[8])
        # # accumulate yaw rotations to avoid jumps
        # if state[8] - self.previous_yaw > np.pi:
        #     self.yaw_correction = self.yaw_correction - 2*np.pi
        # elif state[8] - self.previous_yaw < -np.pi:
        #     self.yaw_correction = self.yaw_correction + 2*np.pi
        
        # state[8] = state[8] + self.yaw_correction
        # self.previous_yaw = self.state[8]
        

        # find closest point on the path
        xyz_state = state[:3]
        # evaluate velocity norm
        v_norm = np.linalg.norm(state[3:6])
        estimated_ds = 0.6 # this is the local area to look into finding closest point on path
        s, current_path_index, dist_to_centerline, xyz_closest_point = self.find_s_of_closest_point_on_global_path_3d( xyz_state, 
                                                        self.s_vals_global_path, 
                                                        self.x_vals_global_path, 
                                                        self.y_vals_global_path, 
                                                        self.z_vals_global_path,
                                                        self.previous_index, 
                                                        estimated_ds)


        # update index
        self.previous_path_index = current_path_index  # update index along the path to know where to search in next iteration
        
        self.s_publisher.publish(Float32(s)) # publish s for simulation purpouses
        if dist_to_centerline > self.lane_radius:
            print('Warning! The drone is outside the lane boundaries! Distance from centerline:', round(dist_to_centerline,2), 'm')
        self.distance_from_centerline_publisher.publish(Float32(dist_to_centerline)) # publish distance from centerline for simulation purpouses    
        
        if s-self.previous_s < -10: # passed finish line
            lap_time = rospy.Time.now() - self.lap_start_time # get the time since the last lap start
            # print with 2 decimal numbers
            #print('Lap time:', round(lap_time.to_sec(), 2))
            # log lap time
            rospy.loginfo('Lap time: ' + str(round(lap_time.to_sec(), 2)) + ' seconds')
            self.lap_start_time = rospy.Time.now() 
        self.previous_s = s  # update s value to know where to search in next iteration


        labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds, labels_d2xds2, labels_d2yds2, labels_d2zds2, labels_k,\
        labels_wz, labels_wx, labels_roll, labels_pitch, labels_yaw = self.produce_ylabels_4_local_kernelized_path(s,self.local_path_length,solver_handler_obj.n_points_kernelized,xyz_closest_point) # path_xyz_rpy
        
        # using the time optimal trajectory for warmstarting the solver
        start_index = np.argmin(np.abs(self.time_optimal_trajectory_4_warmstart[:,13] - s))
        X0 = solver_handler_obj.produce_X0_from_time_optimal_trajectory(self.time_optimal_trajectory_4_warmstart, start_index, xyz_closest_point,
                                                                        self.s_4_local_path, self.x_4_local_path, self.y_4_local_path, self.z_4_local_path ,
                                                                        self.roll_4_local_path, self.pitch_4_local_path ,self.yaw_4_local_path)
        # add reference path states if using CAMPCC_EA




 
        # extract terminal state to be reached
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = X0[-1,solver_handler_obj.nu:solver_handler_obj.nu+6]
        

        # produce parameters
        params_i = np.array([
                            self.q_sdot,                     # p[0]
                            self.q_cont,                     # p[1]
                            self.q_lag,                      # p[2]
                            self.q_roll_pitch,               # p[3]
                            self.q_yaw,                      # p[4]
                            self.q_thrust,                   # p[5]
                            self.qt_pos,                     # p[6]
                            self.qt_v,                       # p[7]
                            self.qt_s,                       # p[8]
                            self.lane_radius,                # p[9]
                            self.local_path_length,          # p[10]
                            terminal_x,                  # p[11] terminal x position
                            terminal_y,                  # p[12] terminal y position
                            terminal_z,                  # p[13] terminal z position
                            terminal_vx,                 # p[14] terminal vx
                            terminal_vy,                 # p[15] terminal vy
                            terminal_vz,                 # p[16] terminal vz                  
                        ])
        
        if controller_type == "MPCC" or controller_type == "MPCCPP":
            params_i = np.array([
                                *params_i,              
                                *labels_x,                      
                                *labels_y,                      
                                *labels_z,                      
                                *labels_dxds,                   
                                *labels_dyds,                   
                                *labels_dzds                    
                            ])
            
        elif controller_type == "CAMPCC":
            params_i = np.array([
                                *params_i,
                                *labels_x,                      
                                *labels_y,                      
                                *labels_z,                      
                                *labels_dxds,                   
                                *labels_dyds,                   
                                *labels_dzds,
                                *labels_d2xds2,       
                                *labels_d2yds2,
                                *labels_d2zds2, 
                                *labels_k ]) 
            
        elif controller_type == "CAMPCC_EA":
            params_i = np.array([
                                *params_i,
                                *labels_wz,
                                *labels_wx  
                            ])
            
        elif controller_type == "CAMPCC_EA_2":
            params_i = np.array([
                                *params_i,
                                *labels_x,                      
                                *labels_y,                      
                                *labels_z,  
                                *labels_roll,
                                *labels_pitch,
                                *labels_yaw,
                                *labels_wz,
                            ])

                

        #ready the solver for this control loop
        # the state is:  pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s
                # notice that position and rotation are expressed relative to the current path index,
                # while velocities are still in the body frame as before

        # xinit = np.array([  drone_relative_pose[0], drone_relative_pose[1], drone_relative_pose[2],
        #                     state[3], state[4], state[5],
        #                     drone_relative_pose[3], drone_relative_pose[4], drone_relative_pose[5], Ds_back])
                # shift the path to be relative to the closest point on the path (to keep solver conditions similar)
        xinit = np.array([  state[0]-xyz_closest_point[0],
                            state[1]-xyz_closest_point[1],
                            state[2]-xyz_closest_point[2],
                            state[3], state[4], state[5],
                            state[6], state[7], state[8], 0])
        if controller_type == "CAMPCC_EA": # add the reference path states
            roll_ref = self.roll_global_path[current_path_index]
            pitch_ref = self.pitch_global_path[current_path_index]
            yaw_ref = self.yaw_global_path[current_path_index]
            xinit = np.array([  *xinit,
                                0,
                                0,
                                0,
                                roll_ref,
                                pitch_ref,
                                yaw_ref])
        

        #labels_s = np.linspace(0, self.local_path_length, self.solver_handler_obj.n_points_kernelized)
        #X0 = self.solver_handler_obj.produce_X0(labels_s,labels_x,labels_y,labels_z)


        # solve the problem and extract the solution   
        state_traj, input_traj = solver_handler_obj.solve_mpc(software_choice, solver, xinit, params_i, X0) # optionally also supply first guess ,X0
        
        # recover global x y z
        state_traj[:, 0] = state_traj[:, 0] + xyz_closest_point[0] 
        state_traj[:, 1] = state_traj[:, 1] + xyz_closest_point[1]
        state_traj[:, 2] = state_traj[:, 2] + xyz_closest_point[2]

        if controller_type == "CAMPCC_EA":
            # recover global roll pitch yaw
            state_traj[:, 10] = state_traj[:, 10] + xyz_closest_point[0] 
            state_traj[:, 11] = state_traj[:, 11] + xyz_closest_point[1] 
            state_traj[:, 12] = state_traj[:, 12] + xyz_closest_point[2] 

        # save the state trajectory to self for when running the controller in testing environment
        self.state_traj = state_traj
        self.input_traj = input_traj

        # publish the control inputs
        self.publish_control_inputs(input_traj,solver_handler_obj)
        

        # --- rviz visualization messages ---
        self.produce_and_publish_rviz_visualization(state_traj, labels_x, labels_y ,labels_z, xyz_closest_point, X0,
                                                    terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz,solver_handler_obj.nu,controller_type) # ,drone_relative_pose,path_xyz_rpy

        # publish mpc output array for the drone controller
        self.publish_trajectory_for_drone_controller(input_traj, state_traj, solver_handler_obj)


    def publish_control_inputs(self, input_traj, solver_handler_obj):
        if solver_handler_obj.controller_type == "MPCCPP":
            # unpack the control inputs for the MPCCPP controller
            roll_desired, pitch_desired, yaw_desired, thrust, slack, s_dot = solver_handler_obj.unpack_u(input_traj[0,:])
        else:
            roll_desired, pitch_desired, yaw_desired, thrust, slack = solver_handler_obj.unpack_u(input_traj[0,:])

    

        des_roll_val = Float32(roll_desired)
        des_pitch_val = Float32(pitch_desired)
        des_yaw_val = Float32(yaw_desired)
        thrust_val = Float32(thrust)
        slack_val = Float32(slack)

        # publish the control inputs
        self.des_roll_publisher.publish(des_roll_val)
        self.des_pitch_publisher.publish(des_pitch_val)
        self.des_yaw_publisher.publish(des_yaw_val)
        self.thrust_publisher.publish(thrust_val)
        self.slack_publisher.publish(slack_val)



    def publish_trajectory_for_drone_controller(self,input_traj, state_traj, solver_handler_obj):
        """takes output from the solver and converts it to the trajectory for the drone controller.
        output is in the form:
        x y z qx qy qz qw vx vy vz vx vy vz wx wy wz
        """
        # initialize the trajectory for the drone controller
        mpc_trajectory_for_drone_controller = np.zeros((state_traj.shape[0]-1,13))  # 5 elements: roll, pitch, yaw, thrust, slack

        # unpack the control inputs for the MPCCPP controller
        for i in range(state_traj.shape[0]-1):
            if solver_handler_obj.controller_type == "MPCCPP":
                roll_desired, pitch_desired, yaw_desired, thrust, slack, s_dot = solver_handler_obj.unpack_u(input_traj[i,:])
            else:
                roll_desired, pitch_desired, yaw_desired, thrust, slack = solver_handler_obj.unpack_u(input_traj[i,:])

            # evaluate the quaternions starting from the roll pitch yaw angles
            # unpack state
            pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = solver_handler_obj.unpack_x(state_traj[i,:])
            q = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()

            # determine the wx wy wz from the inputs by evaluating the angular velocity input from the drone model
            wx, wy, wz = evaluate_w(roll_desired,pitch_desired,yaw_desired,roll,pitch,yaw)

            # fill in the row
            mpc_trajectory_for_drone_controller[i,:] = np.array([pos_x, pos_y, pos_z,
                                                                q[0], q[1], q[2], q[3],
                                                                vx, vy, vz,
                                                                wx, wy, wz])
            
        mpc_output_array_msg = Float32MultiArray()
        mpc_output_array_msg.data = mpc_trajectory_for_drone_controller.flatten().tolist()
        # publish the mpc output array
        self.drone_controller_trajectory_publisher.publish(mpc_output_array_msg)
        return mpc_output_array_msg



    def produce_and_publish_rviz_visualization(self, state_traj, labels_x, labels_y, labels_z,xyz_closest_point,X0,
                                               terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz,nu,controller_type): # , drone_relative_pose, path_xyz_rpy
        # PREPARING THE DATA TO BE PUBLISHED IN RVIZ
        labels_x = labels_x + xyz_closest_point[0]
        labels_y = labels_y + xyz_closest_point[1]
        labels_z = labels_z + xyz_closest_point[2]

        # --- open loop prediction ---
        x_pred = state_traj[:, 0] 
        y_pred = state_traj[:, 1] 
        z_pred = state_traj[:, 2] 

        rgba = [0, 166, 214, 1.0]
        marker_type = 4
        # produce marker for MPCC path prediction and publish it
        rviz_MPCC_path_message = self.produce_marker_array_rviz(x_pred, y_pred, z_pred, rgba, marker_type)
        self.rviz_MPC_path_publisher.publish(rviz_MPCC_path_message)

        # --- local path ---
        # set colors for the local path
        
        # produce marker for local path and publish it
        rgba = [250, 150, 100, 0.5]
        rviz_local_path_message = self.produce_marker_array_rviz(labels_x, labels_y, labels_z, rgba, marker_type)
        self.rviz_local_path_publisher.publish(rviz_local_path_message)

        if controller_type == "CAMPCC_EA":
            rgba = [255, 0, 255, 0.5]
            ref_path_x = state_traj[:, 10]
            ref_path_y = state_traj[:, 11]
            ref_path_z = state_traj[:, 12]
            rviz_local_path_CAMPCC_EA_message = self.produce_marker_array_rviz(ref_path_x, ref_path_y, ref_path_z, rgba, marker_type)
            self.rviz_local_path_CAMPCC_EA_publisher.publish(rviz_local_path_CAMPCC_EA_message)
            # print state trajectory with 2 decimal numbers
            # print('State trajectory:')
            # np.set_printoptions(precision=2, suppress=True)
            # print(state_traj[:,:3])
            # print(state_traj[:,10:13])


        # --- X0 path ---
        rgba = [0, 204, 204, 0.5]
        # produce marker for local path and publish it
        X0_x = X0[:, nu] + xyz_closest_point[0]
        X0_y = X0[:, nu+1] + xyz_closest_point[1]
        X0_z = X0[:, nu+2] + xyz_closest_point[2]
        rviz_X0_path_message = self.produce_marker_array_rviz(X0_x, X0_y, X0_z, rgba, marker_type)
        self.rviz_X0_path_publisher.publish(rviz_X0_path_message)
        



        # final state terminal contraint
        self.publish_arrow_marker(self.rviz_final_state_terminal_contraint_publisher,
                            terminal_x + xyz_closest_point[0],
                            terminal_y + xyz_closest_point[1],
                            terminal_z + xyz_closest_point[2],
                            terminal_vx, 
                            terminal_vy, 
                            terminal_vz,
                            color=(1.0, 0.0, 0.0), scale=0.1)







    # prepare and publish the lane boundaries and gates and for the rviz visualization (position and orientation are related to the local smoothed path)
    def produce_and_publish_local_lane_boundaries(self, position_local_path, tangent_local_path, normal_local_path , rgba, width_vec, step = 5 ,thickness = 0.0001):

        # defining the points for the boundary check
        length = len(position_local_path[:,0])                              # n points that make the global path
        x_lane, y_lane, z_lane, vec_quat, step_width = [], [], [], [],[]    # initialize the lane vectors

        # fill the lane vectors
        for i in range(0,length,step):
            if (i*step < length):
                
                step_width.append(width_vec[i*step])

                # retrieve the position
                actual_x = position_local_path[i*step, 0]
                actual_y = position_local_path[i*step, 1]
                actual_z = position_local_path[i*step, 2]

                # retrieve and the tangent
                tangent = tangent_local_path[i*step]

                # retrieve and normalize the normal vector
                normal = normal_local_path[i*step]

                # compute the binormal
                binormal = np.cross(tangent, normal)
                binormal = binormal / (np.linalg.norm(binormal) + 1e-15)

                # create a rotation matrix from the tangent, normal and binormal
                rotation_matrix = np.column_stack((tangent, normal, binormal))

                # to have the cylinder oriented in the useful direction we should rotate on y axis by 90°
                rotation_matrix = np.dot(rotation_matrix, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

                # convert the rotation matrix to a rotation object for easier manipulation and convert in quaternion
                quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

                x_lane.append(actual_x)
                y_lane.append(actual_y)
                z_lane.append(actual_z)
                vec_quat.append(quaternion)

        ## FOR RVIZ ##

        # marker type (circle)
        marker_type = 3

        # produce marker array for the lane boundaries and publish it
        rviz_lane_bound_message = self.produce_tridimensional_lane_bounds(x_lane, y_lane, z_lane, vec_quat, rgba, marker_type, step_width, thickness)

        return rviz_lane_bound_message

    def publish_arrow_marker(self, pub, x, y, z, vx, vy, vz, frame_id="map", ns="arrows", marker_id=0, color=(1.0, 0.0, 0.0), scale=0.1):
        marker = Marker()
        marker.header.frame_id = self.origin_frame #frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Start and end points of the arrow
        start = Point(x, y, z)
        end = Point(x + vx, y + vy, z + vz)

        marker.points.append(start)
        marker.points.append(end)

        # Scale: shaft diameter, head diameter, head length
        marker.scale.x = scale       # shaft diameter
        marker.scale.y = scale * 2   # head diameter
        marker.scale.z = scale * 3   # head length

        # Color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0)  # 0 = forever

        return pub.publish(marker)

    def simulation_rviz_visuals(self,state):
       
        # VISUALIZE THE VEHICLE AS A MOVING FRAME IN RVIZ
        rviz_message = PoseStamped()

        # position
        rviz_message.pose.position.x = state[0]
        rviz_message.pose.position.y = state[1]
        rviz_message.pose.position.z = state[2]

        # orientation
        quat = tf.transformations.quaternion_from_euler(state[6], state[7], state[8])
        rviz_message.pose.orientation.x = quat[0]
        rviz_message.pose.orientation.y = quat[1]
        rviz_message.pose.orientation.z = quat[2]
        rviz_message.pose.orientation.w = quat[3]

        # frame data is necessary for rviz
        rviz_message.header.frame_id = 'map'

        # publish rviz message of the moving frame
        self.rviz_state_publisher.publish(rviz_message)     
       
       
       
        # VISUALIZE THE DRONE IN RVIZ
        marker_array = MarkerArray()              # definition of an array of markers
        marker = Marker()                         # single marker within the marker_array

        marker.header.frame_id = self.origin_frame            # map frame, used when markers or data should be positioned in relation to a global map.
        marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

        marker.type = marker.MESH_RESOURCE
        marker.mesh_resource = "package://curvature_aware_mpcc_pkg/src/Dae_models/quadrotor_base.dae"
        marker.id = 1

        # Set the scale of the marker
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        # Set the color
        rgba = [100.0, 100.0, 100.0, 1.0]   
        marker.color.r = rgba[0] / 256
        marker.color.g = rgba[1] / 256
        marker.color.b = rgba[2] / 256
        marker.color.a = rgba[3]

        # Set the position of the marker
        marker.pose.position.x = state[0]
        marker.pose.position.y = state[1]
        marker.pose.position.z = state[2]

        # Set the oritentation of the marker  
        quat = tf.transformations.quaternion_from_euler(state[6], state[7], state[8])
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker_array.markers.append(marker)

        # publish rviz message of the drone
        self.rviz_drone_visualization_publisher.publish(marker_array)


        


if __name__ == '__main__':
    try:
        # define where to find complied solvers

        rospy.init_node('mpc_node', anonymous=False)
        global_comptime_publisher = rospy.Publisher('GLOBAL_comptime', Float32, queue_size=1)

        # define controller rate
        dt_controller_rate = 0.1

        #set up vehicle controllers
        #car 1
        drone_controller_obj = MPCC_controller_class() 
        # send static rviz markers like the global path and gates location
        import time
        time.sleep(0.25)
        drone_controller_obj.send_static_rviz_markers()


        # start control loop
        rate = rospy.Rate(1 / dt_controller_rate)
        #NOTE that this rate is the rate to send out ALL control imputs to all vehicles

        #set up GUI manager 
        MPC_GUI_manager_obj = MPC_GUI_manager(drone_controller_obj)



        while not rospy.is_shutdown():
            try:
                start_clock_time = rospy.get_rostime()
                # get controller frequency
                #dt_controller = drone_controller_obj.dt_controller_rate
                #rate = rospy.Rate(1 / dt_controller)

                # --- run 1 control loop ---
                drone_controller_obj.run_one_mpc_control_loop(  drone_controller_obj.controller_type,
                                                                drone_controller_obj.software_choice)


                stop_clock_time = rospy.get_rostime()
                elapsed_time_global_loop = (stop_clock_time - start_clock_time).to_sec()
                global_comptime_publisher.publish(elapsed_time_global_loop)

                # check if the total time was exceeded
                if elapsed_time_global_loop > dt_controller_rate:
                    # display dt vs solve time
                    print('Time for control loop exceeded:')
                    print('Control rate (dt):', dt_controller_rate)
                    print('Control loop time:', elapsed_time_global_loop)
                    


            except Exception as e:
                print('Error in control loop:')
                traceback.print_exc()

            rate.sleep()

 




    except rospy.ROSInterruptException:
        pass