#!/usr/bin/env python3
import numpy as np
import traceback
import rospy
import os
from functions_for_MPCC_node_running import find_s_of_closest_point_on_global_path
from MPC_generate_solvers.path_track_definitions import generate_path_data
import time
from std_msgs.msg import String
import tf_conversions

from std_msgs.msg import Float32, Float32MultiArray, Bool,Float64MultiArray
from geometry_msgs.msg import Point, PoseWithCovarianceStamped,PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from curvature_aware_mpcc_pkg.msg import ThreeTimeStampsFloat32 
from datetime import datetime
import csv
# for dynamic paramters reconfigure (setting param values from rqt_reconfigure GUI)
from dynamic_reconfigure.server import Server
from curvature_aware_mpcc_pkg.cfg import GUI_mpc_dynamic_reconfigureConfig
import rospkg
# THIS NEEDS TO BE FIXED (i.e. only use forces and acados stuff if necessary)
#import forcespro.nlp
from acados_template import AcadosOcpSolver
from scipy.linalg import solve_triangular
from tf.transformations import euler_from_quaternion
from MPC_generate_solvers.functions_for_solver_generation import    generate_high_level_path_planner_ocp,\
                                                                    generate_low_level_solver_ocp,\
                                                                    generate_high_level_MPCC_PP,\
                                                                    generate_single_layer_CAMPCC,\
                                                                    generate_single_layer_MPCCPP


# TODO
# make sure N are the same between high and low level solvers









#fory dynamic parameter change using rqt_reconfigure GUI
class MPC_GUI_manager:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        self.solver_software_options = ['ACADOS' , 'FORCES']
        self.MPC_algorithm_options = ['MPCC', 'CAMPCC','MPCC_PP']
        self.dynamic_model_options = ['kinematic_bicycle', 'dynamic_bicycle', 'dynamic_bicycle_GP']

        # as a last thing creat the server because it will be locked executing here
        srv = Server(GUI_mpc_dynamic_reconfigureConfig, self.reconfig_callback)

        


    def reconfig_callback(self, config, level):
        print('_________________________________________________')
        print('  reconfiguring parameters from dynamic_reconfig ')
        print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
        
        for i in range(len(self.vehicles_list)):
            # to check if some values have changed store the old values
            lane_width_old = self.vehicles_list[i].lane_width


            self.vehicles_list[i].synchronous_simulation = config['synchronous_simulation']
            # high level solver
            self.vehicles_list[i].V_target = config['V_target']
            self.vehicles_list[i].q_con = config['q_con']
            self.vehicles_list[i].q_lag = config['q_lag']
            self.vehicles_list[i].q_u_yaw_rate = config['q_u_yaw_rate']
            self.vehicles_list[i].q_sdot = config['q_sdot']
            self.vehicles_list[i].qt_pos_high = config['qt_pos_high']
            self.vehicles_list[i].qt_rot_high = config['qt_rot_high'] 
            self.vehicles_list[i].qt_s_high = config['qt_s_high']
            # low level solver
            self.vehicles_list[i].q_v = config['q_v']
            self.vehicles_list[i].q_pos = config['q_pos']
            self.vehicles_list[i].q_rot = config['q_rot']
            self.vehicles_list[i].qt_pos = config['qt_pos']
            self.vehicles_list[i].qt_rot = config['qt_rot']
            self.vehicles_list[i].q_acc = config['q_acc']
            self.vehicles_list[i].lane_width = config['lane_width']
            self.vehicles_list[i].minimal_plotting = config['minimal_plotting']
            self.vehicles_list[i].delay_compensation = config['delay_compensation']
            self.vehicles_list[i].single_layer = config['single_layer']
            self.vehicles_list[i].solver_software = config['Solver_software']
            # solver choices

            # this is because of how the slecetion works
            self.vehicles_list[i].solver_software = self.solver_software_options[config['Solver_software']]
            self.vehicles_list[i].MPC_algorithm = self.MPC_algorithm_options[config['MPC_algorithm']]
            self.vehicles_list[i].dynamic_model = self.dynamic_model_options[config['Dynamic_model']]
            self.vehicles_list[i].actuator_dynamics = config['actuator_dynamics'] # signal to the solvers that they need to be reinitialized cause the position may have changed since last time they were called
            
            # set up solver type
            self.vehicles_list[i].set_solver_type(self.vehicles_list[i].solver_software,
                                                  self.vehicles_list[i].MPC_algorithm,
                                                  self.vehicles_list[i].dynamic_model,
                                                  self.vehicles_list[i].single_layer,
                                                  self.vehicles_list[i].actuator_dynamics) 
            
            # check if lane width has changed
            if lane_width_old != self.vehicles_list[i].lane_width:
                self.vehicles_list[i].produce_global_lane_boundaries_4_rviz()

            # signal to the solvers that they need to be reinitialized cause the position may have changed since last time they were called
            self.vehicles_list[i].reinitialize = True
            



            print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

        return config




class path_handeling_utilities_class():
    def __init__(self):
        pass


    def produce_ylabels_4_local_kernelized_path(self,s,Ds_back,Ds_forward,xyyaw_ref_path,n):
        #extract indexes of local path

        mask = (self.s_4_local_path >= s - Ds_back) & (self.s_4_local_path <= s + Ds_forward)
        local_path_length = Ds_back + Ds_forward
        # Extract the indexes where the condition is true
        indexes = np.where(mask)[0]
        s_data_points =  self.s_4_local_path[indexes]  # This will have the local path parametrized starting from 0
        x_data_points = self.x_4_local_path[indexes]
        y_data_points = self.y_4_local_path[indexes]
        heading_data_points = self.heading_4_local_path[indexes]
        k_data_points = self.k_4_local_path[indexes]

        # now rototranslate the x and y points to have the first point at the origin
        x_data_points, y_data_points = self.rototranslate_abs_2_path_frame(x_data_points, y_data_points, xyyaw_ref_path)
        heading_data_points = heading_data_points - heading_data_points[0]

        # resample the data points to have a fixed number of points
        # n = self.high_level_solver_generator_obj.n_points_kernelized 
        labels_s = np.linspace(0, 1, n)
        s_interp = (s_data_points - s_data_points[0])/local_path_length
        labels_x = np.interp(labels_s, s_interp, x_data_points)
        labels_y = np.interp(labels_s, s_interp, y_data_points)
        labels_heading = np.interp(labels_s, s_interp, heading_data_points)
        labels_k = np.interp(labels_s, s_interp, k_data_points)

        return labels_x,labels_y,labels_heading,labels_k,local_path_length, labels_s

    def rototranslate_abs_2_path_frame(self, x, y, xyyaw_ref_path):

        # translate
        x = x - x[0]
        y = y - y[0]

        # Create rotation matrix
        rotation_angle = -xyyaw_ref_path[2]
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta,  cos_theta]])

        # Apply rotation
        rotated_points = R @ np.vstack((x, y))  # Matrix multiplication

        transformed_x = rotated_points[0, :]
        transformed_y = rotated_points[1, :]

        return transformed_x, transformed_y

    def relative_xyyaw_to_current_path(self,x_y_yaw_state,s):
        # evaluate current reference path and derivatives needed for initial conditions
        # find corresponding index for s on s_4_local path
        current_path_index_on_4_local_path = np.argmin(np.abs(self.s_4_local_path - s))
        x_ref_path = self.x_4_local_path[current_path_index_on_4_local_path]
        y_ref_path = self.y_4_local_path[current_path_index_on_4_local_path]
        dx_ds_ref_path = self.dx_ds[current_path_index_on_4_local_path]
        dy_ds_ref_path = self.dy_ds[current_path_index_on_4_local_path]
        #evaluate heading angle
        heading_angle_path = np.arctan2(dy_ds_ref_path, dx_ds_ref_path)
        #heading_angle_path = self.heading_4_local_path[current_path_index_on_4_local_path]

        #self.local_path_ref_x,self.local_path_ref_y, self.local_path_rot_angle
        # apply shift to the x y position of the car
        pos_x_0 =  x_y_yaw_state[0] - x_ref_path
        pos_y_0 =  x_y_yaw_state[1] - y_ref_path
        #now rotate to have the first point aligned with the x axis
        pos_x_init_rot =  pos_x_0 * np.cos(heading_angle_path) + pos_y_0 * np.sin(heading_angle_path)
        pos_y_init_rot = -pos_x_0 * np.sin(heading_angle_path) + pos_y_0 * np.cos(heading_angle_path)

        # apply rotation to yaw
        yaw_init_rot = x_y_yaw_state[2] - heading_angle_path

        # this is needed to keep the yaw angle from going over 2 pi
        if yaw_init_rot > np.pi:
            yaw_init_rot -= 2 * np.pi

        elif yaw_init_rot < -np.pi:
            yaw_init_rot += 2 * np.pi

        #
        xyyaw_ref_path = [x_ref_path, y_ref_path, heading_angle_path]


        return pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path

    def rototranslate_path_2_abs_frame(self, x, y, xyyaw_ref_path):
        # xyyaw_ref_path is the xyyaw of the point on the reference path closest to the car

        # Create rotation matrix
        rotation_angle = xyyaw_ref_path[2]
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta,  cos_theta]])

        # Apply rotation
        rotated_points = R @ np.vstack((x, y))  # Matrix multiplication

        # Apply translation
        tx = xyyaw_ref_path[0]
        ty = xyyaw_ref_path[1]
        transformed_x = rotated_points[0, :] + tx
        transformed_y = rotated_points[1, :] + ty

        return transformed_x, transformed_y




class MPCC_controller_class(path_handeling_utilities_class):
    def __init__(self, car_number,dt_controller_rate):

        # decide where to load the actuaror dynamics from
        import importlib.resources
        with importlib.resources.path('DART_dynamic_models', 'actuator_dynamics_saved_parameters') as act_dyn_data_path:
            self.actuator_dynamics_params_folder = str(act_dyn_data_path)
            print('actuator dynamics folder:', self.actuator_dynamics_params_folder)
        with importlib.resources.path('DART_dynamic_models', 'SVGP_saved_parameters_high_speed') as data_path:
            self.GP_params_folder = str(data_path)


        # set up default solver choices that will be overwritten by the dynamic reconfigure anyway so ok
        self.solver_software = 'ACADOS' # 'FORCES', 'ACADOS'
        self.MPC_algorithm = 'MPCC' # 'CAMPCC', 'MPCC_PP'    # solver algorithm can be standard MPCC or curvature-aware CAMPCC
        self.dynamic_model = 'kinematic_bicycle' # 'dynamic_bicycle', 'kinematic_bicycle'

        # path to where solvers are stored
        # path to where this file is stored
        path_to_this_folder = os.path.dirname(os.path.abspath(__file__))
        self.solvers_folder_path = os.path.join(path_to_this_folder,'MPC_generate_solvers/solvers')   

        #set up variables
        self.car_number = car_number
        self.dt_controller_rate = dt_controller_rate

        self.synchronous_simulation = False
        self.pub_rviz_vehicle_visualization = rospy.Publisher('rviz_data_' + str(car_number), PoseStamped, queue_size=10)

        # initialize state variables
        self.vx = 0
        self.vy = 0
        self.omega = 0
        self.x_y_yaw_state = [-1.6, -2.2, 0] 
        self.pose_msg_time = rospy.get_rostime() # initialize time of pose message


        self.th_past_actions = np.zeros(40) # this can be a large number so that the mpc node will have enough (this is set in the mpc solver build)
        self.st_past_actions = np.zeros(40) 


        # delay compensation if in the lab
        self.delay_compensation = True
        self.delay = 0.03 * 0.5 # communication delay in seconds (in the lab) 0.04  (IT WILL be overwritten if the delay estimation node is running)
        self.actuator_dynamics_compensation = False
        if self.actuator_dynamics_compensation:
            # get current folder
            # current_script_path = os.path.realpath(__file__)
            # act_dyn_path = os.path.join(current_script_path, 'MPC_generate_solvers','actuator_dynamics_saved_parameters.csv')
            # self.load_actuator_dynamics(act_dyn_path)
            # # set up B matrices to store past actions
            # self.B_th = np.zeros((self.n_past_th, self.n_past_th))
            # self.B_st = np.zeros((self.n_past_st, self.n_past_st))
            self.delay_act_steps = 2
            self.th_queue = np.zeros(self.delay_act_steps)
            self.st_queue = np.zeros(self.delay_act_steps)
        
        
        # set p contingency if solver does not converge
        self.last_converged_high = True
        self.last_converged_low = True
        self.last_converged_single_layer = True
        self.reinitialize = True # set to true in the beginning so that solvers will be started with the initial guess (now only for single layer)

        self.safety_value = 0

        # define selected solver
        self.single_layer = False
        self.actuator_dynamics = False
        self.set_solver_type(self.solver_software, self.MPC_algorithm, self.dynamic_model,self.single_layer,self.actuator_dynamics)

        #set up constant problem parameters 
        self.initialize_constant_parameters() # only run this once to initialize, then config will overwrite them

        # initialize path relative variables
        self.previous_path_index = 1

        #define rviz related topics
        self.set_up_topics_for_rviz()

        #produce an object to access the lane boundary definition functions (only really needed for visualization)
        #self.Functions_for_solver_generation_obj = Functions_for_solver_generation()

        # set up utility parameters
        self.minimal_plotting = False
        
        self.solver_converged = True

        #for data time stamp initialize sensor data
        self.start_elapsed_time = rospy.get_rostime()
        self.safety_value = 0
        self.current = 0
        self.voltage = 0
        self.IMU_acceleration = [0, 0, 0]
        self.encoder_velocity = 0
        #self.data_folder_name = 'Data'
        #self.file = 0
        #self. writer = 0
        #self.start_elapsed_time = 0
        #self.setup_data_recording()

        # Track related
        #track_choice = 'racetrack_vicon'
        track_choice = 'racetrack_vicon_2' # simpler racetrack
        n_checkpoints = 100

        self.s_vals_global_path,\
        self.x_vals_global_path,\
        self.y_vals_global_path,\
        self.s_4_local_path,\
        self.x_4_local_path,\
        self.y_4_local_path,\
        self.dx_ds, self.dy_ds, self.d2x_ds2, self.d2y_ds2,\
        self.k_vals_global_path,\
        self.k_4_local_path,\
        self.heading_4_local_path = generate_path_data(track_choice, n_checkpoints)


        # set up publishers for robot velocity estimates
        #set up past position variables
        past_states = 3  # actually this is past states + 1 for current state  --- 2
        self.past_x_vicon = np.zeros(past_states)
        self.past_y_vicon = np.zeros(past_states)
        self.past_yaw_vicon = np.zeros(past_states)
        self.past_time_vicon = np.zeros(past_states)
        self.vx_publisher = rospy.Publisher('vx_mpc_' + str(car_number), Float32, queue_size=1)
        self.vy_publisher = rospy.Publisher('vy_mpc_' + str(car_number), Float32, queue_size=1)
        self.w_publisher = rospy.Publisher('w_mpc_' + str(car_number), Float32, queue_size=1)
        self.s_publisher = rospy.Publisher('s_' + str(car_number), Float32, queue_size=1)
        self.distance_from_centerline_publisher = rospy.Publisher('distance_from_centerline_' + str(car_number), Float32, queue_size=1)
        self.solver_converged_publisher = rospy.Publisher('solver_converged_' + str(car_number), Bool, queue_size=1)


        # publish mpc solution as an array
        self.mpc_high_level_solution_publisher = rospy.Publisher('mpc_high_level_solution_' + str(car_number), Float32MultiArray, queue_size=1)
        self.mpc_low_level_solution_publisher = rospy.Publisher('mpc_low_level_solution_' + str(car_number), Float32MultiArray, queue_size=1)


        # set up subscribers (inputs to the controller)
        self.vicon_subscriber = rospy.Subscriber('vicon/jetracer' + str(car_number), PoseWithCovarianceStamped, self.vicon_subscriber_callback)


        # set up publishers (outpus of the controller)
        self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)
        self.steering_publisher = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=1)
        self.comptime_publisher = rospy.Publisher('comptime_' + str(car_number), Float32, queue_size=1)
        self.mpc_throttle_publisher = rospy.Publisher('throttle_3timestamps_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)
        self.mpc_steering_publisher = rospy.Publisher('steering_3timestamps_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)

        # subscribe to comm delay
        self.comm_delay_subscriber = rospy.Subscriber('commdelay_laptop_2_car_' + str(car_number), Float32, self.comm_delay_subscriber_callback)


        # set up publishers for internal mpc node states (selections from GUI)
        self.GUI_param_names_publisher = rospy.Publisher('GUI_param_names_' + str(car_number), String, queue_size=1)
        self.GUI_param_names_list = ['V_target',
                                'q_con',
                                'q_lag',
                                'q_u_yaw_rate',
                                'q_sdot',
                                'qt_pos_high',
                                'qt_rot_high',
                                'qt_s_high',
                                'q_v', 
                                'q_pos', 
                                'q_rot', 
                                'qt_pos', 
                                'qt_rot', 
                                'q_acc',
                                'lane_width',
                                'minimal_plotting', 
                                'delay_compensation', 
                                'Solver_software',
                                'actuator_dynamics',
                                'MPC_algorithm', 
                                'Dynamic_model']
        # msg_GUI_fields = String()
        # msg_GUI_fields.data = ",".join(GUI_param_names_list)
        # self.msg_GUI_fields = msg_GUI_fields #save to publish later




        # additional subscribers and publishers for visualization and data saving
        # send out global path message to rviz
        z_dummy = np.zeros(len(self.x_vals_global_path))

        rgba = [160, 189, 212, 0.25]
        global_path_message = self.produce_marker_array_rviz(self.x_vals_global_path, self.y_vals_global_path, rgba)
        self.rviz_global_path_publisher.publish(global_path_message)
        # send out lane boundaries to rviz
        self.produce_global_lane_boundaries_4_rviz()

        # for data saving
        self.safety_value_subscriber = rospy.Subscriber('safety_value', Float32, self.safety_value_subscriber_callback)





    def run_one_MPCC_control_loop(self,x_y_yaw_state,vx,vy,omega,V_target):

        start_clock_time = rospy.get_rostime()
        # at runtime, local path and dynamic obstacle need to be updated. Dyn obst is updated by the subscriber callback

        # find the closest point on the global path (i.e. measure s)
        estimated_ds = self.vx * self.dt_controller_rate  # esitmated ds from previous time instant (velocity is measured now so not accounting for acceleration, but this is only for the search of the s initial s value, so no need to be accurate)
        s, self.current_path_index, dist_to_centerline = find_s_of_closest_point_on_global_path(np.array([x_y_yaw_state[0], x_y_yaw_state[1]]), self.s_vals_global_path,
                                                                  self.x_vals_global_path, self.y_vals_global_path,
                                                                  self.previous_path_index, estimated_ds)
        self.previous_path_index = self.current_path_index  # update index along the path to know where to search in next iteration
        self.s = s
        self.s_publisher.publish(Float32(self.s)) # publish s for simulation purpouses
        self.distance_from_centerline_publisher.publish(Float32(dist_to_centerline)) # publish distance from centerline for simulation purpouses    

        # produce Chebyshev coefficients that represent local path
        Ds_forward = 1.5 * V_target * self.high_level_solver_generator_obj.time_horizon #  self.dtt * self.high_level_solver_generator_obj.N
        Ds_back = 0.0 # do NOT change

        pos_x_init_rot, pos_y_init_rot, yaw_init_rot,xyyaw_ref_path = self.relative_xyyaw_to_current_path(x_y_yaw_state,s) # current car state relative to current path index
        n = self.high_level_solver_generator_obj.n_points_kernelized 
        labels_x,labels_y,labels_heading,labels_k,local_path_length,labels_s = self.produce_ylabels_4_local_kernelized_path(s,Ds_back,Ds_forward,xyyaw_ref_path,n)


        if self.single_layer == False:
            # ------ HIGH LEVEL SOLVER ------
            problem_high_level = self.set_up_high_level_solver(Ds_back, pos_x_init_rot, pos_y_init_rot, yaw_init_rot,V_target, local_path_length,labels_x,labels_y,labels_heading,labels_k,labels_s)
            
            start_solve_time = time.time()
            # call the high level solver
            if self.solver_software == 'FORCES':
                output_high_level, exitflag_high, info_high = self.high_level_solver.solve(problem_high_level)
            
            elif self.solver_software == 'ACADOS':
                exitflag_high = self.high_level_solver.solve()

            # extract high level solution
            if self.solver_software == 'FORCES':
                output_array_high_level = np.array(list(output_high_level.values()))

            elif self.solver_software == 'ACADOS':
                output_array_high_level = np.zeros((self.high_level_solver_generator_obj.N+1, self.high_level_solver_generator_obj.nu + self.high_level_solver_generator_obj.nx))
                for i in range(self.high_level_solver_generator_obj.N+1):
                    if i == self.high_level_solver.N:
                        u_i_solution = np.zeros(self.high_level_solver_generator_obj.nu)
                    else:
                        u_i_solution = self.high_level_solver.get(i, "u")
                    x_i_solution = self.high_level_solver.get(i, "x")
                    output_array_high_level[i] = np.concatenate((u_i_solution, x_i_solution))


            end_solve_time = time.time()
            solve_time = end_solve_time - start_solve_time
            if solve_time > self.dt_controller_rate:
                print(f'Solver time limit exceeded: {solve_time:.3f} seconds')


            # check if solver converged
            self.last_converged_high = self.check_solver_convergence(exitflag_high,self.last_converged_high,0) # last input is the choice between high and low level solver
            
            # --------------------------------

            # convert output array if MPCC_PP is used
            if self.MPC_algorithm == 'MPCC_PP':
                # reconstruct the path related quantities from the labels
                #              u_yaw_rate slack s_dot   x y yaw s (MPCC_PP) u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s
                # extract 
                yaw_rate_high_level = output_array_high_level[:,0]
                x_high_level = output_array_high_level[:,3]
                y_high_level = output_array_high_level[:,4]
                yaw_high_level = output_array_high_level[:,5]
                # interpolate to get the path quantities
                s_output_vec = output_array_high_level[:,6]
                x_path = np.interp(s_output_vec/local_path_length, labels_s, labels_x)
                y_path = np.interp(s_output_vec/local_path_length, labels_s, labels_y)
                heading_path = np.interp(s_output_vec/local_path_length, labels_s, labels_heading)

            elif self.MPC_algorithm == 'MPCC' or self.MPC_algorithm == 'CAMPCC':
                #              u_yaw_rate slack   x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
                # extact 
                yaw_rate_high_level = output_array_high_level[:,0]
                x_high_level = output_array_high_level[:,2]
                y_high_level = output_array_high_level[:,3]
                yaw_high_level = output_array_high_level[:,4]
                x_path = output_array_high_level[:,6]
                y_path = output_array_high_level[:,7]
                heading_path = output_array_high_level[:,8]


            # ------ LOW LEVEL SOLVER ------
            problem_low_level = self.set_up_low_level_solver_problem(x_high_level,
                                                                    y_high_level,
                                                                    yaw_high_level,
                                                                    yaw_rate_high_level,
                                                                    x_path,
                                                                    y_path,
                                                                    V_target,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega)

            # call the low level solver
            if self.solver_software == 'FORCES':
                output_low_level, exitflag_low, info = self.low_level_solver.solve(problem_low_level)
            elif self.solver_software == 'ACADOS':
                exitflag_low = self.low_level_solver.solve() # solve the problem

            # extract low level solution
            if self.solver_software == 'FORCES':
                output_array_low_level = np.array(list(output_low_level.values()))

            elif self.solver_software == 'ACADOS':
                # Retrieve the state trajectory
                output_array_low_level = np.zeros((self.low_level_solver_generator_obj.N, self.low_level_solver_generator_obj.nu + self.low_level_solver_generator_obj.nx))
                for i in range(self.low_level_solver_generator_obj.N):
                    u_i_solution = self.low_level_solver.get(i, "u")
                    x_i_solution = self.low_level_solver.get(i, "x")
                    output_array_low_level[i] = np.concatenate((u_i_solution, x_i_solution))

            # check if solver converged
            self.last_converged_low = self.check_solver_convergence(exitflag_low,self.last_converged_low, 1) # last input is the choice between high and low level solver
            

            output_array_high_msg = Float32MultiArray()
            output_array_high_msg.data = output_array_high_level.flatten().tolist()  # Convert NumPy array to list
            self.mpc_high_level_solution_publisher.publish(output_array_high_msg)
            
            self.mpc_low_level_solution_publisher.publish(Float32MultiArray(data=output_array_low_level.flatten().tolist()))

            # extract solution for plotting
            x_low_level = output_array_low_level[:,3]
            y_low_level = output_array_low_level[:,4]

            # publish control inputs
            self.publish_control_inputs(output_array_low_level)
            # --------------------------------




        else: # running single layer solver

            # ------ SINGLE LAYER SOLVER ------
            problem_single_layer = self.set_up_single_layer_solver_problem(pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega,
                                           V_target, self.q_v,local_path_length,labels_k,labels_s,Ds_back,labels_x,labels_y,labels_heading)
            
            start_solve_time = time.time()
            # call the high level solver
            if self.solver_software == 'FORCES':
                output_single_layer, exitflag_single_layer, info_single_layer = self.single_layer_solver.solve(problem_single_layer)
            
            elif self.solver_software == 'ACADOS':
                exitflag_single_layer = self.single_layer_solver.solve()

            # extract high level solution
            if self.solver_software == 'FORCES':
                output_array_single_layer = np.array(list(output_single_layer.values()))

            elif self.solver_software == 'ACADOS':
                output_array_single_layer = np.zeros((self.single_layer_solver_generator_obj.N+1, self.single_layer_solver_generator_obj.nu + self.single_layer_solver_generator_obj.nx))
                for i in range(self.single_layer_solver_generator_obj.N+1):
                    if i == self.single_layer_solver.N:
                        u_i_solution = np.zeros(self.single_layer_solver_generator_obj.nu)
                    else:
                        u_i_solution = self.single_layer_solver.get(i, "u")
                    x_i_solution = self.single_layer_solver.get(i, "x")
                    output_array_single_layer[i] = np.concatenate((u_i_solution, x_i_solution))

            # check if solver converged
            self.last_converged_single_layer = self.check_solver_convergence(exitflag_single_layer,self.last_converged_single_layer, 2) # last input is the choice between high and low level solver

            end_solve_time = time.time()
            solve_time = end_solve_time - start_solve_time
            if solve_time > self.dt_controller_rate:
                print(f'Solver time limit exceeded: {solve_time:.3f} seconds')

            # publish control inputs
            # print throttle with 2 decimals
            #print('throttle:', np.round(output_array_single_layer[:,0],2))
             
            self.publish_control_inputs(output_array_single_layer) # this works the same as the low level output because it's the first two values that get published
            #np.set_printoptions(precision=2, suppress=True)  # Set precision for NumPy


            # extact 

            if self.MPC_algorithm == 'CAMPCC':
                # 0        1        2     3     4     5   6  7  8 9 10    11    12 
                # th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading
                # high level and lowlevel are the same for single layer so send the same values
                x_high_level = output_array_single_layer[:,3]
                y_high_level = output_array_single_layer[:,4]
                x_low_level = output_array_single_layer[:,3]
                y_low_level = output_array_single_layer[:,4]
                x_path = output_array_single_layer[:,10]
                y_path = output_array_single_layer[:,11]
                heading_path = output_array_single_layer[:,12]
            else:
                # 0        1        2     3      4     5      6   7  8  9  10 
                # th_input,st_input,s_dot,slack, pos_x,pos_y, yaw,vx,vy,w  ,s
                x_high_level = output_array_single_layer[:,4]
                y_high_level = output_array_single_layer[:,5]
                x_low_level = output_array_single_layer[:,4]
                y_low_level = output_array_single_layer[:,5]
                # this is more triky because it is not redily available
                # get s_vec 
                s_vec = output_array_single_layer[:,10]
                #evaluate the path quantities
                x_path = np.interp(s_vec/local_path_length, labels_s, labels_x)
                y_path = np.interp(s_vec/local_path_length, labels_s, labels_y)
                heading_path = np.interp(s_vec/local_path_length, labels_s, labels_heading)







        # plot in rviz
        if self.minimal_plotting == False:
            self.produce_and_publish_rviz_visualization(x_high_level,
                                                        y_high_level,
                                                        x_path,
                                                        y_path,
                                                        heading_path,
                                                        x_low_level,
                                                        y_low_level,
                                                        xyyaw_ref_path)
            
        # publish computation time
        stop_clock_time = rospy.get_rostime()
        total_time = (stop_clock_time - start_clock_time).to_sec()
        self.comptime_publisher.publish(total_time)

        # publish GUI values
        GUI_param_values = [self.V_target,
                            self.q_con,
                            self.q_lag,
                            self.q_u_yaw_rate,
                            self.q_sdot,
                            self.qt_pos_high,
                            self.qt_rot_high,
                            self.qt_s_high,
                            self.q_v, 
                            self.q_pos, 
                            self.q_rot, 
                            self.qt_pos, 
                            self.qt_rot, 
                            self.q_acc,
                            self.lane_width,
                            self.minimal_plotting, 
                            self.delay_compensation, 
                            self.solver_software,
                            self.actuator_dynamics,
                            self.MPC_algorithm, 
                            self.dynamic_model]
        # convert values to a list of strings
        GUI_param_values_str = [str(i) for i in GUI_param_values]
        # create a name-value pair list with the names from the GUI_param_names_list
        list_name_val_str = []
        for i in range(len(GUI_param_values)):
            list_name_val_str.append(self.GUI_param_names_list[i])
            list_name_val_str.append(GUI_param_values_str[i])
        
        msg_GUI = String()
        msg_GUI.data = ",".join(list_name_val_str)
        self.GUI_param_names_publisher.publish(msg_GUI)




    def set_solver_type(self,solver_software, MPC_algorithm, dynamic_model,single_layer,actuator_dynamics):
        # delete all previous solvers
        print('setting solver type')

        if single_layer==False:
            # --- load high level solver for reference generation ---
            if MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
                self.high_level_solver_generator_obj = generate_high_level_path_planner_ocp(MPC_algorithm)
            elif MPC_algorithm == 'MPCC_PP':
                self.high_level_solver_generator_obj = generate_high_level_MPCC_PP()


            if solver_software == 'ACADOS':
                high_level_solver_path = os.path.join(self.solvers_folder_path,
                                                        self.high_level_solver_generator_obj.solver_name_acados,
                                                        self.high_level_solver_generator_obj.solver_name_acados + '.json')
                # check if the file exists
                if os.path.isfile(high_level_solver_path) == False:
                    print('')
                    print('Warning! The HIGH LEVEL solver location is invalid')
                    print('')
                else:
                    self.high_level_ocp = self.high_level_solver_generator_obj.produce_ocp()
                    self.high_level_solver = AcadosOcpSolver(self.high_level_ocp, json_file=high_level_solver_path, build=False, generate=False)
                    print('________________________________________________________________________________________')
                    print('Successfully loaded high level solver: ' + self.high_level_solver_generator_obj.solver_name_acados)

            elif solver_software == 'FORCES':
                import forcespro.nlp

                # check if folder exists
                high_level_solver_path = os.path.join(self.solvers_folder_path,self.high_level_solver_generator_obj.solver_name_forces)
                if os.path.isdir(high_level_solver_path) == False:
                    print('')
                    print('Warning! The HIGH LEVEL solver location is invalid')
                    print('')
                else:
                    self.high_level_solver = forcespro.nlp.Solver.from_directory(high_level_solver_path)
                    print('________________________________________________________________________________________')
                    print('Successfully loaded high level solver: ' + self.high_level_solver_generator_obj.solver_name_forces)
            # adjust control rate
            self.dt_controller_rate = self.high_level_solver_generator_obj.time_horizon/self.high_level_solver_generator_obj.N


            # --- load low level solver for control generation---
            self.low_level_solver_generator_obj = generate_low_level_solver_ocp(dynamic_model)
            if solver_software == 'ACADOS':
                low_level_solver_path = os.path.join(self.solvers_folder_path,
                                                        self.low_level_solver_generator_obj.solver_name_acados,
                                                        self.low_level_solver_generator_obj.solver_name_acados + '.json')
                
                if os.path.isfile(low_level_solver_path) == False:
                    print('')
                    print('Warning! The LOW LEVEL solver location is invalid')
                    print('')
                else:
                    self.low_level_ocp = self.low_level_solver_generator_obj.produce_ocp()
                    self.low_level_solver = AcadosOcpSolver(self.low_level_ocp, json_file=low_level_solver_path, build=False, generate=False)
                    print('Successfully loaded low level solver: ' + self.low_level_solver_generator_obj.solver_name_acados)

            elif solver_software == 'FORCES':
                # check if folder exists
                low_level_solver_path = os.path.join(self.solvers_folder_path,self.low_level_solver_generator_obj.solver_name_forces)
                if os.path.isdir(low_level_solver_path) == False:
                    print('')
                    print('Warning! The LOW LEVEL solver location is invalid')
                    print('')
                else:
                    self.low_level_solver = forcespro.nlp.Solver.from_directory(low_level_solver_path)
                    print('Successfully loaded low level solver: ' + self.low_level_solver_generator_obj.solver_name_forces)

        
        
        else: #load single track solver
            if MPC_algorithm == 'CAMPCC':                                
                self.single_layer_solver_generator_obj = generate_single_layer_CAMPCC(dynamic_model,actuator_dynamics,self.actuator_dynamics_params_folder,self.GP_params_folder)
            
            elif MPC_algorithm == 'MPCC_PP':
                
                self.single_layer_solver_generator_obj = generate_single_layer_MPCCPP(dynamic_model,self.GP_params_folder)

            if solver_software == 'ACADOS':
                single_layer_solver_path = os.path.join(self.solvers_folder_path,
                                                        self.single_layer_solver_generator_obj.solver_name_acados,
                                                        self.single_layer_solver_generator_obj.solver_name_acados + '.json')
                self.single_layer_ocp = self.single_layer_solver_generator_obj.produce_ocp()
                self.single_layer_solver = AcadosOcpSolver(self.single_layer_ocp, json_file=single_layer_solver_path, build=False, generate=False)
                print('________________________________________________________________________________________')
                print('Successfully loaded single layer solver: ' + self.single_layer_solver_generator_obj.solver_name_acados)

            elif solver_software == 'FORCES':
                import forcespro.nlp
                single_layer_solver_path = os.path.join(self.solvers_folder_path,self.single_layer_solver_generator_obj.solver_name_forces)
                self.single_layer_solver = forcespro.nlp.Solver.from_directory(single_layer_solver_path)
                print('________________________________________________________________________________________')
                print('Successfully loaded single layer solver: ' + self.single_layer_solver_generator_obj.solver_name_forces)
            
            # adjust control rate
            self.dt_controller_rate = self.single_layer_solver_generator_obj.time_horizon/self.single_layer_solver_generator_obj.N

        
        print('control rate (dt):', np.round(self.dt_controller_rate,3))
        print('________________________________________________________________________________________')


    def load_actuator_dynamics(self,path_2_folder):
        print('loading actuator dynamics from folder: ', path_2_folder)
        # load the actuator dynamics parameters
        dt = np.load(path_2_folder + '/dt.npy').item()
        n_past_actions = np.load(path_2_folder + '/n_past_actions.npy').item()
        weights_th = np.load(path_2_folder + '/weights_throttle.npy')
        weights_st = np.load(path_2_folder + '/weights_steering.npy')


        self.dt_FIR = dt
        self.n_past_actions_FRI = n_past_actions
        self.weights_th_FIR = weights_th
        self.weights_st_FIR = weights_st

        # find the first value from the end of the weights that is larger than 10-6
        n_past_actions_th = np.where(np.abs(weights_th) > 10**-6)[0][-1] + 2
        n_past_actions_st = np.where(np.abs(weights_st) > 10**-6)[0][-1] + 2

        
        time_vec_th = np.arange(0,dt*n_past_actions_th,dt)
        time_vec_st = np.arange(0,dt*n_past_actions_st,dt)
        dt_solver = self.time_horizon / self.N
        n_past_actions_th_solver = int(np.ceil(dt*n_past_actions_th/dt_solver)) 
        n_past_actions_st_solver = int(np.ceil(dt*n_past_actions_st/dt_solver)) 

        time_vec_th_solver = np.arange(0,dt_solver*n_past_actions_th_solver+dt_solver*0.5,dt_solver)
        time_vec_st_solver = np.arange(0,dt_solver*n_past_actions_st_solver+dt_solver*0.5,dt_solver)

        # now interpolate the weights to the solver time horizon
        weights_th_solver = np.interp(time_vec_th_solver,time_vec_th,np.squeeze(weights_th[:n_past_actions_th]),right=0)
        weights_st_solver = np.interp(time_vec_st_solver,time_vec_st,np.squeeze(weights_st[:n_past_actions_st]),right=0)

        # set small values to 0 to simplyfy things
        threshold = 10**-6
        weights_th_solver[np.abs(weights_th_solver) < threshold] = 0
        weights_st_solver[np.abs(weights_st_solver) < threshold] = 0

        # assign to self
        self.weights_th_FIR_solver = weights_th_solver[:-1] / np.sum(weights_th_solver[:-1]) # skip last value that will be 0 (this was needed to interpolate correctly)
        self.weights_st_FIR_solver = weights_st_solver[:-1] / np.sum(weights_st_solver[:-1])

        # # #  VERY TEMPORARY for debugging
        # print('TEMPORARY: setting weights to 0 except for the first element')
        # # replace with zeros except a one for the first element
        # self.weights_th_FIR_solver = np.zeros_like(self.weights_th_FIR_solver)
        # #self.weights_st_FIR_solver = np.zeros_like(self.weights_st_FIR_solver)
        # self.weights_th_FIR_solver[1] = 1
        # self.weights_st_FIR_solver[1] = 1

        # to check at solver build time that the weights are correct
        print('loaded actuator dynamics parameters')
        print('throttle FIR weights: ', self.weights_th_FIR_solver)
        print('steering FIR weights: ', self.weights_st_FIR_solver)

        self.n_past_th = len(self.weights_th_FIR_solver)
        self.n_past_st = len(self.weights_st_FIR_solver) 

        # prodece A_w matrix for actuator signal tracking
        self.Aw_th = np.zeros((self.n_past_th, self.n_past_th))
        self.Aw_st = np.zeros((self.n_past_st, self.n_past_st))

        for ii in range(self.n_past_th):
            w_assignment = np.fliplr(self.weights_th_FIR_solver[:ii+1])
            self.Aw_th[ii,:ii] = w_assignment
        for ii in range(self.n_past_st):
            w_assignment = np.fliplr(self.weights_st_FIR_solver[:ii+1])
            self.Aw_st[ii,:ii] = w_assignment



    def initialize_constant_parameters(self):

        # high level parameters
        self.V_target = 1  # in terms of being scaled down the proportion is vreal life[km/h] = v[m/s]*42.0000  (assuming the 30cm jetracer is a 3.5 m long car)
        
        # # mpc stage cost tuning weights
        self.q_v = 1  # relative weight of s_dot following
        self.q_con = 1  # relative weight of lat error
        self.q_lag = 1  # relative weight of lag error (only for MPCC)
        self.q_u_yaw_rate = 0.1  # relative weight of inputs
        self.q_sdot = 0.1  # relative weight of sdot for MPCC_PP fromulation
        self.qt_pos_high = 1  # relative weight of missing end point
        self.qt_rot_high = 1  # relative weight of path allignment
        self.qt_s_high = 1  # relative weight of progress along the path weight

        # mpc terminal cost tuning weights --> j term = qt_pos * err_pos_sqrd + qt_rot * misallignment ** 2
        self.q_pos = 1  # relative weight of missing end point
        self.q_rot = 1  # relative weight of path allignment
        self.qt_pos = 10  # relative weight of missing end point
        self.qt_rot = 10  # relative weight of path allignment
        self.q_u = 0.01  # relative weight of inputs
        self.q_acc = 0.01  # relative weight of acceleration 
        # slack cost tuning weight
        self.slack_p_1 = 1000  # controls the cost of the slack variable
        
        # constraint related parameters SORT THIS OUT LATER
        self.lane_width = 0.6  # width of lane
        


    
    

       






    def set_up_high_level_solver(self,Ds_back,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,V_target, local_path_length, labels_x,labels_y,labels_heading,labels_k,labels_s):
        # set smalle oprimization step
        #self.high_level_solver.options_set('step_length',0.75)

        # x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
        # x y yaw s (MPCC_PP)
        xinit = np.zeros(self.high_level_solver_generator_obj.nx) # all zeros
        xinit[0] = pos_x_init_rot
        xinit[1] = pos_y_init_rot
        xinit[2] = yaw_init_rot
        xinit[3] = Ds_back # setting to v target

        # # define parameters and first guess
        if self.MPC_algorithm == 'MPCC' or self.MPC_algorithm == 'CAMPCC':
                                
            params_i = np.array([V_target, local_path_length, self.q_con, self.q_lag, self.q_u_yaw_rate, self.qt_pos_high, self.qt_rot_high,self.lane_width,self.qt_s_high,*labels_k])
            # define first guess
            X0_array_high_level = self.high_level_solver_generator_obj.produce_X0(self.V_target,local_path_length,labels_k,labels_s,labels_x,labels_y,labels_heading) 
        
        elif self.MPC_algorithm == 'MPCC_PP':
            #this uses different states and initial conditions
                               # V_target, local_path_length,      q_con,      q_lag,      q_u,               q_sdot,      qt_pos,           qt_rot,          lane_width,     qt_s_high ,labels_x,  labels_y,   labels_heading
            params_i = np.array([V_target, local_path_length, self.q_con, self.q_lag, self.q_u_yaw_rate, self.q_sdot ,self.qt_pos_high, self.qt_rot_high,self.lane_width,self.qt_s_high,*labels_x, *labels_y, *labels_heading])
            # define first guess
            X0_array_high_level = self.high_level_solver_generator_obj.produce_X0(self.V_target,local_path_length,labels_x,labels_y,labels_heading)
        


        # stack parameters for all time steps
        param_array = np.zeros((self.high_level_solver_generator_obj.N+1, self.high_level_solver_generator_obj.n_parameters))
        for i in range(self.high_level_solver_generator_obj.N+1):
            param_array[i,:] = params_i

        # assign the value to the solver
        if self.solver_software == 'FORCES':
            # - set up initial guess and parameters
            x0_array_forces = X0_array_high_level.ravel()
            all_params_array_forces = param_array.ravel()
            # , "reinitialize": False
            problem_high_leval = {"x0": x0_array_forces, "xinit": xinit, "all_parameters": all_params_array_forces, "reinitialize": False} 
        else: # ACADOS

            # assign initial state
            self.high_level_solver.set(0, "lbx", xinit)
            self.high_level_solver.set(0, "ubx", xinit)

            # assign parameters
            for i in range(self.high_level_solver_generator_obj.N+1):
                self.high_level_solver.set(i, "p", params_i)

            # assign frist guess
            for i in range(self.high_level_solver_generator_obj.N):
                self.high_level_solver.set(i, "u", X0_array_high_level[i,:self.high_level_solver_generator_obj.nu])
                self.high_level_solver.set(i, "x", X0_array_high_level[i, self.high_level_solver_generator_obj.nu:])
            self.high_level_solver.set(self.high_level_solver_generator_obj.N, "x", X0_array_high_level[self.high_level_solver_generator_obj.N, self.high_level_solver_generator_obj.nu:])
            problem_high_leval = [] # dummy value if using acados

        return problem_high_leval


    def set_up_low_level_solver_problem(self,
                                        x_high_level,
                                        y_high_level,
                                        yaw_high_level,
                                        yaw_rate_high_level,
                                        x_path,
                                        y_path,
                                        V_target,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega):

        # define initial condition (it's the same for both solvers)
        xinit = np.zeros(self.low_level_solver_generator_obj.nx) # all zeros
                    # x y yaw vx vy w

        xinit[0] = pos_x_init_rot
        xinit[1] = pos_y_init_rot
        xinit[2] = yaw_init_rot
        xinit[3] = vx # setting to v target
        xinit[4] = vy
        xinit[5] = omega

        # define parameters
        param_array = np.zeros((self.low_level_solver_generator_obj.N+1, self.low_level_solver_generator_obj.n_parameters))
        params_base = np.array([V_target, self.q_v, self.q_pos, self.q_rot, self.q_u, self.qt_pos, self.qt_rot, self.q_acc])
        for i in range(self.low_level_solver_generator_obj.N+1):
            # append ref positions
            param_array[i,:] = np.array([*params_base, x_high_level[i], y_high_level[i], yaw_high_level[i], x_path[i], y_path[i], self.lane_width])
        
        # define first guess
        X0_array = self.low_level_solver_generator_obj.produce_X0(  V_target,
                                                                    x_high_level,
                                                                    y_high_level,
                                                                    yaw_high_level,
                                                                    yaw_rate_high_level)
        


        # --- set up problem differently depending on the solver software ---

        if self.solver_software == 'FORCES':
            # - set up initial guess and parameters
            x0_array_forces = X0_array[:-1,:].ravel() # unpack row-wise (forces has 1 less state than acados)
            all_params_array_forces = param_array[:-1].ravel() # unpack row-wise (forces has 1 less state than acados)

            problem = {"x0": x0_array_forces, "xinit": xinit, "all_parameters": all_params_array_forces}

        else: # ACADOS

            # set up parameters
            for i in range(self.low_level_solver_generator_obj.N+1):
                self.low_level_solver.set(i, "p", param_array[i,:])
            # set up initial condition
            self.low_level_solver.set(0, "lbx", xinit)
            self.low_level_solver.set(0, "ubx", xinit)
            # Initial guess for state trajectory
            X0_array = self.low_level_solver_generator_obj.produce_X0(  V_target,
                                                                        x_high_level,
                                                                        y_high_level,
                                                                        yaw_high_level,
                                                                        yaw_rate_high_level)

            # assign frist guess
            for i in range(self.low_level_solver_generator_obj.N):
                self.low_level_solver.set(i, "u", X0_array[i,:self.low_level_solver_generator_obj.nu])
                self.low_level_solver.set(i, "x", X0_array[i, self.low_level_solver_generator_obj.nu:])
            self.low_level_solver.set(self.low_level_solver_generator_obj.N, "x", X0_array[self.low_level_solver_generator_obj.N, self.low_level_solver_generator_obj.nu:])


            problem = [] # dummy value if using acados 


        return problem
    

    def set_up_single_layer_solver_problem(self,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega,
                                           V_target, q_v,local_path_length,labels_k,labels_s,Ds_back,labels_x,labels_y,labels_heading):
        # pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading
        xinit = np.zeros(self.single_layer_solver_generator_obj.nx) # all zeros
        xinit[0] = pos_x_init_rot
        xinit[1] = pos_y_init_rot
        xinit[2] = yaw_init_rot
        xinit[3] = np.max([vx,0.3]) 
        xinit[4] = vy
        xinit[5] = omega 
        xinit[6] = Ds_back # s is the current position along the path

        if self.actuator_dynamics:
            print('past inputs not yet supported')
            # initialize past actions
            n_th_past_actions = self.single_layer_solver_generator_obj.weights_th_FIR_solver.shape[0] -1
            n_st_past_actions = self.single_layer_solver_generator_obj.weights_st_FIR_solver.shape[0] -1

            # print values
            print('n_th_past_actions:',self.th_past_actions[:n_th_past_actions])
            print('n_st_past_actions:',self.st_past_actions[:n_st_past_actions])
            past_th_st = [*self.th_past_actions[:n_th_past_actions],*self.st_past_actions[:n_st_past_actions]]
            xinit[10:] = past_th_st


        # the other states should be zero

        # stack parameters for all time steps
                            #local_path_length,       q_con,      q_u,     q_acc,     qt_pos,      qt_rot,    lane_width,        qt_s_high,  q_v, labels_k
        if self.MPC_algorithm == 'CAMPCC':
            params_i = np.array([local_path_length, self.q_con, self.q_u, self.q_acc, self.qt_pos_high, self.qt_rot_high, self.lane_width, self.qt_s_high, q_v,*labels_k])
        elif self.MPC_algorithm == 'MPCC_PP':
                              #  local_path_length,      q_con,      q_lag,      q_u,     q_sdot,     q_acc,      qt_pos,           qt_rot,           lane_width,      qt_s_high,     q_v, labels_x,  labels_y,  labels_heading
            params_i = np.array([local_path_length, self.q_con, self.q_lag, self.q_u,self.q_sdot,self.q_acc, self.qt_pos_high, self.qt_rot_high, self.lane_width, self.qt_s_high,self.q_v,*labels_x, *labels_y, *labels_heading])


        param_array = np.zeros((self.single_layer_solver_generator_obj.N+1, self.single_layer_solver_generator_obj.n_parameters))
        for i in range(self.single_layer_solver_generator_obj.N+1):
            param_array[i,:] = params_i

        # X0
        X0_array_single_layer = self.single_layer_solver_generator_obj.produce_X0(V_target,local_path_length,labels_k,labels_s)   

        # assign the value to the solver
        if self.solver_software == 'FORCES':
            # - set up initial guess and parameters
            x0_array_forces = X0_array_single_layer.ravel()
            all_params_array_forces = param_array.ravel()

            if self.reinitialize == True:
                print('resetting warm start first guess')
                self.set_solver_type(self.solver_software,self.MPC_algorithm,self.dynamic_model,self.single_layer,self.actuator_dynamics)
                

            try:
                problem_single_layer = {"x0":x0_array_forces,"xinit": xinit, "all_parameters": all_params_array_forces,"reinitialize": self.reinitialize} #  ,"reinitialize": self.reinitialize
            except:
                problem_single_layer = {"x0":x0_array_forces,"xinit": xinit, "all_parameters": all_params_array_forces} #  
            #self.reinitialize = False # set to false after first call
        
        else: # ACADOS

            # assign initial state
            self.single_layer_solver.set(0, "lbx", xinit)
            self.single_layer_solver.set(0, "ubx", xinit)

            # assign parameters
            for i in range(self.single_layer_solver_generator_obj.N+1):
                self.single_layer_solver.set(i, "p", params_i)

            # assign frist guess
            for i in range(self.single_layer_solver_generator_obj.N):
                self.single_layer_solver.set(i, "u", X0_array_single_layer[i,:self.single_layer_solver_generator_obj.nu])
                self.single_layer_solver.set(i, "x", X0_array_single_layer[i, self.single_layer_solver_generator_obj.nu:])
            self.single_layer_solver.set(self.single_layer_solver_generator_obj.N, "x", X0_array_single_layer[self.single_layer_solver_generator_obj.N, self.single_layer_solver_generator_obj.nu:])
            


            problem_single_layer = [] # dummy value if using acados
        return problem_single_layer


    
    def check_solver_convergence(self,exitflag,solver_converged_previous,hig_low_single_tag):

        if hig_low_single_tag == 0:
            solver_level = 'HIGH level'
        elif hig_low_single_tag == 1:
            solver_level = 'LOW level'
        elif hig_low_single_tag == 2:
            solver_level = 'SINGLE level'

        # define the different exit flags for the different solvers
        if self.solver_software == 'FORCES':
            all_good_number = 1
            maxit_number = 0
        elif self.solver_software == 'ACADOS':
            all_good_number = 0
            maxit_number = 1

        # check if solver converged
        if exitflag != all_good_number:
            solver_converged = False
            if exitflag == maxit_number:
                maxit_reached = True
            else:
                maxit_reached = False
        else: 
            solver_converged = True

        # print out messages for the user
        if solver_converged == True and solver_converged_previous == True:
            pass # all good in the neighbourhood
        elif solver_converged == True and solver_converged_previous == False:
            print(solver_level, 'solver recovered from previous failure, now converged')
            print(' ')
            print(' ----------------- ')

        elif solver_converged == False:
            print(solver_level, self.solver_software + f" solver failed with exitflag/status {exitflag}")
            if maxit_reached == True:
                print('Max iterations reached')

        # as a recovery measure re-initialize the solver from the standard first guess
        if hig_low_single_tag == 2:
            if exitflag == all_good_number or maxit_reached == True: # don't reinitialize if the solver reached max iter cause they are usually very low
                self.reinitialize = False
            else:
                self.reinitialize = True # reset if the solver did not converge


        # publish if the solver converged or not
        if hig_low_single_tag == 0 or hig_low_single_tag == 2: #only for high and single layer
            self.solver_converged_publisher.publish(solver_converged)

        return solver_converged

        



    def publish_control_inputs(self, output_array_low_level):
        #print('last converged', self.last_converged)    
        # publish input values
        # for i in range(output_array_low_level.shape[0]):
        #     print(f"throttle: {output_array_low_level[i, 0]:.2f}, "
        #         f"steering: {output_array_low_level[i, 1]:.2f}, "
        #         f"slack: {output_array_low_level[i, 2]:.2f}, "
        #         f"vx: {output_array_low_level[i, 6]:.2f}, "
        #         f"vy: {output_array_low_level[i, 7]:.2f}, "
        #         f"omega: {output_array_low_level[i, 8]:.2f}")

        if self.actuator_dynamics_compensation == False:
        # just publush the values
            throttle_val = Float32(output_array_low_level[0, 0])
            steering_val = Float32(output_array_low_level[0, 1])

        # account for actuator dynamics if enabled
        else:
            # update actutation queue
            # self.th_queue = np.array([output_array_low_level[self.delay_act_steps, 0],*self.th_queue[:-1]])
            # self.st_queue = np.array([output_array_low_level[self.delay_act_steps, 1],*self.st_queue[:-1]])

            # publish the last value

            throttle_val = Float32(output_array_low_level[self.delay_act_steps, 0])
            steering_val = Float32(output_array_low_level[self.delay_act_steps, 1])

            #print('throttle queue:',self.th_queue)

            # #solve the actuator action tracking problem
            # b_th = self.B_th @ self.weights_th_FIR_solver
            # b_st = self.B_st @ self.weights_st_FIR_solver

            # th_target = output_array_low_level[:self.n_past_th, 0]
            # st_target = output_array_low_level[:self.n_past_st, 1]

            # th_solution = solve_triangular(self.Aw_th, th_target - b_th, lower=True)
            # st_solution = solve_triangular(self.Aw_st, st_target - b_st, lower=True)


            # # update past actions in the B matrices
            # # add current value on the diagonal
            # self.B_th = np.fill_diagonal(self.B_th, output_array_low_level[0, 0])
            # self.B_st = np.fill_diagonal(self.B_st, output_array_low_level[0, 1])

            # # push previous values to the right
            # for kk in range(self.n_past_th):
            #     self.B_th[kk,1:] = self.B_th[kk,:-1]
            # for kk in range(self.n_past_st):
            #     self.B_st[kk,1:] = self.B_st[kk,:-1] 


        self.throttle_publisher.publish(throttle_val)
        self.steering_publisher.publish(steering_val)


        # for data storage purpouses
        self.throttle = throttle_val.data
        self.steering = steering_val.data




        # update past values
        if self.safety_value == 0:
            self.th_past_actions = [0.0, *self.th_past_actions[:-1]]
        else:
            self.th_past_actions = [self.throttle, *self.th_past_actions[:-1]]
        self.st_past_actions = [self.steering, *self.st_past_actions[:-1]]


        # publishe timestamped versions of the inputs to get the time delay data
        msg_mpc_th = ThreeTimeStampsFloat32()
        msg_mpc_st = ThreeTimeStampsFloat32()
        # add header 1 timestamp
        time_now = rospy.Time.now()
        msg_mpc_th.header1.stamp = time_now
        msg_mpc_st.header1.stamp = time_now
        #add the inputs
        msg_mpc_th.data = throttle_val.data
        msg_mpc_st.data = steering_val.data



            


        # publish the messages
        self.mpc_throttle_publisher.publish(msg_mpc_th)
        self.mpc_steering_publisher.publish(msg_mpc_st)

        if self.synchronous_simulation:
            # publish next step in open loop solution
            pos_x_init_rot, pos_y_init_rot, yaw_init_rot,xyyaw_ref_path = self.relative_xyyaw_to_current_path(self.x_y_yaw_state,self.s)
            transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(output_array_low_level[1,3],
                                                                                output_array_low_level[1,4],
                                                                                xyyaw_ref_path)
            # clip yaw to [-pi, pi]
            yaw_next = xyyaw_ref_path[2] + output_array_low_level[1,5]  #np.arctan2(np.sin(output_array_low_level[1,5]), np.cos(output_array_low_level[1,5]))
            
            # simulate vicon motion capture system output
            #publish rviz vehicle visualization
            rviz_message = PoseStamped()
            quaternion = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0,yaw_next)
            rviz_message.pose.position.x = transformed_x[0]
            rviz_message.pose.position.y = transformed_y[0]
            rviz_message.pose.orientation.x = quaternion[0]
            rviz_message.pose.orientation.y = quaternion[1]
            rviz_message.pose.orientation.z = quaternion[2]
            rviz_message.pose.orientation.w = quaternion[3]

            # frame data is necessary for rviz
            rviz_message.header.frame_id = 'map'
            self.pub_rviz_vehicle_visualization.publish(rviz_message)
            
            if self.safety_value == 1:
                self.x_y_yaw_state = np.array([transformed_x[0],transformed_y[0],yaw_next])
                self.vx = output_array_low_level[1,6]
                self.vy = output_array_low_level[1,7]
                self.omega = output_array_low_level[1,8]

                self.vx_publisher.publish(Float32(self.vx))
                self.vy_publisher.publish(Float32(self.vy))
                self.w_publisher.publish(Float32(self.omega))




        






    def produce_and_publish_rviz_visualization(self,
                                               x_high_level,
                                               y_high_level,
                                               x_path,
                                               y_path,
                                               heading_path,
                                               x_low_level,
                                               y_low_level,
                                               x_y_yaw_rototranslation):
        #high_level_solver_array = [yaw_rate x y yaw ref_x ref_y ref_heading]
        

        # resend gloabal path just because it misses the first send
        rgba = [160, 189, 212, 0.25]
        global_path_message = self.produce_marker_array_rviz(self.x_vals_global_path, self.y_vals_global_path,rgba)
        self.rviz_global_path_publisher.publish(global_path_message)
        # ---- high level solver ----

        # local path from solver solution
        # u x y yaw s ref_x ref_y ref_heading
        rgba = [255, 0, 255, 0.5]
        transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(x_path, y_path,x_y_yaw_rototranslation)
        rviz_local_path_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_local_path_publisher.publish(rviz_local_path_message)

        # publish high level reference
        rgba = [0, 153, 76, 0.5]
        transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(x_high_level, y_high_level,x_y_yaw_rototranslation)
        rviz_high_level_reference_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_high_level_solution_publisher.publish(rviz_high_level_reference_message)

        # lane boundaries
        x_left_lane_boundary = np.zeros(x_path.shape[0])
        y_left_lane_boundary = np.zeros(x_path.shape[0])
        x_right_lane_boundary = np.zeros(x_path.shape[0])
        y_right_lane_boundary = np.zeros(x_path.shape[0])

        for ii in range(x_path.shape[0]):
            x_Cdev = np.cos(heading_path[ii])
            y_Cdev = np.sin(heading_path[ii])

            V_x_left = (self.lane_width/2) * x_Cdev
            V_y_left = ( self.lane_width/2) * y_Cdev
            V_x_right = (self.lane_width/2) * x_Cdev
            V_y_right = (self.lane_width/2) * y_Cdev

            x_left_lane_boundary[ii] = x_path[ii] - V_y_left
            y_left_lane_boundary[ii] = y_path[ii] + V_x_left
            x_right_lane_boundary[ii] = x_path[ii] + V_y_right
            y_right_lane_boundary[ii] = y_path[ii] - V_x_right



        # left lane boundary as for solver
        rgba = [57.0, 81.0, 100.0, 1.0]
        transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(x_left_lane_boundary, y_left_lane_boundary,x_y_yaw_rototranslation)
        rviz_left_lane_bound_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_left_lane_publisher.publish(rviz_left_lane_bound_message)

        # right lane boundary as for solver
        transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(x_right_lane_boundary, y_right_lane_boundary,x_y_yaw_rototranslation)
        rviz_right_lane_publisher_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_right_lane_publisher.publish(rviz_right_lane_publisher_message)


        # --- low level solver ---

        # open loop prediction mean
        rgba = [0, 166, 214, 1.0]
        transformed_x, transformed_y = self.rototranslate_path_2_abs_frame(x_low_level,y_low_level,x_y_yaw_rototranslation)
        rviz_MPCC_path_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_MPC_path_publisher.publish(rviz_MPCC_path_message)


    def produce_global_lane_boundaries_4_rviz(self):
        dim = len(self.x_4_local_path)
        # lane boundaries
        x_left_lane_boundary = np.zeros(dim)
        y_left_lane_boundary = np.zeros(dim)
        x_right_lane_boundary = np.zeros(dim)
        y_right_lane_boundary = np.zeros(dim)

        for ii in range(dim):
            x_Cdev = self.dx_ds[ii]
            y_Cdev = self.dy_ds[ii]

            V_x_left = (self.lane_width/2) * x_Cdev
            V_y_left = ( self.lane_width/2) * y_Cdev
            V_x_right = (self.lane_width/2) * x_Cdev
            V_y_right = (self.lane_width/2) * y_Cdev

            x_left_lane_boundary[ii] = self.x_4_local_path[ii] - V_y_left
            y_left_lane_boundary[ii] = self.y_4_local_path[ii] + V_x_left
            x_right_lane_boundary[ii] = self.x_4_local_path[ii] + V_y_right
            y_right_lane_boundary[ii] = self.y_4_local_path[ii] - V_x_right

        # now publish the lane boundaries
        rgba = [57.0, 81.0, 100.0, 1.0]
        rviz_left_lane_bound_message = self.produce_marker_array_rviz(x_left_lane_boundary, y_left_lane_boundary,rgba)
        self.rviz_global_left_lane_publisher.publish(rviz_left_lane_bound_message)

        rviz_right_lane_bound_message = self.produce_marker_array_rviz(x_right_lane_boundary, y_right_lane_boundary,rgba)
        self.rviz_global_right_lane_publisher.publish(rviz_right_lane_bound_message)





    def set_up_topics_for_rviz(self):

        self.rviz_MPC_path_publisher = rospy.Publisher('rviz_MPC_path_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_global_path_publisher = rospy.Publisher('rviz_global_path_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_local_path_publisher = rospy.Publisher('rviz_local_path_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_high_level_solution_publisher = rospy.Publisher('rviz_high_level_solution_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_local_path_publisher_tangent = rospy.Publisher('rviz_local_path_tangent_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_left_lane_publisher = rospy.Publisher('rviz_left_lane_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_right_lane_publisher = rospy.Publisher('rviz_right_lane_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_R_path_publisher = rospy.Publisher('rviz_R_path_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_initial_guess_publisher = rospy.Publisher('rviz_initial_guess_' + str(self.car_number), MarkerArray, queue_size=10)

        self.rviz_global_left_lane_publisher =  rospy.Publisher('rviz_global_left_lane_' + str(self.car_number), MarkerArray, queue_size=10)
        self.rviz_global_right_lane_publisher = rospy.Publisher('rviz_global_right_lane_' + str(self.car_number), MarkerArray, queue_size=10)


    def produce_marker_array_rviz(self, x, y, rgba):
        marker_array = MarkerArray()
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; LINE_STRIP: 4
        marker.type = 4
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
        #marker.pose.position.x = x[i]
        #marker.pose.position.y = y[i]
        #marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        points_list = []
        for i in range(len(x)):
            p = Point()
            p.x = x[i]
            p.y = y[i]
            p.z = 0.0
            points_list = points_list + [p]

        marker.points = points_list

        # assign to array
        marker_array.markers.append(marker)

        return  marker_array
    
    def produce_marker_array_rviz_arrows(self, x, y, z, rgba, vx, vy):
        """
        Produce an RViz MarkerArray to visualize the path and direction vectors as arrows.

        :param x: List of x-coordinates of the path.
        :param y: List of y-coordinates of the path.
        :param z: List of z-coordinates of the path (usually zeros for 2D).
        :param rgba: Color and alpha values (list of 4 elements).
        :param vx: List of x-components of the direction vectors.
        :param vy: List of y-components of the direction vectors.
        :return: MarkerArray with path visualization and direction arrows.
        """
        marker_array = MarkerArray()

        # Create a marker for the path (LINE_STRIP)
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.type = Marker.LINE_STRIP  # LINE_STRIP to visualize the path
        path_marker.id = 0
        path_marker.scale.x = 0.025  # Path line width
        path_marker.color.r = rgba[0] / 256
        path_marker.color.g = rgba[1] / 256
        path_marker.color.b = rgba[2] / 256
        path_marker.color.a = rgba[3]

        # Populate the points for the path
        points_list = []
        for i in range(len(x)):
            p = Point()
            p.x = x[i]
            p.y = y[i]
            p.z = z[i]
            points_list.append(p)

        path_marker.points = points_list
        marker_array.markers.append(path_marker)

        # Create markers for direction vectors (ARROWS)
        for i in range(len(x)):
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "map"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.type = Marker.ARROW  # ARROW to visualize direction vectors
            arrow_marker.id = i + 1  # Unique ID for each arrow

            # Adjust the scale of the arrow for better visualization
            arrow_marker.scale.x = 0.02  # Shaft diameter
            arrow_marker.scale.y = 0.035  # Head diameter
            arrow_marker.scale.z = 0.035  # Head length

            # Set the color of the arrow
            arrow_marker.color.r = rgba[0] / 256
            arrow_marker.color.g = rgba[1] / 256
            arrow_marker.color.b = rgba[2] / 256
            arrow_marker.color.a = rgba[3]

            # Define the start and end points of the arrow
            start_point = Point()
            start_point.x = x[i]
            start_point.y = y[i]
            start_point.z = z[i]

            arrow_scale = 0.2  # Scale down the direction vectors for visualization
            end_point = Point()
            end_point.x = x[i] + vx[i] * arrow_scale
            end_point.y = y[i] + vy[i] * arrow_scale
            end_point.z = z[i]  # Arrows are in the plane of the path

            arrow_marker.points = [start_point, end_point]  # Define the arrow geometry

            # Add the arrow marker to the MarkerArray
            marker_array.markers.append(arrow_marker)

        return marker_array



    def safety_value_subscriber_callback(self, msg):
        self.safety_value = msg.data

    def vicon_subscriber_callback(self,msg):

        # here we need to evaluate the velocities
        # for now a very simple derivation
        # extract current orientation
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
        # convert to Euler
        quaternion = [q_x, q_y, q_z, q_w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        #update past states
        # shift them back by 1 step and update the last value
        self.past_x_vicon[:-1] = self.past_x_vicon[1:]
        self.past_y_vicon[:-1] = self.past_y_vicon[1:]
        self.past_yaw_vicon[:-1] = self.past_yaw_vicon[1:]
        self.past_time_vicon[:-1] = self.past_time_vicon[1:]

        # add last entry
        self.past_x_vicon[-1] = msg.pose.pose.position.x
        self.past_y_vicon[-1] = msg.pose.pose.position.y
        self.past_yaw_vicon[-1] = yaw
        self.past_time_vicon[-1] = msg.header.stamp.to_sec()

        #evalaute velocities using finite differences on last values

        vx_abs = (self.past_x_vicon[-1] - self.past_x_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])
        vy_abs = (self.past_y_vicon[-1] - self.past_y_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])

        #convert to body frame
        self.vx = +vx_abs * np.cos(yaw) + vy_abs * np.sin(yaw)
        self.vy = -vx_abs * np.sin(yaw) + vy_abs * np.cos(yaw)



        # unwrap past angles to avoid jumps when flipping from - pi to + pi
        delta_yaw = self.past_yaw_vicon[-1] - self.past_yaw_vicon[0]
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi

        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        
        self.omega = (delta_yaw) / (self.past_time_vicon[-1] - self.past_time_vicon[0])


        # update the pose
        # if delay compensation is used, forward propagate the current state into the future
        if self.delay_compensation:
            #print('delay=',self.delay)
            #determine absolute velocities
            self.x_y_yaw_state = [msg.pose.pose.position.x + vx_abs * self.delay,
                                    msg.pose.pose.position.y+ vy_abs * self.delay,
                                    yaw + self.omega * self.delay]
        else:
            self.x_y_yaw_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]


        
        self.pose_msg_time = msg.header.stamp


        # publish velocity states for rviz
        self.vx_publisher.publish(Float32(self.vx))
        self.vy_publisher.publish(Float32(self.vy))
        self.w_publisher.publish(Float32(self.omega))
        
    def comm_delay_subscriber_callback(self,msg):
        # update the delay value
        self.delay = msg.data + 0.1









if __name__ == '__main__':
    try:
        # define where to find complied solvers

        rospy.init_node('MPCC_node', anonymous=False)
        global_comptime_publisher = rospy.Publisher('GLOBAL_comptime', Float32, queue_size=1)

        # define controller rate
        dt_controller_rate = 0.05

        #set up vehicle controllers
        #car 1
        car_number_1 = 1
        vehicle_1_controller = MPCC_controller_class(car_number_1,dt_controller_rate) 

        # start control loop
        
        rate = rospy.Rate(1 / dt_controller_rate)
        #NOTE that this rate is the rate to send out ALL control imputs to all vehicles

        vehicle_controllers_list = [vehicle_1_controller] # , vehicle_3_controller

        #set up GUI manager 
        MPC_GUI_manager_obj = MPC_GUI_manager(vehicle_controllers_list)

        while not rospy.is_shutdown():
            try:
                start_clock_time = rospy.get_rostime()
                # get controller frequency
                dt_controller = vehicle_controllers_list[0].dt_controller_rate
                rate = rospy.Rate(1 / dt_controller)
                # run 1 loop on all vehicles
                for i in range(len(vehicle_controllers_list)):
                    # check if vehicle is stationary
                    vehicle_controllers_list[i].run_one_MPCC_control_loop(  vehicle_controllers_list[i].x_y_yaw_state,
                                                                            vehicle_controllers_list[i].vx,
                                                                            vehicle_controllers_list[i].vy,
                                                                            vehicle_controllers_list[i].omega,
                                                                            vehicle_controllers_list[i].V_target)

                stop_clock_time = rospy.get_rostime()
                elapsed_time_global_loop = (stop_clock_time - start_clock_time).to_sec()
                global_comptime_publisher.publish(elapsed_time_global_loop)
            except Exception as e:
                print('Error in control loop:')
                traceback.print_exc()

            rate.sleep()




    except rospy.ROSInterruptException:
        pass
