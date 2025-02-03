#!/usr/bin/env python3
import numpy as np
import traceback
from scipy.spatial.transform import Rotation
import rospy
import os
from functions_for_MPCC_node_running import find_s_of_closest_point_on_global_path
from MPC_generate_solvers.path_track_definitions import generate_path_data



from std_msgs.msg import Float32
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from datetime import datetime
import csv
# for dynamic paramters reconfigure (setting param values from rqt_reconfigure GUI)
from dynamic_reconfigure.server import Server
from curvature_aware_mpcc_pkg.cfg import GUI_mpc_dynamic_reconfigureConfig
import rospkg
# THIS NEEDS TO BE FIXED (i.e. only use forces and acados stuff if necessary)
#import forcespro.nlp
from acados_template import AcadosOcpSolver

from tf.transformations import euler_from_quaternion
from MPC_generate_solvers.functions_for_solver_generation import generate_high_level_path_planner_ocp, generate_low_level_solver_ocp, generate_high_level_MPCC_PP


# TODO
# make sure N are the same between high and low level solvers









#fory dynamic parameter change using rqt_reconfigure GUI
class MPC_GUI_manager:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        self.solver_software_options = ['ACADOS' , 'FORCES']
        self.MPC_algorithm_options = ['MPCC', 'CAMPCC','MPCC_PP']
        self.dynamic_model_options = ['kinematic_bicycle', 'dynamic_bicycle']

        # as a last thing creat the server because it will be locked executing here
        srv = Server(GUI_mpc_dynamic_reconfigureConfig, self.reconfig_callback)

        


    def reconfig_callback(self, config, level):
        print('_________________________________________________')
        print('  reconfiguring parameters from dynamic_reconfig ')
        print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
        
        for i in range(len(self.vehicles_list)):
            # to check if some values have changed store the old values
            lane_width_old = self.vehicles_list[i].lane_width



            # high level solver
            self.vehicles_list[i].V_target = config['V_target']
            self.vehicles_list[i].q_con = config['q_con']
            self.vehicles_list[i].q_lag = config['q_lag']
            self.vehicles_list[i].q_u_yaw_rate = config['q_u_yaw_rate']
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
            self.vehicles_list[i].solver_software = config['Solver_software']
            # solver choices

            # this is because of how the slecetion works
            self.vehicles_list[i].solver_software = self.solver_software_options[config['Solver_software']]
            self.vehicles_list[i].MPC_algorithm = self.MPC_algorithm_options[config['MPC_algorithm']]
            self.vehicles_list[i].dynamic_model = self.dynamic_model_options[config['Dynamic_model']] 
            
            # set up solver type
            self.vehicles_list[i].set_solver_type(self.vehicles_list[i].solver_software,
                                                  self.vehicles_list[i].MPC_algorithm,
                                                  self.vehicles_list[i].dynamic_model) 
            
            # check if lane width has changed
            if lane_width_old != self.vehicles_list[i].lane_width:
                self.vehicles_list[i].produce_global_lane_boundaries_4_rviz()
            



            print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

        return config











class MPCC_controller_class():
    def __init__(self, car_number,dt_controller_rate):

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

        # initialize state variables
        self.vx = 0
        self.vy = 0
        self.omega = 0
        self.x_y_yaw_state = [2, 0, 0] # default to not start in a weird point with vicon_racetrack
        self.pose_msg_time = rospy.get_rostime() # initialize time of pose message


        # delay compensation if in the lab
        self.delay_compensation = True
        self.delay = 0.04 # communication delay in seconds (in the lab)

        # set p contingency if solver does not converge
        self.last_converged_high = True
        self.last_converged_low = True


        # define selected solver
        self.set_solver_type(self.solver_software, self.MPC_algorithm, self.dynamic_model)

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
        self.save_data = False
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

        self.s_vals_global_path,\
        self.x_vals_global_path,\
        self.y_vals_global_path,\
        self.s_4_local_path,\
        self.x_4_local_path,\
        self.y_4_local_path,\
        self.dx_ds, self.dy_ds, self.d2x_ds2, self.d2y_ds2,\
        self.k_vals_global_path,\
        self.k_4_local_path = generate_path_data(track_choice)


        # set up publishers for robot velocity estimates
        #set up past position variables
        past_states = 2  # actually this is past states + 1 for current state
        self.past_x_vicon = np.zeros(past_states)
        self.past_y_vicon = np.zeros(past_states)
        self.past_yaw_vicon = np.zeros(past_states)
        self.past_time_vicon = np.zeros(past_states)
        self.vx_publisher = rospy.Publisher('vx_mpc_' + str(car_number), Float32, queue_size=1)
        self.vy_publisher = rospy.Publisher('vy_mpc_' + str(car_number), Float32, queue_size=1)
        self.w_publisher = rospy.Publisher('w_mpc_' + str(car_number), Float32, queue_size=1)

        # set up subscribers (inputs to the controller)
        self.vicon_subscriber = rospy.Subscriber('vicon/jetracer' + str(car_number), PoseWithCovarianceStamped, self.vicon_subscriber_callback)

        # set up publishers (outpus of the controller)
        self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)
        self.steering_publisher = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=1)
        self.comptime_publisher = rospy.Publisher('comptime_' + str(car_number), Float32, queue_size=1)





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
        s, self.current_path_index = find_s_of_closest_point_on_global_path(np.array([x_y_yaw_state[0], x_y_yaw_state[1]]), self.s_vals_global_path,
                                                                  self.x_vals_global_path, self.y_vals_global_path,
                                                                  self.previous_path_index, estimated_ds)
        self.previous_path_index = self.current_path_index  # update index along the path to know where to search in next iteration
        self.s = s


        # produce Chebyshev coefficients that represent local path
        Ds_forward = 1.2 * V_target * self.high_level_solver_generator_obj.time_horizon #  self.dtt * self.high_level_solver_generator_obj.N
        Ds_back = 0.0 # this is the length of the path that is behind the car


        # ------ HIGH LEVEL SOLVER ------

        labels_k, local_path_length = self.produce_ylabels_4_local_kernelized_path(s,Ds_back,Ds_forward)
        pos_x_init_rot, pos_y_init_rot, yaw_init_rot,xyyaw_ref_path = self.relative_xyyaw_to_current_path(x_y_yaw_state) # current car state relative to current path index
        problem_high_level = self.set_up_high_level_solver(pos_x_init_rot, pos_y_init_rot, yaw_init_rot,V_target, local_path_length,labels_k)

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
                    u_i_solution = np.array([0.0, 0.0])
                else:
                    u_i_solution = self.high_level_solver.get(i, "u")
                x_i_solution = self.high_level_solver.get(i, "x")
                output_array_high_level[i] = np.concatenate((u_i_solution, x_i_solution))

        # check if solver converged
        self.last_converged_high = self.check_solver_convergence(exitflag_high,self.last_converged_high,0) # last input is the choice between high and low level solver
        
        # --------------------------------



        # ------ LOW LEVEL SOLVER ------
            
        problem_low_level = self.set_up_low_level_solver_problem(output_array_high_level,V_target,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega)

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

        # --------------------------------
        



        # publish control inputs
        self.publish_control_inputs(output_array_low_level)


        if self.minimal_plotting == False:
            self.produce_and_publish_rviz_visualization(output_array_high_level, output_array_low_level,xyyaw_ref_path)
            
        # publish computation time
        stop_clock_time = rospy.get_rostime()
        total_time = (stop_clock_time - start_clock_time).to_sec()
        self.comptime_publisher.publish(total_time)


    def set_solver_type(self,solver_software, MPC_algorithm, dynamic_model):
        print('setting solver type')

        # --- load high level solver for reference generation ---
        self.high_level_solver_generator_obj = generate_high_level_path_planner_ocp(MPC_algorithm)

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
            

        
        
        print('________________________________________________________________________________________')


    def initialize_constant_parameters(self):

        # high level parameters
        self.V_target = 1  # in terms of being scaled down the proportion is vreal life[km/h] = v[m/s]*42.0000  (assuming the 30cm jetracer is a 3.5 m long car)
        
        # # mpc stage cost tuning weights
        self.q_v = 1  # relative weight of s_dot following
        self.q_con = 1  # relative weight of lat error
        self.q_lag = 1  # relative weight of lag error (only for MPCC)
        self.q_u_yaw_rate = 0.1  # relative weight of inputs
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
        


    
    
    def produce_ylabels_4_local_kernelized_path(self,s,Ds_back,Ds_forward):
        #extract indexes of local path

        mask = (self.s_4_local_path >= s - Ds_back) & (self.s_4_local_path <= s + Ds_forward)
        local_path_length = Ds_back + Ds_forward
        # Extract the indexes where the condition is true
        indexes = np.where(mask)[0]
        s_data_points =  self.s_4_local_path[indexes]  # This will have the local path parametrized starting from 0
        k_data_points = self.k_4_local_path[indexes]

        # resample the data points to have a fixed number of points
        n = self.high_level_solver_generator_obj.n_points_kernelized 

        s_data_points_fit = np.linspace(s_data_points[0], s_data_points[-1], n)
        labels_k = np.interp(s_data_points_fit, s_data_points, k_data_points)

        return labels_k, local_path_length
       


    def relative_xyyaw_to_current_path(self,x_y_yaw_state):
        # evaluate current reference path and derivatives needed for initial conditions
        # find corresponding index for s on s_4_local path
        current_path_index_on_4_local_path = np.argmin(np.abs(self.s_4_local_path - self.s))
        x_ref_path = self.x_4_local_path[current_path_index_on_4_local_path]
        y_ref_path = self.y_4_local_path[current_path_index_on_4_local_path]
        dx_ds_ref_path = self.dx_ds[current_path_index_on_4_local_path]
        dy_ds_ref_path = self.dy_ds[current_path_index_on_4_local_path]
        #evaluate heading angle
        heading_angle_path = np.arctan2(dy_ds_ref_path, dx_ds_ref_path)

        #self.local_path_ref_x,self.local_path_ref_y, self.local_path_rot_angle
        # apply shift to the x y position of the car
        pos_x_0 =  x_y_yaw_state[0] - x_ref_path
        pos_y_0 =  x_y_yaw_state[1] - y_ref_path
        #now rotate to have the first point aligned with the x axis
        pos_x_init_rot =  pos_x_0 * np.cos(heading_angle_path) + pos_y_0 * np.sin(heading_angle_path)
        pos_y_init_rot = -pos_x_0 * np.sin(heading_angle_path) + pos_y_0 * np.cos(heading_angle_path)

        # apply rotation to yaw
        yaw_init_rot = self.x_y_yaw_state[2] - heading_angle_path

        # this is needed to keep the yaw angle from going over 2 pi
        if yaw_init_rot > np.pi:
            yaw_init_rot -= 2 * np.pi

        elif yaw_init_rot < -np.pi:
            yaw_init_rot += 2 * np.pi

        #
        xyyaw_ref_path = [x_ref_path, y_ref_path, heading_angle_path]


        return pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path



    def set_up_high_level_solver(self,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,V_target, local_path_length, labels_k):
        # set smalle oprimization step
        #self.high_level_solver.options_set('step_length',0.75)

        # x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
        # x y yaw s (MPCC_PP)
        xinit = np.zeros(self.high_level_solver_generator_obj.nx) # all zeros
        xinit[0] = pos_x_init_rot
        xinit[1] = pos_y_init_rot
        xinit[2] = yaw_init_rot

        # # define parameters and first guess
        if self.MPC_algorithm == 'MPCC' or self.MPC_algorithm == 'CAMPCC':
                                
            params_i = np.array([V_target, local_path_length, self.q_con, self.q_lag, self.q_u_yaw_rate, self.qt_pos_high, self.qt_rot_high,self.lane_width,self.qt_s_high,*labels_k])
            # define first guess
            X0_array_high_level = self.high_level_solver_generator_obj.produce_X0(self.V_target,local_path_length,labels_k)
        
        elif self.MPC_algorithm == 'MPCC_PP':
            #this uses different states and initial conditions
            params_i = np.array([V_target, local_path_length, self.q_con, self.q_lag, self.q_u_yaw_rate, self.qt_pos_high, self.qt_rot_high,self.lane_width,self.qt_s_high,*labels_x, *labels_y, *labels_yaw])
            # define first guess
            X0_array_high_level = self.high_level_solver_generator_obj.produce_X0(self.V_target,local_path_length,labels_k)
        


        # stack parameters for all time steps
        param_array = np.zeros((self.high_level_solver_generator_obj.N+1, self.high_level_solver_generator_obj.n_parameters))
        for i in range(self.high_level_solver_generator_obj.N+1):
            param_array[i,:] = params_i

        # assign the value to the solver
        if self.solver_software == 'FORCES':
            # - set up initial guess and parameters
            x0_array_forces = X0_array_high_level.ravel()
            all_params_array_forces = param_array.ravel()
            problem_high_leval = {"x0": x0_array_forces, "xinit": xinit, "all_parameters": all_params_array_forces} 
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


    def set_up_low_level_solver_problem(self,output_array_high_level,V_target,pos_x_init_rot, pos_y_init_rot, yaw_init_rot,vx,vy,omega):

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
            x_ref = output_array_high_level[i,2]
            y_ref = output_array_high_level[i,3]
            yaw_ref = output_array_high_level[i,4]
            x_path = output_array_high_level[i,6]
            y_path = output_array_high_level[i,7]
            # append ref positions
            param_array[i,:] = np.array([*params_base, x_ref, y_ref, yaw_ref, x_path, y_path, self.lane_width])
        
        # define first guess
        X0_array = self.low_level_solver_generator_obj.produce_X0(V_target, output_array_high_level)
            


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
            X0_array = self.low_level_solver_generator_obj.produce_X0(V_target, output_array_high_level)

            # assign frist guess
            for i in range(self.low_level_solver_generator_obj.N):
                self.low_level_solver.set(i, "u", X0_array[i,:self.low_level_solver_generator_obj.nu])
                self.low_level_solver.set(i, "x", X0_array[i, self.low_level_solver_generator_obj.nu:])
            self.low_level_solver.set(self.low_level_solver_generator_obj.N, "x", X0_array[self.low_level_solver_generator_obj.N, self.low_level_solver_generator_obj.nu:])


            problem = [] # dummy value if using acados 


        return problem
    

    
    def check_solver_convergence(self,exitflag,solver_converged_previous,hig_low_tag):
        if hig_low_tag == 0:
            solver_level = 'HIGH level'
        elif hig_low_tag == 1:
            solver_level = 'LOW level'

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

        return solver_converged

        



    def publish_control_inputs(self, output_array_low_level):
        #print('last converged', self.last_converged)    
        # publish input values
        throttle_val = Float32(output_array_low_level[0, 0])
        steering_val = Float32(output_array_low_level[0, 1])

        # for data storage purpouses
        self.throttle = throttle_val.data
        self.steering = steering_val.data

        self.throttle_publisher.publish(throttle_val)
        self.steering_publisher.publish(steering_val)



    def rototranslate(self, x, y,x_y_yaw):

        """
        Apply translation and rotation to x, y coordinates.

        Args:
        x: numpy array of x coordinates
        y: numpy array of y coordinates
        translation: (tx, ty) tuple for translation
        rotation_angle: Rotation angle in radians

        Returns:
        Transformed x, y coordinates as numpy arrays
        """
        # Create rotation matrix
        rotation_angle = x_y_yaw[2]
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta,  cos_theta]])

        # Apply rotation
        rotated_points = R @ np.vstack((x, y))  # Matrix multiplication

        # Apply translation
        tx = x_y_yaw[0]
        ty = x_y_yaw[1]
        transformed_x = rotated_points[0, :] + tx
        transformed_y = rotated_points[1, :] + ty

        return transformed_x, transformed_y

    def produce_and_publish_rviz_visualization(self, output_array_high_level, output_array_low_level,x_y_yaw_rototranslation):

        # resend gloabal path just because it misses the first send
        rgba = [160, 189, 212, 0.25]
        global_path_message = self.produce_marker_array_rviz(self.x_vals_global_path, self.y_vals_global_path,rgba)
        self.rviz_global_path_publisher.publish(global_path_message)

        # ---- high level solver ----

        # local path from solver solution
        # u x y yaw s ref_x ref_y ref_heading
        rgba = [255, 0, 255, 0.5]
        transformed_x, transformed_y = self.rototranslate(output_array_high_level[:,6], output_array_high_level[:,7],x_y_yaw_rototranslation)
        rviz_local_path_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_local_path_publisher.publish(rviz_local_path_message)

        # publish high level reference
        rgba = [0, 153, 76, 0.5]
        transformed_x, transformed_y = self.rototranslate(output_array_high_level[:,2], output_array_high_level[:,3],x_y_yaw_rototranslation)
        rviz_high_level_reference_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_high_level_solution_publisher.publish(rviz_high_level_reference_message)

        # lane boundaries
        x_left_lane_boundary = np.zeros(output_array_high_level.shape[0])
        y_left_lane_boundary = np.zeros(output_array_high_level.shape[0])
        x_right_lane_boundary = np.zeros(output_array_high_level.shape[0])
        y_right_lane_boundary = np.zeros(output_array_high_level.shape[0])

        for ii in range(output_array_high_level.shape[0]):
            x_Cdev = np.cos(output_array_high_level[ii,-1])
            y_Cdev = np.sin(output_array_high_level[ii,-1])

            V_x_left = (self.lane_width/2) * x_Cdev
            V_y_left = ( self.lane_width/2) * y_Cdev
            V_x_right = (self.lane_width/2) * x_Cdev
            V_y_right = (self.lane_width/2) * y_Cdev

            x_left_lane_boundary[ii] = output_array_high_level[ii,6] - V_y_left
            y_left_lane_boundary[ii] = output_array_high_level[ii,7] + V_x_left
            x_right_lane_boundary[ii] = output_array_high_level[ii,6] + V_y_right
            y_right_lane_boundary[ii] = output_array_high_level[ii,7] - V_x_right



        # left lane boundary as for solver
        rgba = [57.0, 81.0, 100.0, 1.0]
        transformed_x, transformed_y = self.rototranslate(x_left_lane_boundary, y_left_lane_boundary,x_y_yaw_rototranslation)
        rviz_left_lane_bound_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_left_lane_publisher.publish(rviz_left_lane_bound_message)

        # right lane boundary as for solver
        transformed_x, transformed_y = self.rototranslate(x_right_lane_boundary, y_right_lane_boundary,x_y_yaw_rototranslation)
        rviz_right_lane_publisher_message = self.produce_marker_array_rviz(transformed_x, transformed_y,rgba)
        self.rviz_right_lane_publisher.publish(rviz_right_lane_publisher_message)


        # --- low level solver ---

        # open loop prediction mean
        rgba = [0, 166, 214, 1.0]
        transformed_x, transformed_y = self.rototranslate(output_array_low_level[:,3], output_array_low_level[:,4],x_y_yaw_rototranslation)
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
                # run 1 loop on all vehicles
                for i in range(len(vehicle_controllers_list)):
                    # check if vehicle is stationary
                    vehicle_controllers_list[i].run_one_MPCC_control_loop(vehicle_controllers_list[i].x_y_yaw_state,
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
