from MPC_generate_solvers.functions_for_solver_generation import generate_high_level_path_planner_ocp, generate_high_level_MPCC_PP
import numpy as np
import os
import forcespro.nlp
import matplotlib.pyplot as plt
from mpc_node import path_handeling_utilities_class

# select the solver to build MPCC or CAMPCC
MPC_algorithm = 'MPCC' # 'MPCC' or 'CAMPCC'

# load test track
from functions_for_MPCC_node_running import find_s_of_closest_point_on_global_path
from MPC_generate_solvers.path_track_definitions import generate_path_data

track_choice = 'racetrack_vicon_2' # simpler racetrack

s_vals_global_path,\
x_vals_global_path,\
y_vals_global_path,\
s_4_local_path,\
x_4_local_path,\
y_4_local_path,\
dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
k_vals_global_path,\
k_4_local_path,\
heading_4_local_path = generate_path_data(track_choice)

# plot track
fig_track, axes_track = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))  # Wider figure for side-by-side layout
axes_track.plot(x_vals_global_path, y_vals_global_path, 'gray')
axes_track.axis('equal')
axes_track.set_title('Global track')
axes_track.set_xlabel('x [m]')
axes_track.set_ylabel('y [m]')







# generate object for path handeling utilities
path_handeling_utilities_obj = path_handeling_utilities_class()
# assign path data to the object
path_handeling_utilities_obj.s_4_local_path = s_4_local_path
path_handeling_utilities_obj.x_4_local_path = x_4_local_path
path_handeling_utilities_obj.y_4_local_path = y_4_local_path
path_handeling_utilities_obj.heading_4_local_path = heading_4_local_path
path_handeling_utilities_obj.k_4_local_path = k_4_local_path
path_handeling_utilities_obj.dx_ds = dx_ds
path_handeling_utilities_obj.dy_ds = dy_ds



# instantiate the class
high_level_solver_generator_obj = generate_high_level_path_planner_ocp(MPC_algorithm)


# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_solver = os.path.join(current_script_dir,'MPC_generate_solvers/solvers',high_level_solver_generator_obj.solver_name_forces)

# load pre-built solver
high_level_solver_forces = forcespro.nlp.Solver.from_directory(path_to_solver)


# ---- TESTING THE SOLVER by calling it with a test scenario ----
V_target = 3
local_path_length = V_target * high_level_solver_generator_obj.time_horizon * 1.5 # this is how it would be evalauted in the mpc
q_con = 1 
q_lag = 1 
q_u_yaw_rate = 0.01 #0.005 
qt_pos_high = 10
qt_rot_high = 0
lane_width = 0.5
qt_s_high = 10



# testing
s = 0
Ds_back = 0
Ds_forward = local_path_length
xyyaw_ref_path = np.array([0,0,0])
n = high_level_solver_generator_obj.n_points_kernelized 
labels_x,\
labels_y,\
labels_heading,\
labels_k,\
local_path_length,\
labels_s = path_handeling_utilities_obj.produce_ylabels_4_local_kernelized_path(s,Ds_back,Ds_forward,xyyaw_ref_path,n)

# set up initial position
# x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
# x y yaw s (MPCC_PP)


# assign initial position as starting on the path at a certain s_0
s = 0
current_path_index_on_4_local_path = np.argmin(np.abs(s_4_local_path - s))
x_0 = x_4_local_path[current_path_index_on_4_local_path]
y_0 = y_4_local_path[current_path_index_on_4_local_path]
yaw_0 = heading_4_local_path[current_path_index_on_4_local_path]
x_y_yaw_state = np.array([x_0,y_0,yaw_0])




# this will be in a loop
pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path = path_handeling_utilities_obj.relative_xyyaw_to_current_path(x_y_yaw_state,s)


# assign initial position (since all is relative to the current closest point on the path setting all to zero means starting on the path at s = s_0)
xinit = np.zeros(high_level_solver_generator_obj.nx) # all zeros
xinit[0] = pos_x_init_rot
xinit[1] = pos_y_init_rot
xinit[2] = yaw_init_rot


# set up parameters
params_i = np.array([V_target, local_path_length, q_con, q_lag, q_u_yaw_rate, qt_pos_high, qt_rot_high,lane_width,qt_s_high,*labels_k])

param_array = np.zeros((high_level_solver_generator_obj.N+1, high_level_solver_generator_obj.n_parameters))
# put all parameters together
for i in range(high_level_solver_generator_obj.N+1):
    param_array[i,:] = params_i
param_array = param_array.ravel() # unpack row-wise

# set up initial guess
if MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
    X0_array_high_level = high_level_solver_generator_obj.produce_X0(V_target,local_path_length,labels_k)
elif MPC_algorithm == 'MPCC_PP':
    X0_array_high_level = high_level_solver_generator_obj.produce_X0(V_target,local_path_length,labels_x,labels_y,labels_heading)

X0_array_high_level = X0_array_high_level.ravel() # unpack row-wise

# produce problem as a dictionary for forces
problem = {"x0": X0_array_high_level, "xinit": xinit, "all_parameters": param_array} # all_parameters






# --- solve the problem ---
output, exitflag, info = high_level_solver_forces.solve(problem)
output_array_high_level = np.array(list(output.values()))
print('')
print('')
# extract the x-y path
if MPC_algorithm == 'MPCC_PP':
    # reconstruct the path related quantities from the labels
    #              u_yaw_rate slack s_dot   x y yaw s (MPCC_PP)
    # extract 
    yaw_rate_high_level = output_array_high_level[:,0]
    x_out = output_array_high_level[:,3]
    y_out = output_array_high_level[:,4]
    yaw_high_level = output_array_high_level[:,5]
    # interpolate to get the path quantities
    s_output_vec = output_array_high_level[:,6]
    x_path = np.interp(s_output_vec/local_path_length, labels_s, labels_x)
    y_path = np.interp(s_output_vec/local_path_length, labels_s, labels_y)
    heading_path = np.interp(s_output_vec/local_path_length, labels_s, labels_heading)

elif MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
    #              u_yaw_rate slack   x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
    # extact 
    yaw_rate_high_level = output_array_high_level[:,0]
    x_out = output_array_high_level[:,2]
    y_out = output_array_high_level[:,3]
    yaw_high_level = output_array_high_level[:,4]
    x_path = output_array_high_level[:,6]
    y_path = output_array_high_level[:,7]
    heading_path = output_array_high_level[:,8]


# transform the path to global coordinates
transformed_x, transformed_y = path_handeling_utilities_obj.rototranslate_path_2_abs_frame(x_path, y_path,xyyaw_ref_path)

# plot a little arrow where the car is
car_length = 0.35
plt.quiver(x_y_yaw_state[0], x_y_yaw_state[1], np.cos(x_y_yaw_state[2])*car_length, np.sin(x_y_yaw_state[2])*car_length, angles='xy', scale_units='xy', scale=1, color='r', width=0.02)




# plot the results
axes_track.plot(x_out, y_out, 'orange')
plt.show()