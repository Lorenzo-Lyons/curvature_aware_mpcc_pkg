import numpy as np
import os
from functions_for_solver_generation import generate_high_level_path_planner_ocp
# load pre-built solver
import forcespro.nlp
import matplotlib.pyplot as plt

# select the solver to build MPCC or CAMPCC
MPC_algorithm = 'MPCC' # 'MPCC' or 'CAMPCC'

# instantiate the class
solver_maker_obj = generate_high_level_path_planner_ocp(MPC_algorithm)


# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_solver = os.path.join(current_script_dir,'solvers',solver_maker_obj.solver_name_forces)

# load pre-built solver
high_level_solver_forces = forcespro.nlp.Solver.from_directory(path_to_solver)


# ---- TESTING THE SOLVER by calling it with a test scenario ----

V_target = 3
local_path_length = V_target * solver_maker_obj.time_horizon * 1.2 # this is how it would be evalauted in the mpc
q_con_high = 1 
q_lag_high = 1 
q_u_high = 0.01 #0.005 
qt_pos_high = 5 
qt_rot_high = 5
lane_width = 0.5

R = 0.75
k = -1/R # going right
print('k =', k)
labels_k_params = -1/R * np.ones(solver_maker_obj.n_points_kernelized) # -1.5
labels_k_params[:10] = 0 # go straight at the beginning

# put all paratemters together
params_i = np.array([V_target, local_path_length, q_con_high,q_lag_high,q_u_high,qt_pos_high, qt_rot_high,lane_width,*labels_k_params])


# set up parameters
param_array = np.zeros((solver_maker_obj.N, solver_maker_obj.n_parameters))
# put all parameters together
for i in range(solver_maker_obj.N):
    param_array[i,:] = params_i
param_array = param_array.ravel() # unpack row-wise



# set up initial condition
xinit = np.zeros(solver_maker_obj.nx) # all zeros
xinit[0] = 0
xinit[1] = 0.2
xinit[2] = np.pi/4 # np.pi/4 # assign initial yaw of 45 degrees
xinit[6] = 0

# set up initial guess
X0_array = solver_maker_obj.produce_X0(V_target,local_path_length,labels_k_params)
x0_array = X0_array[:-1,:].ravel() # unpack row-wise

# produce problem as a dictionary for forces
problem = {"x0": x0_array, "xinit": xinit, "all_parameters": param_array} # all_parameters



# --- solve the problem ---
output, exitflag, info = high_level_solver_forces.solve(problem)
output_array_high_level = np.array(list(output.values()))
print('')
print('')

# Print high-level solver results
print(' --- High Level Solver ---')

if exitflag == 1:
    print('Solver converged successfully!')
else:
    print(f'Solver failed with exitflag {exitflag}.')
print(f'Exit flag: {exitflag}')
# Print additional solver information
print('Solver Information:')
#print(f'  Number of iterations: {info.it}')
print(f'  Solve time: {info.solvetime:.6f} seconds')
#print(f'  Objective value: {info.obj_val:.6f}')





# Assuming u_solution and x_solution are already defined as NumPy arrays

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))  # Wider figure for side-by-side layout

# Plot throttle (top-left subplot)
axes[0,0].plot(output_array_high_level[:-1, 0], color='r', marker='o',label='yaw rate input')  # Use color and marker in plot()
axes[0,0].set_title('yaw dot')  # Title is plain text
axes[0,0].plot(X0_array[:-1,0], color='gray',marker='o',markersize=2, label = 'first guess') # skip last value that will not be used
axes[0,0].legend()

# slack activation
axes[1,0].plot(output_array_high_level[:-1, 1], color='darkred', marker='o',label='slack variable')  # Use color and marker in plot()
axes[1,0].set_title('slack')  # Title is plain text
axes[1,0].plot(X0_array[:-1,1], color='gray',marker='o',markersize=2, label = 'first guess') # skip last value that will not be used
axes[1,0].legend()


# plot time evolution of the reference theta
axes[0,1].plot(output_array_high_level[:, 8], color='b', marker='o',label='heading angle')  # Add color and marker to plot()
# plot first guess
axes[0,1].plot(X0_array[:,-1], color='gray',marker='o',markersize=2, label = 'first guess')
axes[0,1].set_title('Reference Theta')  # Title as plain text
axes[0,1].legend()

# add plot of s coordinate over the time
dt = 1.5 / solver_maker_obj.N
axes[1,1].plot(np.diff(output_array_high_level[:, 5])/dt, color='darkgreen', marker='o',label='s dot')  # Add color and marker to plot()
# plot first guess
axes[1,1].plot(V_target * np.ones(solver_maker_obj.N), color='gray', linestyle='--',label = 'V target')
axes[1,1].set_title('s dot coordinate')  # Title as plain text
axes[1,1].legend()
# set x-axis label for the last subplot
axes[1,1].set_xlabel('solver stage')  # Label for x-axis
# set limits on y
axes[1,1].set_ylim([0, V_target + 0.5])





plt.figure()
# plot x y trajectory
plt.plot(output_array_high_level[:, 2], output_array_high_level[:, 3], color='g', marker='o',label = 'trajectory')  # x and y states

# plot the reference path as it comes out of the solver
plt.plot(output_array_high_level[:,-3], output_array_high_level[:,-2], label='Path from solver', color='violet',alpha=1,marker='o',markersize=2)

# plot first guess
plt.plot(X0_array[:,2], X0_array[:,3], color='gray',alpha=0.5,marker='o',markersize=2, label = 'reference first guess')


plt.grid(True)
plt.title('x-y Trajectory')  # Title for the x-y trajectory
plt.xlabel('x')  # Label for x-axis
plt.ylabel('y')  # Label for y-axis
# plt.xlim([0, 4])  # Set x-axis limits
# plt.ylim([-3, 1])  # Set y-axis limits
plt.axis('equal')  # Equal scaling for x and y axes
plt.legend()  # Show legend



# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()