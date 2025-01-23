import numpy as np
import os
from acados_template import AcadosOcpSolver
from functions_for_solver_generation import generate_low_level_solver_ocp



# choose what solver you want to build
# They differ in how the dynamic constraint and cost function are determined. Standard MPCC has needs lag error
# penalization, thus has an extra term in cost function. It also tracks velocity. In the dynamic constraint it
# uses Runge kutta order 4.
# Large ds uses path progress speed traking In the cost function and doesn't need the lag error. In the dynamic
# constraint it uses runge kutta for the vehicle dynamics and my formulation for the path progress.
# Both formulations use a (single) slack variable for the dynamic constraints.



# select the solver to build MPCC or CAMPCC
dynamic_model = 'dynamic_bicycle' # 'kinematic_bicycle', 'dynamic_bicycle', 'SVGP'

# instantiate the class
ocp_maker_obj = generate_low_level_solver_ocp(dynamic_model)



 


# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers = os.path.join(current_script_dir,'solvers/' + ocp_maker_obj.solver_name_acados)
if not os.path.exists(path_to_built_solvers):
    os.makedirs(path_to_built_solvers)  # Creates the folder and any necessary parent folders
os.chdir(path_to_built_solvers)



print('_________________________________________________')
print('Building solver ', ocp_maker_obj.solver_name_acados)
print
ocp = ocp_maker_obj.produce_ocp()
solver = AcadosOcpSolver(ocp, json_file=ocp_maker_obj.solver_name_acados + '.json') # this will regenerate the solver
#solver = AcadosOcpSolver(ocp, json_file=ocp_maker_obj.solver_name_acados + '.json', build=False, generate=False) # this will use a previously compiled solver

print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

print('------------------------------')
print('All done with solver building!')
print('------------------------------')


# # save solver_choices in the same folder as the solver, This will be used at runtime to give the solver the correct parameters
# file_path = os.path.join(path_to_built_solvers, 'solver_choices.npy')
# np.save(file_path, solver_choices)




# ---- TESTING THE SOLVER by calling it with a test scenario ----

# load the high level solver
from functions_for_solver_generation import generate_high_level_path_planner_ocp
MPC_algorithm = 'CAMPCC' # 'MPCC' or 'CAMPCC'
high_level_ocp_maker_obj = generate_high_level_path_planner_ocp(MPC_algorithm)
ocp = high_level_ocp_maker_obj.produce_ocp()
path_to_high_level_solver = os.path.join(current_script_dir,'solvers/' + high_level_ocp_maker_obj.solver_name_acados) + '/' + high_level_ocp_maker_obj.solver_name_acados
high_level_solver = AcadosOcpSolver(ocp, json_file=path_to_high_level_solver + '.json', build=False, generate=False) # this will use a previously compiled solver



V_target = 1
local_path_length = V_target * high_level_ocp_maker_obj.time_horizon * 1.2 # this is how it would be evalauted in the mpc
q_con_high = 1
q_lag_high = 1
q_u_high = 0.01
qt_pos_high = 5
qt_rot_high = 5
lane_width = 0.1
R = 0.75
labels_k_params = -1/R * np.ones(high_level_ocp_maker_obj.n_points_kernelized) # -1.5
labels_k_params[:10] = 0 # go straight at the beginning




# put all parameters together
params_i = np.array([V_target, local_path_length, q_con_high,q_lag_high,q_u_high,qt_pos_high, qt_rot_high,lane_width,*labels_k_params])
for i in range(high_level_ocp_maker_obj.N+1):
    high_level_solver.set(i, "p", params_i)

# set initial condition
                # x y yaw s ref_x ref_y ref_heading 

xinit_high_level = np.zeros(high_level_ocp_maker_obj.nx) # all zeros
xinit_high_level[0] = 0
xinit_high_level[1] = -0.1
xinit_high_level[2] = np.pi/4 # np.pi/4 # assign initial yaw of 45 degrees
xinit_high_level[6] = 0

high_level_solver.set(0, "lbx", xinit_high_level)
high_level_solver.set(0, "ubx", xinit_high_level)

# produce reference x y and yaw fro the path to use as a first guess
X0_array = high_level_ocp_maker_obj.produce_X0(V_target,local_path_length,labels_k_params)

# assign frist guess
for i in range(high_level_ocp_maker_obj.N):
    high_level_solver.set(i, "u", X0_array[i,:high_level_ocp_maker_obj.nu])
    high_level_solver.set(i, "x", X0_array[i, high_level_ocp_maker_obj.nu:])
high_level_solver.set(high_level_ocp_maker_obj.N, "x", X0_array[high_level_ocp_maker_obj.N, high_level_ocp_maker_obj.nu:])

# solve the problem
status_high_level = high_level_solver.solve()

output_array_high_level = np.zeros((high_level_ocp_maker_obj.N+1, high_level_ocp_maker_obj.nu + high_level_ocp_maker_obj.nx))
for i in range(high_level_solver.N+1):
    if i == high_level_solver.N:
        u_i_solution = np.array([0.0,0.0])
    else:
        u_i_solution = high_level_solver.get(i, "u")
    x_i_solution = high_level_solver.get(i, "x")
    output_array_high_level[i] = np.concatenate((u_i_solution, x_i_solution))



import matplotlib.pyplot as plt
fig = plt.figure()
ax_traj = fig.add_subplot(111)  # Get the current axis
ax_traj.plot(output_array_high_level[:,6],output_array_high_level[:,7], label='high level solver local path',color='violet',marker='o',markersize=2)
ax_traj.plot(output_array_high_level[:,2],output_array_high_level[:,3], label='high level solver trajectory',color='darkgreen',marker='o',markersize=2)
ax_traj.set_title('Solver time for different initial conditions')
ax_traj.set_xlabel('Iteration')
ax_traj.set_ylabel('Solver time (s)')
ax_traj.grid(True)
ax_traj.legend()




# solve the high level problem
print('')
print('')
print(' --- high level solver ---')
high_level_solver.print_statistics()
total_time = high_level_solver.get_stats('time_tot')
print(f"Total high level solver time: {total_time:.6f} seconds")
print(' --------------------------')
print('')
print('')









# --- low level solver ---
#solver.options_set('step_length', 0.02) # use this to reduce the step size

# # set up parameters for each stage
#solver.options_set('step_length',0.5)
N = ocp_maker_obj.N
q_v = 1
q_pos = 1
q_rot = 1
q_u = 0.01
qt_pos = 10
qt_rot = 10
slack_p_1 = 1
q_acc = 0.1


# put all parameters together
# output_array_high_level = u_yaw slack x y yaw s ref_x ref_y ref_heading
for i in range(N+1):
    x_ref = output_array_high_level[i,2]
    y_ref = output_array_high_level[i,3]
    yaw_ref = output_array_high_level[i,4]
    x_path = output_array_high_level[i,6]
    y_path = output_array_high_level[i,7]

    params_i = np.array([V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width])
    solver.set(i, "p", params_i)


# set initial condition
xinit = np.zeros(ocp_maker_obj.nx) # all zeros
            # x y yaw vx vy w
xinit[0] = xinit_high_level[0]
xinit[1] = xinit_high_level[1]
xinit[2] = xinit_high_level[2]
xinit[3] = V_target # setting to v target

solver.set(0, "lbx", xinit)
solver.set(0, "ubx", xinit)

# # assign control input bounds
# for k in range(N):
#         solver.set(k, "lbu", np.array([0,-1,0]))  # Lower bound on u at stage
#         solver.set(k, "ubu", np.array([1,1,100]))



# Initial guess for state trajectory
X0_array = ocp_maker_obj.produce_X0(V_target, output_array_high_level)


# assign frist guess
for i in range(N):
    solver.set(i, "u", X0_array[i,:ocp_maker_obj.nu])
    solver.set(i, "x", X0_array[i, ocp_maker_obj.nu:])
solver.set(N, "x", X0_array[N, ocp_maker_obj.nu:])



# solve the problem
status = solver.solve()

# Retrieve the state trajectory
output_array_low_level = np.zeros((ocp_maker_obj.N+1, ocp_maker_obj.nu + ocp_maker_obj.nx))
for i in range(ocp_maker_obj.N+1):
    if i == ocp_maker_obj.N:
        u_i_solution = np.array([0.0,0.0,0.0])
    else:
        u_i_solution = solver.get(i, "u")

    x_i_solution = solver.get(i, "x")
    output_array_low_level[i] = np.concatenate((u_i_solution, x_i_solution))



print('')
print('')
print(' --- low level solver ---')

if status != 0:
    print(f"Solver failed with status {status}")
else:
    print("Solver converged successfully!")
solver.print_statistics()
total_time = solver.get_stats('time_tot')
print(f"Total solver time: {total_time:.6f} seconds")



# plot trajectory
ax_traj.plot(output_array_low_level[:,3],output_array_low_level[:,4], label='low level solver trajectory',color='blue',marker='o',markersize=2)
ax_traj.legend()




# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))  # Wider figure for side-by-side layout

# Plot throttle 
axes[0, 0].plot(output_array_low_level[:-1, 0], color='r', marker='o')  # Use color and marker in plot()
axes[0, 0].set_title('Throttle')  # Title is plain text
axes[0, 0].set_ylim([0, 1])

# Plot steering 
axes[1, 0].plot(output_array_low_level[:-1, 1], color='r', marker='o')  # Add marker to plot()
axes[1, 0].set_title('Steering')  # Correctly use set_title
axes[1, 0].set_ylim([-1, 1])

# plot slack
axes[2, 0].plot(output_array_low_level[:-1, 2], color='r', marker='o')  # Add marker to plot()
axes[2, 0].set_title('Slack')  # Correctly use set_title
axes[2, 0].set_ylim([0, 1])

# Plot velocity 
axes[0, 1].plot(output_array_low_level[:, 6], color='b', marker='o')  # Add color and marker to plot()
axes[0, 1].set_title('Vx State Trajectory')  # Title as plain text
axes[0, 1].set_ylim([0, 5])

# Plot velocity 
axes[1, 1].plot(output_array_low_level[:, 7], color='b', marker='o')  # Add color and marker to plot()
axes[1, 1].set_title('Vy State Trajectory')  # Title as plain text
axes[1, 1].set_ylim([-1, 1])

# Plot velocity 
axes[2, 1].plot(output_array_low_level[:, 8], color='b', marker='o')  # Add color and marker to plot()
axes[2, 1].set_title('W State Trajectory')  # Title as plain text
axes[2, 1].set_ylim([-5, 5])

plt.show()
