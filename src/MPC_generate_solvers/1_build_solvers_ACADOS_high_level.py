import numpy as np
import os
from acados_template import AcadosOcpSolver
from functions_for_solver_generation import generate_high_level_path_planner_ocp


# choose what solver you want to build
# MPCC is the standard baseline, CAMPCC is our new version  

# select the solver to build MPCC or CAMPCC
MPC_algorithm = 'MPCC' # 'MPCC' or 'CAMPCC'

# instantiate the class
ocp_maker_obj = generate_high_level_path_planner_ocp(MPC_algorithm)



# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers = os.path.join(current_script_dir,'solvers/' + ocp_maker_obj.solver_name)
if not os.path.exists(path_to_built_solvers):
    os.makedirs(path_to_built_solvers)  # Creates the folder and any necessary parent folders
os.chdir(path_to_built_solvers)



print('_________________________________________________')
print('Building solver ', ocp_maker_obj.solver_name)
print

ocp = ocp_maker_obj.produce_ocp()
solver = AcadosOcpSolver(ocp, json_file=ocp_maker_obj.solver_name + '.json') # this will regenerate the solver
#solver = AcadosOcpSolver(ocp, json_file=ocp_maker_obj.solver_name + '.json', build=False, generate=False) # this will use a previously compiled solver

print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

print('------------------------------')
print('All done with solver building!')
print('------------------------------')




# TESTING THE SOLVER by calling it with test values


V_target = 3
local_path_length = V_target * ocp_maker_obj.time_horizon * 1.2 # this is how it would be evalauted in the mpc
q_con_high = 1 
q_lag_high = 1 
q_u_high = 0.01 #0.005 
qt_pos_high = 5 
qt_rot_high = 5
lane_width = 0.5

R = 0.75
k = -1/R # going right
print('k =', k)
labels_k_params = -1/R * np.ones(ocp_maker_obj.n_points_kernelized) # -1.5
labels_k_params[:10] = 0 # go straight at the beginning

# put all paratemters together
params_i = np.array([V_target, local_path_length, q_con_high,q_lag_high,q_u_high,qt_pos_high, qt_rot_high,lane_width,*labels_k_params])
for i in range(ocp_maker_obj.N+1):
    solver.set(i, "p", params_i)


# set initial condition
                # x y yaw s ref_x ref_y ref_heading 

xinit = np.zeros(ocp_maker_obj.nx) # all zeros
xinit[0] = 0
xinit[1] = 0.2
xinit[2] = np.pi/4 # np.pi/4 # assign initial yaw of 45 degrees
xinit[6] = 0

solver.set(0, "lbx", xinit)
solver.set(0, "ubx", xinit)


# produce reference x y and yaw fro the path to use as a first guess
X0_array = ocp_maker_obj.produce_X0(V_target,local_path_length,labels_k_params)

# assign frist guess
for i in range(ocp_maker_obj.N):
    solver.set(i, "u", X0_array[i,:ocp_maker_obj.nu])
    solver.set(i, "x", X0_array[i, ocp_maker_obj.nu:])
solver.set(ocp_maker_obj.N, "x", X0_array[ocp_maker_obj.N, ocp_maker_obj.nu:])



# solve the problem
status = solver.solve()

if status != 0:
    print(f"Solver failed with status {status}")
else:
    print("Solver converged successfully!")
solver.print_statistics()
total_time = solver.get_stats('time_tot')
print(f"Total solver time: {total_time:.6f} seconds")



# retrieve solution like mpc node would do it
output_array_high_level = np.zeros((ocp_maker_obj.N+1, ocp_maker_obj.nu + ocp_maker_obj.nx))
for i in range(ocp_maker_obj.N+1):
    if i == ocp_maker_obj.N:
        u_i_solution = np.array([0.0,0.0])
    else:
        u_i_solution = solver.get(i, "u")
    x_i_solution = solver.get(i, "x")
    output_array_high_level[i] = np.concatenate((u_i_solution, x_i_solution))





# now change the initial condition and solve again a number of times to get some statistics out
n_tries = 10 #10000
solver_time_vec = np.zeros(n_tries)
# set random seed
np.random.seed(0)
for trial in range(n_tries):
    # set initial condition
    xinit = np.zeros(ocp_maker_obj.nx) # all zeros

    # pick random initial condition on y and yaw
    xinit[1] =  np.random.uniform(-0.2,0.2)
    xinit[2] = np.random.uniform(-np.pi/4,np.pi/4)
    solver.set(0, "lbx", xinit)
    solver.set(0, "ubx", xinit)

    # assign frist guess
    for i in range(ocp_maker_obj.N):
        solver.set(i, "u", X0_array[i,:ocp_maker_obj.nu])
        solver.set(i, "x", X0_array[i, ocp_maker_obj.nu:])
    solver.set(ocp_maker_obj.N, "x", X0_array[ocp_maker_obj.N, ocp_maker_obj.nu:])

    # solve problem ACADOS_SILENT=ON
    status = solver.solve()

    if status != 0:
        print(f"Solver failed with status {status}")
    else:
        pass

    #solver.print_statistics()
    total_time = solver.get_stats('time_tot')
    solver_time_vec[trial] = total_time
    #print(f"Total solver time: {total_time:.6f} seconds")







import matplotlib.pyplot as plt
plt.figure()
plt.plot(solver_time_vec, label='Solver time')
#plot a dashed line at 0.1s comp time
plt.plot(np.ones(n_tries)*0.01, label='0.01s comp time', linestyle='--')
plt.title('Solver time for different initial conditions')
plt.xlabel('Iteration')
plt.ylabel('Solver time (s)')
plt.grid(True)
plt.legend()












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
dt = 1.5 / ocp_maker_obj.N
axes[1,1].plot(np.diff(output_array_high_level[:, 5])/dt, color='darkgreen', marker='o',label='s dot')  # Add color and marker to plot()
# plot first guess
axes[1,1].plot(V_target * np.ones(ocp_maker_obj.N), color='gray', linestyle='--',label = 'V target')
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






