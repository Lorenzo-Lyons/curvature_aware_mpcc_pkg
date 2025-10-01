import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import roslib
pkg_path = roslib.packages.get_pkg_dir('curvature_aware_mpcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))    

from mpc_node import MPCC_controller_class
import rospy

rospy.init_node('mpc_testing_node', anonymous=False)

# set testin gparaemters
n_solves = 1000 # number of solves to perform in this test (will be equally spaced along the track)





# ---- load solver ----
software_choice = "forcespro"  # "acados", "forcespro"
controller_types = ["CAMPCC"] # "CAMPCC", "MPCCPP", "MPCC", , "CAMPCC_EA", "CAMPCC_EA_2"






drone_controller_obj = MPCC_controller_class()




# set up internal variables for the parameters (taken from optimal optuna gains)

# drone_controller_obj.q_cont = 0.1
# drone_controller_obj.q_lag = 5              
# drone_controller_obj.q_roll_pitch = 0.1
# drone_controller_obj.q_sdot = 10 #2.1060237541752933
# drone_controller_obj.q_thrust = 0.1
# drone_controller_obj.q_yaw = 0.1                
# drone_controller_obj.qt_pos = 0 
# drone_controller_obj.qt_s = 0
# drone_controller_obj.qt_v = 0
# drone_controller_obj.lane_radius = 0.51
# drone_controller_obj.local_path_length = 6

drone_controller_obj.q_cont = 0.0
drone_controller_obj.q_lag = 5              
drone_controller_obj.q_roll_pitch = 10
drone_controller_obj.q_sdot = 2.1060237541752933
drone_controller_obj.q_thrust = 0.0
drone_controller_obj.q_yaw = 7               
drone_controller_obj.qt_pos = 100
drone_controller_obj.qt_s = 0
drone_controller_obj.qt_v = 50
drone_controller_obj.lane_radius = 0.51
drone_controller_obj.local_path_length = 6





s_vec_tests = np.linspace(0.05, drone_controller_obj.s_vals_global_path[-1], n_solves)

# Storage dictionaries per controller type
open_loop_solutions = {}
data_by_controller = {}
convergence_rates = {}

for controller_type in controller_types:
    # Create per-controller storage
    data = np.zeros((n_solves, 4))  # [s, computation time, delta s,distance_to_centerline]
    data[:, 0] = s_vec_tests
    solutions = []
    solver_fails = 0
    for i in range(n_solves):
        # Get current s value
        s = s_vec_tests[i]

        # Set initial state by interpolating the optimal path
        drone_controller_obj.state = drone_controller_obj.interpolate_array_given_s(
            s,
            drone_controller_obj.time_optimal_trajectory_4_warmstart[:, 13],
            drone_controller_obj.time_optimal_trajectory_4_warmstart[:, 4:13]
        )

        # Solve the MPC problem
        start_time = rospy.get_time()
        drone_controller_obj.run_one_mpc_control_loop(controller_type, software_choice)
        stop_time = rospy.get_time()

        solve_time = stop_time - start_time
        compt_time = solve_time

        # save if converged
        if not drone_controller_obj.solver_handler_obj.converged:
            solver_fails += 1
        # Save open-loop solution
        solutions.append(drone_controller_obj.state_traj.copy())

        # Compute distance along path to final predicted state
        xyz_final = drone_controller_obj.state_traj[-1, 0:3]
        estimated_ds = 5

        s_final_state, current_path_index, dist_to_centerline, xyz_closest_point = (
            drone_controller_obj.find_s_of_closest_point_on_global_path_3d(
                xyz_final,
                drone_controller_obj.s_vals_global_path,
                drone_controller_obj.x_vals_global_path,
                drone_controller_obj.y_vals_global_path,
                drone_controller_obj.z_vals_global_path,
                drone_controller_obj.previous_index,
                estimated_ds
            )
        )

        delta_s = s_final_state - s
        if delta_s < 0:
            delta_s += drone_controller_obj.s_vals_global_path[-1]  # handle wrap-around

        # evaluate distance from centerline of the open loop trajectory
        dist_2_centerline_open_loop = np.zeros(drone_controller_obj.state_traj.shape[0])
        for n in range(drone_controller_obj.state_traj.shape[0]):
            pos_x = drone_controller_obj.state_traj[-1, 0]
            pos_y = drone_controller_obj.state_traj[-1, 1]
            pos_z = drone_controller_obj.state_traj[-1, 2]

            # find the closest point on the global path
            _, _, dist_to_centerline, _ = (
                drone_controller_obj.find_s_of_closest_point_on_global_path_3d(
                    [pos_x, pos_y, pos_z],
                    drone_controller_obj.s_vals_global_path,
                    drone_controller_obj.x_vals_global_path,
                    drone_controller_obj.y_vals_global_path,
                    drone_controller_obj.z_vals_global_path,
                    drone_controller_obj.previous_index,
                    estimated_ds
                )
            )

            # store the distance to centerline in the state trajectory
            dist_2_centerline_open_loop[n] = dist_to_centerline
        


        # Store computation time and delta s
        data[i, 1:] = [compt_time, delta_s,np.max(dist_2_centerline_open_loop)]

    # Save results per controller
    open_loop_solutions[controller_type] = solutions
    data_by_controller[controller_type] = data
    convergence_rates[controller_type] = 1 - solver_fails / n_solves





# ----- convergence rates -----
print('')
print('')
print('Convergence rates:')
for controller_type in controller_types:
    rate = convergence_rates[controller_type]
    print(f"{controller_type}: {rate * 100:.2f}%")



# ----- plotting -----
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Extract positions for the optimal trajectory
x_opt = drone_controller_obj.time_optimal_trajectory_4_warmstart[:, 4]
y_opt = drone_controller_obj.time_optimal_trajectory_4_warmstart[:, 5]
z_opt = drone_controller_obj.time_optimal_trajectory_4_warmstart[:, 6]


# Define a color per controller
controller_colors = {
    "MPCC": "peru",
    "CAMPCC": "dodgerblue",
    "MPCCPP": "orangered",
    "CAMPCC_EA": "limegreen"
}

# Create 3D plot
fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plot the optimal trajectory
ax_3d.plot3D(
    x_opt, y_opt, z_opt,
    label='Optimal Trajectory',
    color=np.array([0, 204, 204]) / 256,
    linewidth=1,
    alpha=0.5
)

# Plot the reference path
ax_3d.plot3D(
    drone_controller_obj.x_vals_global_path,
    drone_controller_obj.y_vals_global_path,
    drone_controller_obj.z_vals_global_path,
    label='Reference Path',
    color='gray',
    linewidth=2.0
)

# Plot open-loop solutions for each controller
for controller_type, solutions in open_loop_solutions.items():
    color = controller_colors.get(controller_type, 'black')
    for i, solution in enumerate(solutions):
        x_sol = solution[:, 0]
        y_sol = solution[:, 1]
        z_sol = solution[:, 2]

        label = controller_type if i == 0 else None  # only label the first trajectory per type

        ax_3d.plot3D(
            x_sol, y_sol, z_sol,
            label=label,
            linestyle='-',
            alpha=0.25,
            linewidth=1.2,
            color=color
        )

# Labels and formatting
ax_3d.set_title("Open-loop Trajectories from Different Controllers")
ax_3d.set_xlabel("X [m]")
ax_3d.set_ylabel("Y [m]")
ax_3d.set_zlabel("Z [m]")
ax_3d.grid(True)
ax_3d.legend()
plt.tight_layout()










# ---- plot the state solution ----
import numpy as np
import matplotlib.pyplot as plt

# Properly grouped state labels
grouped_labels = [
    ["comp time"],        
    ["Path progress"],   
    ["max dist from centerline"] 
]

# Corresponding indices in state_traj (original ordering!)
label_to_index = {
    "comp time": 1, "Path progress": 2, "max dist from centerline": 3,
}



# Set up subplots
rows = 3
columns = 1
fig, axes = plt.subplots(rows, columns, figsize=(14, 10))
fig.suptitle("Solver Output Metrics per Controller", fontsize=16)

# Flatten axes array if needed
if rows == 1 or columns == 1:
    axes = axes.flatten()

# Plot each label in its respective subplot
for row in range(rows):
    ax = axes[row]
    label = grouped_labels[row][0]
    idx = label_to_index[label]

    for controller_type in controller_types:
        data = data_by_controller[controller_type]
        color = controller_colors.get(controller_type, 'black')
        ax.plot(data[:, 0], data[:, idx],
                label=controller_type,
                color=color)

    ax.set_title(label)
    ax.set_xlabel("s (m)")
    ax.grid(True)

    if label == "yaw":
        ax.set_ylim([-np.pi, np.pi])

    if row == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


























# add an extra plot of velocity norm Vs s_dot
# Calculate velocity norm
velocity_norm = np.linalg.norm(state_traj[:, 3:6], axis=1)
# calculate s_dot
#s_dot = np.gradient(state_traj[:, 9], solver_handler_obj.time_horizon/solver_handler_obj.N)  # assuming dt is the time step size




# ---- evaluate s_dot just how solver would  ----
# extract labels as solver would do it
if controller_type == "MPCC":
    q_sdot, q_cont, q_lag, q_thrust, q_roll, q_pitch, q_yaw, qt_pos, qt_s, lane_radius,\
    local_path_length, labels_x_as_solver, labels_y_as_solver, labels_z_as_solver, labels_dxds_as_solver, labels_dyds_as_solver, labels_dzds_as_solver =\
    solver_handler_obj.unpack_parameters_MPCC(params_i)
elif controller_type == "CAMPCC":
    q_sdot, q_cont, q_lag, q_thrust, q_roll, q_pitch, q_yaw, qt_pos, qt_s,lane_radius, local_path_length,\
    labels_x_as_solver, labels_y_as_solver, labels_z_as_solver, labels_dxds_as_solver, labels_dyds_as_solver, labels_dzds_as_solver, \
    labels_d2x_ds2_as_solver, labels_d2y_ds2_as_solver, labels_d2z_ds2_as_solver =\
    solver_handler_obj.unpack_parameters_CAMPCC(params_i)



s_dot_as_solver = np.zeros(solver_handler_obj.N + 1)
rex_x_as_solver = np.zeros(solver_handler_obj.N + 1)
rex_y_as_solver = np.zeros(solver_handler_obj.N + 1)
rex_z_as_solver = np.zeros(solver_handler_obj.N + 1)
ref_dxds_as_solver = np.zeros(solver_handler_obj.N + 1)
ref_dyds_as_solver = np.zeros(solver_handler_obj.N + 1)
ref_dzds_as_solver = np.zeros(solver_handler_obj.N + 1)
if controller_type == "CAMPCC":
    ref_d2x_ds2_as_solver = np.zeros(solver_handler_obj.N + 1)
    ref_d2y_ds2_as_solver = np.zeros(solver_handler_obj.N + 1)
    ref_d2z_ds2_as_solver = np.zeros(solver_handler_obj.N + 1)

for i in range(solver_handler_obj.N + 1):
    pos_x = state_traj[i, 0]
    pos_y = state_traj[i, 1]
    pos_z = state_traj[i, 2]
    vx = state_traj[i, 3]
    vy = state_traj[i, 4]
    vz = state_traj[i, 5]

    # define path quantities
    s = state_traj[i, 9]  # arc length
    ref_x = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_x_as_solver)
    ref_y = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_y_as_solver)
    ref_z = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_z_as_solver)
    ref_dxds = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_dxds_as_solver)
    ref_dyds = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_dyds_as_solver)
    ref_dzds = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_dzds_as_solver)

    if solver_handler_obj.controller_type == 'CAMPCC':
        ref_d2x_ds2 = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_d2x_ds2_as_solver)
        ref_d2y_ds2 = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_d2y_ds2_as_solver)
        ref_d2z_ds2 = solver_handler_obj.evaluate_kernelized_line_reg([s], local_path_length, labels_d2z_ds2_as_solver)
    
        s_dot_as_solver[i] = solver_handler_obj.s_dot_CAMPCC(pos_x, pos_y, pos_z,vx,vy,vz,ref_x, ref_y, ref_z,ref_d2x_ds2,ref_d2y_ds2,ref_d2z_ds2,ref_dxds,ref_dyds,ref_dzds)  # path progress rate is the speed of the drone ref_dxds, ref_dyds, ref_dzds,
    elif solver_handler_obj.controller_type == 'MPCC':
        # path progress rate is the speed of the drone
        s_dot_as_solver[i] = vx**2 + vy**2 + vz**2
    
    # assign the values to the arrays for later plotting
    rex_x_as_solver[i] = ref_x
    rex_y_as_solver[i] = ref_y
    rex_z_as_solver[i] = ref_z
    ref_dxds_as_solver[i] = ref_dxds
    ref_dyds_as_solver[i] = ref_dyds
    ref_dzds_as_solver[i] = ref_dzds
    if controller_type == "CAMPCC":
        ref_d2x_ds2_as_solver[i] = ref_d2x_ds2
        ref_d2y_ds2_as_solver[i] = ref_d2y_ds2
        ref_d2z_ds2_as_solver[i] = ref_d2z_ds2
    


# add the s_dot as solver Vs v_norm to the state trajectory
axes[1, 3].plot(s_dot_as_solver, label='s_dot as solver', color='limegreen')
axes[1, 3].plot(velocity_norm, label='Velocity Norm', color='darkgreen')
axes[1, 3].set_title("s_dot Vs Velocity Norm")
axes[1, 3].set_xlabel("Time step")
axes[1, 3].grid(True)
axes[1, 3].legend()






# ---- plot the input solution ----
input_labels = ["roll_desired", "pitch_desired", "yaw_desired", "thrust", "slack"]
fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
fig2.suptitle("Solver Output Trajectories (Inputs)", fontsize=16)

min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds = solver_handler_obj.action_state_bounds()

# De-normalize initial guess
X0_denormalized = np.zeros((solver_handler_obj.N + 1, solver_handler_obj.nu))
for i in range(solver_handler_obj.N + 1):
    X0_denormalized[i, :] = solver_handler_obj.denormalize_u(X0[i, :solver_handler_obj.nu])

# Plot each input
for idx in range(solver_handler_obj.nu):
    ax = axes2.flat[idx]
    ax.plot(input_traj[:solver_handler_obj.N, idx], color='orangered')
    ax.plot(X0_denormalized[:solver_handler_obj.N, idx], color='k', linestyle='--', label='Initial Guess')
    if idx < 4:  # Only plot bounds for the first four inputs
        ax.axhline(y=min_action_bounds[idx], color="grey", linestyle=":", linewidth=1.0, label="Lower bound" if idx == 0 else None)
        ax.axhline(y=max_action_bounds[idx], color="grey", linestyle=":", linewidth=1.0, label="Upper bound" if idx == 0 else None)
        
    ax.set_title(input_labels[idx])
    ax.set_xlabel("Time step")
    ax.grid(True)

# Hide the unused subplot (last one in 3x2 grid)
if solver_handler_obj.nu < len(axes2.flat):
    for j in range(solver_handler_obj.nu, len(axes2.flat)):
        fig2.delaxes(axes2.flat[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])









# plot the 3D trajectory
from mpl_toolkits.mplot3d import Axes3D






# Extract position coordinates
x = state_traj[:, 0]
y = state_traj[:, 1]
z = state_traj[:, 2]

# Create 3D plot
fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.plot(x, y, z, marker='o', linestyle='-', color='b', lw=2.0)
ax_3d.plot(rex_x_as_solver, rex_y_as_solver, rex_z_as_solver, label='Reference Path as solver', color='gray', lw=2.0)
# add intial guess plot+
ax_3d.plot(X0[:, 5], X0[:, 6], X0[:, 7], label='Initial Guess', color='k', linestyle='--', lw=2.0)
# add a single market to show intial state
ax_3d.scatter(xinit[0], xinit[1], xinit[2], color='red', s=100, label='Initial State', marker='x')

ax_3d.quiver(
    rex_x_as_solver,
    rex_y_as_solver,
    rex_z_as_solver,
    ref_dxds_as_solver,
    ref_dyds_as_solver,
    ref_dzds_as_solver,
    length=0.5,  # scale of arrows
    normalize=True,  # arrows will have unit length
    color='darkgoldenrod',
    label='Direction Vectors'
)
# evaluate the norm of the curvature vector
# Calculate curvature magnitude
k_vector = 1 / np.sqrt(
    ref_d2x_ds2_as_solver**2 +
    ref_d2y_ds2_as_solver**2 +
    ref_d2z_ds2_as_solver**2
)

# Scale the vector components
scaled_dx = ref_d2x_ds2_as_solver * k_vector**2
scaled_dy = ref_d2y_ds2_as_solver * k_vector**2
scaled_dz = ref_d2z_ds2_as_solver * k_vector**2

# Plot with scaled arrows
ax_3d.quiver(
    rex_x_as_solver,
    rex_y_as_solver,
    rex_z_as_solver,
    scaled_dx,
    scaled_dy,
    scaled_dz,
    normalize=False,              # use actual vector length
    color='tan',
    arrow_length_ratio=0.1,       # make arrowhead smaller (default is 0.3)
    label='centre of curvature'
)



ax_3d.set_title("What the solver sees (3D Position Trajectory)")
ax_3d.set_xlabel("X [m]")
ax_3d.set_ylabel("Y [m]")
ax_3d.set_zlabel("Z [m]")
ax_3d.grid(True)

# Aspect ratio workaround for older matplotlib versions
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

set_axes_equal(ax_3d)

plt.tight_layout()
plt.show()
