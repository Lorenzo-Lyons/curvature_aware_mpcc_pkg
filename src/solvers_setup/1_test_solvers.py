import numpy as np
import os
from solver_manager_classes import MPC_solver_handler
from acados_template import AcadosOcpSolver
import matplotlib.pyplot as plt
import sys
import roslib
pkg_path = roslib.packages.get_pkg_dir('curvature_aware_mpcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))    
from reference_path_handeling_functions import generate_path_data
from mpc_node import path_handeling_utilities_class

# ---- load solver ----
controller_type = "CAMPCC"
software_choice = "forcespro"  # "acados", "forcespro"

solver_handler_obj = MPC_solver_handler(controller_type,software_choice)
solvers_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'solvers')
if software_choice == "acados":
    solver_path = os.path.join( solvers_folder_path,
                                solver_handler_obj.solver_name,
                                solver_handler_obj.solver_name + '.json')

    # load the solver
    ocp = solver_handler_obj.produce_ocp()
    solver = AcadosOcpSolver(ocp, json_file=solver_path, build=False, generate=False)
elif software_choice == "forcespro":
    solver_path = os.path.join(solvers_folder_path, solver_handler_obj.solver_name_forcespro)
    # load the solver
    import forcespro 
    solver_forcespro = forcespro.nlp.Solver.from_directory(solver_path)

test_on_track = True










# create parameters
lane_radius = 0.51
local_path_length = 6
nk = solver_handler_obj.n_points_kernelized

params_i = np.zeros(solver_handler_obj.n_parameters)  # Pre-allocate full param vector

# Cost weights
params_i[0] = 10       # q_sdot: tracking velocity weight
params_i[1] = 1    # q_cont: contouring error weight
params_i[2] = 5        # q_lag: lag error weight 
params_i[3] = 0.1 #0.01    # q_roll: roll_rate weight   
params_i[4] = 0.1 #0.01    # q_pitch: pitch_rate weight
params_i[5] = 0.1 #0.01    # q_yaw: yaw_rate weight
params_i[6] = 0.1 #0.01    # q_thrust: control input cost weight
params_i[7] = 10000      # qt_pos: terminal position weight
params_i[8] = 1        # qt_s: terminal fial path progress weight
params_i[9] = lane_radius   # local_path_length
params_i[10] = local_path_length  # slack variable weight




if test_on_track == False:
    # initial condition
    # xinit = [pos_x pos_y pos_z vx vy vz roll pitch yaw s]
    xinit = np.array([  0,    # pos_x
                        0,    # pos_y
                        0,    # pos_z
                        0,    # vx
                        0,    # vy
                        0,    # vz
                        0,    # roll
                        0,    # pitch
                        0,    # yaw
                        0])   # s (arc length, not used in this case, so set to 0)

    # Labels
    labels_x    = np.linspace(0, local_path_length, nk)
    labels_y    = np.zeros(nk)
    labels_z    = np.zeros(nk)
    labels_dxds = np.ones(nk)
    labels_dyds = np.zeros(nk)
    labels_dzds = np.zeros(nk)

    # skip first guess for now
    labels_s = np.linspace(0, local_path_length, solver_handler_obj.n_points_kernelized)
    #X0 = solver_handler_obj.produce_X0(labels_s,labels_x,labels_y,labels_z)
    X0 = np.zeros((solver_handler_obj.N + 1, solver_handler_obj.nx + solver_handler_obj.nu))  # assuming nx = 10, nu = 4
    # set inputs to be 0.5
    X0[:, 0:solver_handler_obj.nu] = 0.5



else:
    # ---- generate path data ----
    #track_choice = 'second_spline_race_track' ###
    #track_choice = 'spline_circle'
    track_choice = 'vicon_racetrack'
    n_path_checkpoints = 300
    s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
    s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
    dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, \
    gates_coordinates, gates_s, time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, n_path_checkpoints)


    # Generate path lables
    path_handeling_utilities_class_obj = path_handeling_utilities_class()
    # asgn the variables to the calss object
    path_handeling_utilities_class_obj.s_4_local_path = s_4_local_path
    path_handeling_utilities_class_obj.x_4_local_path = x_4_local_path
    path_handeling_utilities_class_obj.y_4_local_path = y_4_local_path
    path_handeling_utilities_class_obj.z_4_local_path = z_4_local_path
    path_handeling_utilities_class_obj.dx_ds = dx_ds
    path_handeling_utilities_class_obj.dy_ds = dy_ds
    path_handeling_utilities_class_obj.dz_ds = dz_ds
    path_handeling_utilities_class_obj.d2x_ds2 = d2x_ds2
    path_handeling_utilities_class_obj.d2y_ds2 = d2y_ds2
    path_handeling_utilities_class_obj.d2z_ds2 = d2z_ds2

    # find closest point on the path
    # set initial conditions for the solver
    start_index = 10
    xyz_state = time_optimal_trajectory_4_warmstart[start_index,4:7]  # [x, y, z]
    estimated_ds = 1 # this is the local area to look into finding closest point on path
    previous_index = 0  # this is the index of the previous closest point on the path
    s, current_path_index, dist_to_centerline, xyz_closest_point_on_ref_path = path_handeling_utilities_class_obj.find_s_of_closest_point_on_global_path_3d( xyz_state, 
                                                    s_vals_global_path, 
                                                    x_vals_global_path, 
                                                    y_vals_global_path, 
                                                    z_vals_global_path,
                                                    previous_index, 
                                                    estimated_ds)




    labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds, labels_d2xds2, labels_d2yds2, labels_d2zds2 = \
        path_handeling_utilities_class_obj.produce_ylabels_4_local_kernelized_path(time_optimal_trajectory_4_warmstart[start_index,13], local_path_length, nk, xyz_closest_point_on_ref_path) 



    # produce the initial guess for the solver
    # X0 has position relative to closest point on the path
    X0 = solver_handler_obj.produce_X0_from_time_optimal_trajectory(time_optimal_trajectory_4_warmstart, start_index, xyz_closest_point_on_ref_path)

    xinit = X0[0,solver_handler_obj.nu:] # this needs to have realtive position



    # plot the 3D trajectory with speed encoding
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # 1. positions and speed ------------------------------------------------------
    x, y, z  = time_optimal_trajectory_4_warmstart[:, 4], time_optimal_trajectory_4_warmstart[:, 5], time_optimal_trajectory_4_warmstart[:, 6]
    speed    = np.linalg.norm(time_optimal_trajectory_4_warmstart[:, 7:9], axis=1)      # no normalisation

    # 2. coloured 3‑D line --------------------------------------------------------
    points   = np.column_stack((x, y, z))
    segments = np.stack([points[:-1], points[1:]], axis=1)     # (N, 2, 3)

    lc = Line3DCollection(
            segments,
            cmap='plasma',                                     # ← requested cmap
            norm=plt.Normalize(vmin=speed.min(), vmax=speed.max()))
    lc.set_array(speed[:-1])                                   # colour per segment
    lc.set_linewidth(3.0)

    # 3. plot ---------------------------------------------------------------------
    fig_3d = plt.figure(figsize=(10, 6))
    ax_3d  = fig_3d.add_subplot(111, projection='3d')

    ax_3d.add_collection(lc)
    ax_3d.plot(x_vals_global_path, y_vals_global_path, z_vals_global_path,
            label='Reference Path', color='gray', lw=2.0)
    # add the path labels for local path
    ax_3d.plot( labels_x + xyz_closest_point_on_ref_path[0],
                labels_y + xyz_closest_point_on_ref_path[1],
                labels_z + xyz_closest_point_on_ref_path[2], label='Local Path', color='orangered', linestyle='--', lw=2.0)
    # add X0 plot
    ax_3d.plot( X0[:, 5] + xyz_closest_point_on_ref_path[0],
                X0[:, 6] + xyz_closest_point_on_ref_path[1],
                X0[:, 7] + xyz_closest_point_on_ref_path[2], label='Initial Guess', color='blue', linestyle='--', lw=5.0)

    cbar = fig_3d.colorbar(lc, ax=ax_3d, pad=0.1, shrink=0.7)
    cbar.set_label('Speed [m/s]')

    ax_3d.set_title("Offline optimal solution 3D Position Trajectory (speed‑encoded)")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.grid(True)






# assemble lables
index_init = 11
# Assign labels into param vector
params_i[index_init : index_init + nk]          = labels_x
params_i[index_init + nk : index_init + 2*nk]   = labels_y
params_i[index_init + 2*nk : index_init + 3*nk] = labels_z
params_i[index_init + 3*nk : index_init + 4*nk] = labels_dxds
params_i[index_init + 4*nk : index_init + 5*nk] = labels_dyds
params_i[index_init + 5*nk : index_init + 6*nk] = labels_dzds
if controller_type == "CAMPCC":
    params_i[index_init + 6*nk : index_init + 7*nk] = labels_d2xds2
    params_i[index_init + 7*nk : index_init + 8*nk] = labels_d2yds2
    params_i[index_init + 8*nk : index_init + 9*nk] = labels_d2zds2




# print runtime parameters info
 
# print initial state xinit with 2 decimals
print("Initial state xinit:")
print(np.round(xinit, 2))
# print out X0 with 2 decimals
print("")
print("Initial guess X0:")
print(np.array2string(np.round(X0, 2), separator=', ', max_line_width=np.inf))







# assign the runtime parameters
if software_choice == "acados":
    # assign frist guess if supplied
    if len(X0) > 0:
        for i in range(solver_handler_obj.N):
            solver.set(i, "u", X0[i, :solver_handler_obj.nu])
            solver.set(i, "x", X0[i, solver_handler_obj.nu:])

        solver.set(solver_handler_obj.N, "x", X0[solver_handler_obj.N, solver_handler_obj.nu:])

    # assign initial state
    solver.set(0, "lbx", xinit)
    solver.set(0, "ubx", xinit)

    # assign parameers
    for i in range(solver_handler_obj.N+1):
        solver.set(i, "p", params_i)

    # ---- solve the problem ----
    #solver.set('step_length', 0.1)  # Set step length for the solver, if needed
    status = solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")


    # Retrieve solver time (in seconds)
    solve_time = solver.get_stats('time_tot')
    print(f"Solve time: {solve_time:.6f} seconds")

    print('---')
    solver.print_statistics()

    # ---- extract the solution ----
    state_traj = np.zeros((solver_handler_obj.N + 1, solver_handler_obj.nx))  # assuming 10 states
    input_traj = np.zeros((solver_handler_obj.N, solver_handler_obj.nu))       # assuming 5 inputs

    for i in range(solver_handler_obj.N + 1):
        state_traj[i, :] = solver.get(i, "x")

        if i < solver_handler_obj.N:
            input_traj[i, :] = solver_handler_obj.denormalize_u(solver.get(i, "u"))

elif software_choice == "forcespro":

    # produce problem as a dictionary for forces
    param_array = np.tile(params_i, (solver_handler_obj.N + 1, 1)).ravel()

    problem = {"x0": X0, "xinit": xinit, "all_parameters": param_array} # all_parameters 

    # --- solve the problem ---
    output, exitflag, info = solver_forcespro.solve(problem)
    output_array = np.array(list(output.values()))

    state_traj = output_array[:,solver_handler_obj.nu :]
    # denormalize input trajectory and extract
    input_traj = np.zeros((solver_handler_obj.N, solver_handler_obj.nu))
    for i in range(solver_handler_obj.N):
        input_traj[i, :] = solver_handler_obj.denormalize_u(output_array[i, :solver_handler_obj.nu])






# ---- plot the state solution ----
import numpy as np
import matplotlib.pyplot as plt

# Properly grouped state labels
grouped_labels = [
    ["x", "y", "z"],         # Position
    ["vx", "vy", "vz"],      # Velocity
    ["roll", "pitch", "yaw"],# Orientation
    ["s", None, None]        # Other
]

# Corresponding indices in state_traj (original ordering!)
label_to_index = {
    "x": 0, "vx": 3, "roll": 6, "s": 9,
    "y": 1, "vy": 4, "pitch": 7,
    "z": 2, "vz": 5, "yaw": 8
}

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle("Solver Output Trajectories (States)", fontsize=16)

for col in range(4):
    for row in range(3):
        ax = axes[row, col]
        label = grouped_labels[col][row]

        if label is None:
            if not (row == 1 and col == 3):  # keep axes[1, 3] for custom plot
                ax.axis("off")
            continue

        idx = label_to_index[label]

        ax.plot(state_traj[:, idx], label='Solver Output')
        ax.plot(X0[:, idx + solver_handler_obj.nu], color='k', linestyle='--', label='Initial Guess')
        ax.set_title(label)
        ax.set_xlabel("Time step")
        ax.grid(True)

        if label == "yaw":
            ax.set_ylim([-np.pi, np.pi])

        if col == 0 and row == 0:
            ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])




# add an extra plot of velocity norm Vs s_dot
# Calculate velocity norm
velocity_norm = np.linalg.norm(state_traj[:, 3:6], axis=1)
# calculate s_dot
#s_dot = np.gradient(state_traj[:, 9], solver_handler_obj.time_horizon/solver_handler_obj.N)  # assuming dt is the time step size




# --- evaluate s_dot just how solver would  ----
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
    
        s_dot_as_solver[i] = solver_handler_obj.s_dot_CAMPCC(pos_x, pos_y, pos_z,vx,vy,vz,ref_x, ref_y, ref_z, ref_d2x_ds2,ref_d2y_ds2,ref_d2z_ds2)  # path progress rate is the speed of the drone ref_dxds, ref_dyds, ref_dzds,
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
