from MPC_generate_solvers.functions_for_solver_generation import generate_high_level_path_planner_ocp, generate_high_level_MPCC_PP
import numpy as np
import os
import forcespro.nlp
import matplotlib.pyplot as plt
from mpc_node import path_handeling_utilities_class
import optuna
import time

# select the solver to build MPCC or CAMPCC
warm_up_steps = 15
MPC_algorithm = 'CAMPCC' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'
plot_sim = False

# load test track
from functions_for_MPCC_node_running import find_s_of_closest_point_on_global_path
from MPC_generate_solvers.path_track_definitions import generate_path_data

track_choice = 'racetrack_vicon_2' # simpler racetrack
n_checkpoints = 100

s_vals_global_path,\
x_vals_global_path,\
y_vals_global_path,\
s_4_local_path,\
x_4_local_path,\
y_4_local_path,\
dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
k_vals_global_path,\
k_4_local_path,\
heading_4_local_path = generate_path_data(track_choice,n_checkpoints)

# plot track
fig_track, axes_track = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))  # Wider figure for side-by-side layout





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
if MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
    high_level_solver_generator_obj = generate_high_level_path_planner_ocp(MPC_algorithm)
elif MPC_algorithm == 'MPCC_PP':
    high_level_solver_generator_obj = generate_high_level_MPCC_PP()

# dt_controller_rate
dt_controller_rate = high_level_solver_generator_obj.time_horizon/high_level_solver_generator_obj.N


# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_solver = os.path.join(current_script_dir,'MPC_generate_solvers/solvers',high_level_solver_generator_obj.solver_name_forces)

# load pre-built solver
high_level_solver_forces = forcespro.nlp.Solver.from_directory(path_to_solver)



# ---- TESTING THE SOLVER by calling it with a test scenario ----
V_target = 2
local_path_length = V_target * high_level_solver_generator_obj.time_horizon * 1.5 # this is how it would be evalauted in the mpc
lane_width = 0.6

# q_con = 0.1
# q_lag = 5 #0.5 
# q_u_yaw_rate = 0.03 #0.005 
# qt_pos_high = 1
# qt_rot_high = 1
# qt_s_high = 12
# q_sdot = 0.03

# set up fixed parameters
Ds_back = 0
Ds_forward = local_path_length
n = high_level_solver_generator_obj.n_points_kernelized 



# set up initial position
# x y yaw s ref_x ref_y ref_heading  (MPCC and CAMPPC)
# x y yaw s (MPCC_PP)



estimated_ds = V_target * dt_controller_rate  # esitmated ds from previous time instant (velocity is measured now so not accounting for acceleration, but this is only for the search of the s initial s value, so no need to be accurate)
# define plotting limits
lim_x = [np.min(x_4_local_path) - 1, np.max(x_4_local_path) + 1]
lim_y = [np.min(y_4_local_path) - 1, np.max(y_4_local_path) + 1]
                        
solve_time_max = 0.05 # seconds to close the loop
overtime_penalty_coeff = 1

# -------------------------------- simualtion loop --------------------------------
def objective(trial):
    start_time = time.time()
    if MPC_algorithm == 'MPCC_PP':
        q_sdot = trial.suggest_float("q_sdot", 0.001, 0.1, log=True)

    if MPC_algorithm == 'MPCC' or MPC_algorithm == 'MPCC_PP':
        q_lag = trial.suggest_float("q_lag", 0.001, 50, log=True) #0.5
    elif MPC_algorithm == 'CAMPCC':
        q_lag = 0.5 # it is acually not used in the CAMPPC

    # universal parameters
    q_con = trial.suggest_float("q_con", 0.001, 1, log=True)
    q_u_yaw_rate = trial.suggest_float("q_u_yaw_rate", 0.001, 1, log=True) #0.03 #0.005 
    qt_pos_high = trial.suggest_float("qt_pos_high", 0.01, 10, log=True)
    qt_s_high = trial.suggest_float("qt_s_high", 0.01, 20, log=True)

    # fixed parameters
    qt_rot_high = 1 #trial.suggest_float("qt_rot_high", 0.001, 10, log=True) This is really not needed but ok

    # assign initial position as starting on the path at a certain s_0
    s_0 = 0
    current_path_index_on_4_local_path = np.argmin(np.abs(s_4_local_path - s_0))
    x_0 = x_4_local_path[current_path_index_on_4_local_path] + 0.05
    y_0 = y_4_local_path[current_path_index_on_4_local_path] + 0.01
    yaw_0 = heading_4_local_path[current_path_index_on_4_local_path]
    x_y_yaw_state = np.array([x_0,y_0,yaw_0])

    # find the closest point on the global path (a bit redundant but ok)
    previous_path_index_global = np.argmin(np.abs(s_vals_global_path - s_0))




    s_history = [s_0]
    x_history = [x_0]
    y_history = [y_0]
    solver_time_history = []

    if plot_sim:
        plt.ion()
    t = 1
    ds_jump = 1 # initialize to a positive value
    while ds_jump > -1:
        # take time now

        # find the closest point on the path

        # find the closest point on the global path (i.e. measure s)

        s, previous_path_index_global = find_s_of_closest_point_on_global_path(np.array([x_y_yaw_state[0], x_y_yaw_state[1]]), s_vals_global_path,
                                                                                x_vals_global_path, y_vals_global_path,
                                                                                previous_path_index_global, estimated_ds)

        # determine the closest point on local path position
        current_path_index_on_4_local_path = np.argmin(np.abs(s_4_local_path - s))
        x_path_current = x_4_local_path[current_path_index_on_4_local_path]
        y_path_current = y_4_local_path[current_path_index_on_4_local_path]
        yaw_path_current = heading_4_local_path[current_path_index_on_4_local_path]
        xyyaw_ref_path = np.array([x_path_current,y_path_current,yaw_path_current])

        labels_x,\
        labels_y,\
        labels_heading,\
        labels_k,\
        local_path_length,\
        labels_s = path_handeling_utilities_obj.produce_ylabels_4_local_kernelized_path(s,Ds_back,Ds_forward,xyyaw_ref_path,n)


        pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path = path_handeling_utilities_obj.relative_xyyaw_to_current_path(x_y_yaw_state,s)


        # assign initial position (since all is relative to the current closest point on the path setting all to zero means starting on the path at s = s_0)
        xinit = np.zeros(high_level_solver_generator_obj.nx) # all zeros
        xinit[0] = pos_x_init_rot
        xinit[1] = pos_y_init_rot
        xinit[2] = yaw_init_rot


        # set up parameters
        if MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
            params_i = np.array([V_target, local_path_length, q_con, q_lag, q_u_yaw_rate, qt_pos_high, qt_rot_high,lane_width,qt_s_high,*labels_k])
        elif MPC_algorithm == 'MPCC_PP':
            # modify this later
            params_i = np.array([V_target, local_path_length, q_con, q_lag, q_u_yaw_rate, q_sdot ,qt_pos_high, qt_rot_high,lane_width,qt_s_high,*labels_x, *labels_y, *labels_heading])
            #params_i = np.array([V_target, local_path_length, q_con, q_lag, q_u_yaw_rate, qt_pos_high, qt_rot_high,lane_width,qt_s_high,*labels_k])

        # assign parameters
        param_array = np.zeros((high_level_solver_generator_obj.N+1, high_level_solver_generator_obj.n_parameters))
        for i in range(high_level_solver_generator_obj.N+1):
            param_array[i,:] = params_i
        param_array = param_array.ravel() # unpack row-wise


        # set up initial guess
        if MPC_algorithm == 'MPCC' or MPC_algorithm == 'CAMPCC':
            X0_array_high_level = high_level_solver_generator_obj.produce_X0(V_target,local_path_length,labels_k,labels_s,labels_x,labels_y,labels_heading)
        elif MPC_algorithm == 'MPCC_PP':
            X0_array_high_level = high_level_solver_generator_obj.produce_X0(V_target,local_path_length,labels_x,labels_y,labels_heading)

        X0_array_high_level = X0_array_high_level.ravel() # unpack row-wise

        # produce problem as a dictionary for forces
        # "reinitialize": True  will use the provided x0 instead of the internally sotred solution
        problem = {"x0": X0_array_high_level, "xinit": xinit, "all_parameters": param_array, "reinitialize": False} # all_parameters



        # --- solve the problem ---
        solve_time_start = time.time()
        output, exitflag, info = high_level_solver_forces.solve(problem)
        solve_time_end = time.time()
        output_array_high_level = np.array(list(output.values()))
        
        # extract the x-y path
        if MPC_algorithm == 'MPCC_PP':
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
            yaw_rate_high_level = output_array_high_level[:,0]
            x_out = output_array_high_level[:,2]
            y_out = output_array_high_level[:,3]
            yaw_high_level = output_array_high_level[:,4]
            x_path = output_array_high_level[:,6]
            y_path = output_array_high_level[:,7]
            heading_path = output_array_high_level[:,8]


        # transform the path to global coordinates
        x_out_transformed, y_out_transformed = path_handeling_utilities_obj.rototranslate_path_2_abs_frame(x_out, y_out, xyyaw_ref_path)
        x_path_transformed, y_path_transformed = path_handeling_utilities_obj.rototranslate_path_2_abs_frame(x_path, y_path, xyyaw_ref_path)

        # plot the results
        if plot_sim:
            axes_track.clear()
            axes_track.plot(x_4_local_path, y_4_local_path, 'gray')
            axes_track.axis('equal')
            axes_track.set_title('Global track')
            axes_track.set_xlabel('x [m]')
            axes_track.set_ylabel('y [m]')
            axes_track.plot(x_out_transformed, y_out_transformed)
            axes_track.plot(x_path_transformed, y_path_transformed, 'purple')
            axes_track.scatter(x_y_yaw_state[0], x_y_yaw_state[1], color='red', label='initial position',marker='.',s=50,zorder=20) # initial position
            axes_track.legend()
            # set lims
            axes_track.set_xlim(lim_x)
            axes_track.set_ylim(lim_y)



        # update the state
        if t > warm_up_steps:
            x_y_yaw_state[0] = x_out_transformed[1]
            x_y_yaw_state[1] = y_out_transformed[1]
            x_y_yaw_state[2] = yaw_high_level[1] + xyyaw_ref_path[2]

        # collect state history
        s_history.append(s)
        x_history.append(x_y_yaw_state[0])
        y_history.append(x_y_yaw_state[1])

        # check if lap finished
        ds_jump = s_history[t] - s_history[t-1]
        t += 1 # update t

        solver_time = solve_time_end - solve_time_start
        solver_time_history.append(solver_time)

        if plot_sim:
            plt.pause(0.01)

    # Example loss function (you'd replace this with actual training loss)
    # evaluate mean solver time
    for i in range(len(solver_time_history)):
        if solver_time_history[i] < solve_time_max:
            solver_time_history[i] = 0
        else:
            solver_time_history[i] = overtime_penalty_coeff * (solver_time_history[i]/solve_time_max-1)**2
    solve_time_penalty = np.mean(solver_time_history)

    # check if something wnet wrong by filtering for too small lap times
    if t * dt_controller_rate < s_vals_global_path[-1]/(V_target)*0.5: # this means that the lap time was too small to be credible
        loss = s_vals_global_path[-1]/(V_target) # this value is the time that you would have if you travel exaclty on the centre line
    else:
        loss = t * dt_controller_rate + solve_time_penalty

    

    end_time = time.time()
    loop_time = end_time - start_time  # Calculate the elapsed time
    
    print(f"Execution time: {loop_time:.6f} seconds")
    print(f"solve_time_penalty: {solve_time_penalty:.6f} seconds")
    print('')
    print('')

    
    return loss

# 
study_name = "optuna_study_results" + MPC_algorithm + ".csv"
storage_name = "sqlite:///"+study_name+".db"  # SQLite database file

study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)

study.trials_dataframe().to_csv(study_name)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

plt.show()

if plot_sim:
    plt.ioff()  # Turn off interactive mode

# extract useful points
s_history = s_history[warm_up_steps+1:-1]
x_history = x_history[warm_up_steps+1:-1]
y_history = y_history[warm_up_steps+1:-1]

# add color coded plot of x-y according the s_dot
s_dot_history = np.diff(s_history)/dt_controller_rate

# plot the results
axes_track.clear()
axes_track.plot(x_vals_global_path, y_vals_global_path, 'gray', linewidth=2,lineStyle='--')
axes_track.axis('equal')
axes_track.set_title('Global track')
axes_track.set_xlabel('x [m]')
axes_track.set_ylabel('y [m]')
scatter = axes_track.scatter(x_history[:-1], y_history[:-1], c=s_dot_history, cmap='viridis',vmin=V_target-0.5)
# Add a colorbar to the figure
fig_track.colorbar(scatter, ax=axes_track, label="Speed (s_dot)")


#plot simulation results of full trajectory
# plt.figure()
# plt.plot(s_history[warm_up_steps+1:-1])

plt.show()  # Show final static plot


