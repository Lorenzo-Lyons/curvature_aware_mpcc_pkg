import os
import optuna

import rospy
from std_msgs.msg import Float32
from dynamic_reconfigure.client import Client
from solver_manager_classes import MPC_solver_handler
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# select algorithm to tune
MPC_algorithms = ['MPCCPP','CAMPCC'] # 'MPCC' - 'CAMPCC' - 'MPCCPP'
time_horizon_vec = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # start from longer horizons to shorter ones
software = 'forcespro'  # 'acados' or 'forcespro'

optuna_studies_folder = 'optuna_studies'
track = "analytic_circle"
if track == "analytic_circle":
    offline_opt_time = 4 # [s]

# Define figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.ion()
plt.show()


for time_horizon in time_horizon_vec:
    for controller_type in MPC_algorithms:
        
        # define MPC algorithm and time horizon in the GUI
        MPC_solver_handler_obj = MPC_solver_handler(controller_type,time_horizon,software)

        # define study name
        study_name = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track)
        storage_name = os.path.join("sqlite:///", study_name + ".db")

        study = optuna.load_study(study_name=study_name, storage=storage_name)

        if controller_type == 'MPCCPP':
            # Extract the completed trials
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value < offline_opt_time*2]

            # Get parameter values
            qt_pos = [t.params["qt_pos"] for t in trials if "qt_pos" in t.params]
            qt_s = [t.params["qt_s"] for t in trials if "qt_s" in t.params]
            values = [t.value for t in trials]  # optional, if you want to color by objective value

            # Scatter plot on the defined axis
            sc = ax.scatter(qt_pos, qt_s, c=values, cmap="viridis", edgecolor="k")

            # Add colorbar and labels
            if 'cbar' in locals():
                cbar.remove()
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Objective Value")

            ax.set_xlabel(r"$q^t_{pos}$")
            ax.set_ylabel(r"$q^t_s$")
            ax.set_title("Optuna Trials: qt_pos vs qt_s")
            ax.grid(True)
            ax.set_xlim(0, 100*1.1)
            ax.set_ylim(0, 100*1.1)
            plt.pause(0.5) 


plt.ioff()
plt.show()











MPC_algorithm = 'MPCCPP' # 'MPCC' - 'CAMPCC' - 'MPCCPP'
# load the study from the database


# save study
study_name = optuna_studies_folder + "/optuna_study_results_ROS_" + MPC_algorithm
storage_name = "sqlite:///" + study_name + ".db"  # SQLite database file


print('loading GUI parameters from: ', study_name)



study = optuna.load_study(study_name=study_name, storage=storage_name)



# Print best parameters found so far
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# visualize the optimization history
#optuna.visualization.plot_optimization_history(study).show()
#optuna.visualization.plot_param_importances(study).show()



# -------------------------------------------
# Assign best parameters to the GUI
if MPC_algorithm == 'MPCC':
    algorithm_number = 0
elif MPC_algorithm == 'CAMPCC':
    algorithm_number = 1
elif MPC_algorithm == 'MPCCPP':
    algorithm_number = 2

# set GUI params
if assign_to_GUI:
    GUI_mpc_node.update_configuration({"controller_type": algorithm_number})

params_2_load = study.best_params

if assign_to_GUI:
    for key, value in params_2_load.items():
        GUI_mpc_node.update_configuration({key: value})









# from optuna.trial import TrialState
# from optuna.study import StudyDirection

# # ... your existing load_study code above ...

# # Find 2nd-best completed trial
# trials = [t for t in study.get_trials(deepcopy=False)
#           if t.state == TrialState.COMPLETE and t.value is not None]

# if len(trials) < 2:
#     raise RuntimeError(f"Need at least 2 completed trials; got {len(trials)}.")

# # Handle direction (maximize/minimize)
# direction = study.direction if hasattr(study, "direction") else study.directions[0]
# reverse = (direction == StudyDirection.MAXIMIZE)  # sort best->worst if maximizing

# trials_sorted = sorted(trials, key=lambda t: t.value, reverse=reverse)
# second = trials_sorted[1]  # 2nd best

# print(f"2nd best trial: #{second.number}, value={second.value}")
# print("2nd best params:", second.params)

# # -------------------------------------------
# # Assign 2nd-best parameters to the GUI
# if MPC_algorithm == 'MPCC':
#     algorithm_number = 0
# elif MPC_algorithm == 'CAMPCC':
#     algorithm_number = 1
# elif MPC_algorithm == 'MPCCPP':
#     algorithm_number = 2
# else:
#     algorithm_number = 0

# if assign_to_GUI:
#     GUI_mpc_node.update_configuration({"controller_type": algorithm_number})
#     for key, value in second.params.items():
#         GUI_mpc_node.update_configuration({key: value})

# # If you also want to keep a reference:
# params_2_load = second.params






