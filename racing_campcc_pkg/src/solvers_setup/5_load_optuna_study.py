import os
import optuna

import rospy
from std_msgs.msg import Float32
from dynamic_reconfigure.client import Client

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



optuna_studies_folder = 'optuna_studies_th_07'





assign_to_GUI = True

if assign_to_GUI:
    print('Will assign best parameters to the GUI')
    rospy.init_node("assign_optuna_paramters_node")  # Initialize the node
    GUI_mpc_node = Client("/mpc_node", timeout=1)
    # set up constant parameters
    GUI_mpc_node.update_configuration({"lane_radius": 0.5})
    GUI_mpc_node.update_configuration({"local_path_length": 6})
    GUI_mpc_node.update_configuration({"software_choice": 1}) # 1 for forcespro, 0 for acados

    # set GUI params that will not be tuned
    GUI_mpc_node.update_configuration({"q_sdot": 0.0})
    GUI_mpc_node.update_configuration({"q_cont": 0.0})
    GUI_mpc_node.update_configuration({"q_thrust": 0.1})
    GUI_mpc_node.update_configuration({"q_roll_pitch": 0.1})
    GUI_mpc_node.update_configuration({"q_yaw": 0.1})
    GUI_mpc_node.update_configuration({"q_lag": 100.0})
    #GUI_mpc_node.update_configuration({"qt_s": 100.0})







MPC_algorithm = 'MPCCPP' # 'MPCC' - 'CAMPCC' - 'MPCCPP'
# load the study from the database


# save study
study_name = optuna_studies_folder+"/optuna_study_results_ROS_" + MPC_algorithm
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






