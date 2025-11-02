import os
import optuna

import rospy
from std_msgs.msg import Float32
from dynamic_reconfigure.client import Client
from solver_manager_classes import MPC_solver_handler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



optuna_studies_folder = 'optuna_studies'

# select algorithm to tune
controller_type = 'MPCCPP' # 'MPCC' - 'CAMPCC' - 'MPCCPP'
time_horizon = 0.5 # start from longer horizons to shorter ones  , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
software = 'forcespro'  # 'acados' or 'forcespro'
track = "vicon_racetrack" # "vicon_racetrack", 'analytic_circle'
# training CAMPCC with qt_pos from  MPCCPP?
CAMPCC_qtpos_flag = 2






# define study name
if CAMPCC_qtpos_flag == 0:
    CAMPCC_qt_pos_name_tag = "CAMPCC"
elif CAMPCC_qtpos_flag == 1:
    CAMPCC_qt_pos_name_tag = "CAMPCC_qtpos_from_MPCCPP"
elif CAMPCC_qtpos_flag == 2:
    CAMPCC_qt_pos_name_tag = "CAMPCC_qtpos_tuned"

# define MPC algorithm and time horizon in the GUI
MPC_solver_handler_obj = MPC_solver_handler(controller_type,time_horizon,software)

# define study name and storage
study_name = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track)
if controller_type == 'CAMPCC' and CAMPCC_qt_pos_name_tag != "":
    # replace "CAMPCC" with the tag
    study_name = study_name.replace("CAMPCC", CAMPCC_qt_pos_name_tag)

storage_name = os.path.join("sqlite:///", study_name + ".db")
study = optuna.load_study(study_name=study_name, storage=storage_name)

# Print best parameters found so far
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# visualize the optimization history
#optuna.visualization.plot_optimization_history(study).show()
#optuna.visualization.plot_param_importances(study).show()



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




# -------------------------------------------
# Assign best parameters to the GUI
if controller_type == 'MPCC':
    algorithm_number = 0
elif controller_type == 'CAMPCC':
    algorithm_number = 1
elif controller_type == 'MPCCPP':
    algorithm_number = 2

# set GUI params
if assign_to_GUI:
    GUI_mpc_node.update_configuration({"controller_type": algorithm_number})
    GUI_mpc_node.update_configuration({"time_horizon": MPC_solver_handler_obj.time_horizon})




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






