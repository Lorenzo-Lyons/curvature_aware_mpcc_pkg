import os
import optuna

assign_to_GUI = True

ROS_study = True
single_layer = True
MPC_algorithm = 'CAMPCC' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# save study

# optuna_studies/

if ROS_study:
    study_name = "optuna_studies/optuna_study_results_ROS_" + MPC_algorithm
else:
    study_name = "optuna_studies/optuna_study_results_" + MPC_algorithm


storage_name = "sqlite:///" + study_name + ".db"  # SQLite database file

study = optuna.load_study(study_name=study_name, storage=storage_name)

# Print best parameters found so far
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# visualize the optimization history
#optuna.visualization.plot_optimization_history(study).show()
#optuna.visualization.plot_param_importances(study).show()






# -------------------------------------------
# Assign best parameters to the GUI



if assign_to_GUI:
    import rospy
    from dynamic_reconfigure.client import Client
    from std_msgs.msg import Float32

    rospy.init_node("assign_optuna_paramters_node")  # Initialize the node
    pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)

    GUI_mpc_node = Client("/mpc_node", timeout=5)

    # assign the right mpc algorithm
    if MPC_algorithm == 'MPCC':
        GUI_mpc_node.update_configuration({"MPC_algorithm": 0})
    elif MPC_algorithm == 'CAMPCC':
        GUI_mpc_node.update_configuration({"MPC_algorithm": 1})
    elif MPC_algorithm == 'MPCC_PP':
        GUI_mpc_node.update_configuration({"MPC_algorithm": 2})
    
    # assign the right software
    GUI_mpc_node.update_configuration({"Solver_software": 1}) # forces
    # assign low level controller params
    GUI_mpc_node.update_configuration({"Dynamic_model": 1}) # Dynamic bicycle
    # assign velocity target
    GUI_mpc_node.update_configuration({"V_target": 2.5})

    for key, value in study.best_params.items():
        GUI_mpc_node.update_configuration({key: value})

