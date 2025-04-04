import os
import optuna
from optuna_study_functions import set_up_GUI_optuna

import rospy
from std_msgs.msg import Float32
from dynamic_reconfigure.client import Client

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




rospy.init_node("assign_optuna_paramters_node")  # Initialize the node
GUI_mpc_node = Client("/mpc_node", timeout=1)





assign_to_GUI = True


ROS_study = True
single_layer = True
MPC_algorithm = 'CAMPCC' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'

set_up_GUI_optuna_obj = set_up_GUI_optuna(GUI_mpc_node, single_layer, MPC_algorithm, ROS_study)



# save study

# optuna_studies/

# if ROS_study:
#     study_name = "optuna_studies/optuna_study_results_ROS_" + MPC_algorithm
# else:
#     study_name = "optuna_studies/optuna_study_results_" + MPC_algorithm
study_name = set_up_GUI_optuna_obj.study_name
storage_name = set_up_GUI_optuna_obj.storage_name   #"sqlite:///" + study_name + ".db"  # SQLite database file
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



if assign_to_GUI:
    
    pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)


    for key, value in study.best_params.items():
        GUI_mpc_node.update_configuration({key: value})

