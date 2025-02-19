import rospy
from dynamic_reconfigure.client import Client
import optuna
import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32


# select algorithm to tune
MPC_algorithm = 'CAMPCC' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'

rospy.init_node("optuna_node")  # Initialize the node

# Create a client to communicate with the reconfigurable node
GUI_client_simulator = Client("/dart_simulator_node", timeout=5)  # Change "your_node_name" to the correct node name
GUI_mpc_node = Client("/mpc_node", timeout=5)  # Change "your_node_name" to the correct node name

max_laps = 1




def reset_initial_position(GUI_client_simulator):
    GUI_client_simulator.update_configuration({"reset_state": True})
    GUI_client_simulator.update_configuration({"reset_state_x": -1.6})
    GUI_client_simulator.update_configuration({"reset_state_y": -2.2})
    GUI_client_simulator.update_configuration({"reset_state_theta": 0})

def set_mpc_node_GUI(trial,GUI_mpc_node):
    # generate random parameters
    if MPC_algorithm == 'MPCC_PP':
        q_sdot = trial.suggest_float("q_sdot", 0.001, 0.1, log=True)
        GUI_mpc_node.update_configuration({"q_sdot": q_sdot})

    if MPC_algorithm == 'MPCC' or MPC_algorithm == 'MPCC_PP':
        q_lag = trial.suggest_float("q_lag", 0.001, 50, log=True) #0.5
        GUI_mpc_node.update_configuration({"q_lag": q_lag})

    elif MPC_algorithm == 'CAMPCC':
        q_lag = 0.5 # it is acually not used in the CAMPPC
        GUI_mpc_node.update_configuration({"q_lag": q_lag})

    # universal parameters
    q_con = trial.suggest_float("q_con", 0.001, 1, log=True)
    q_u_yaw_rate = trial.suggest_float("q_u_yaw_rate", 0.001, 1, log=True) #0.03 #0.005 
    qt_pos_high = trial.suggest_float("qt_pos_high", 0.01, 10, log=True)
    qt_s_high = trial.suggest_float("qt_s_high", 0.01, 20, log=True)

    #read config
    #config_mpc = GUI_mpc_node.get_configuration()
    #cange parameters of interest
    if MPC_algorithm == 'MPCC':
        algorithm_number = 0
    elif MPC_algorithm == 'CAMPCC':
        algorithm_number = 1
    elif MPC_algorithm == 'MPCC_PP':
        algorithm_number = 2

    # set GUI params
    GUI_mpc_node.update_configuration({"MPC_algorithm": algorithm_number})
    GUI_mpc_node.update_configuration({"q_con": q_con})
    GUI_mpc_node.update_configuration({"q_u_yaw_rate": q_u_yaw_rate})
    GUI_mpc_node.update_configuration({"qt_pos_high": qt_pos_high})
    GUI_mpc_node.update_configuration({"qt_s_high": qt_s_high})

    # make sure velocity is 2.5
    GUI_mpc_node.update_configuration({"V_target": 2.5})


    

    


config_simulator = GUI_client_simulator.get_configuration()
#print(config_simulator)  # Print all available parameters
config_mpc = GUI_mpc_node.get_configuration()
#print(config_mpc)








# -------------------------------- simualtion loop --------------------------------
def objective(trial):
    # reset initial position to before the start of the track
    reset_initial_position(GUI_client_simulator)
    set_mpc_node_GUI(trial,GUI_mpc_node)

    start_time = time.time()
    interrupt = False
    s_1_prev = rospy.wait_for_message("/s_1", Float32)
    lap_count = 0
    while lap_count <= max_laps:
        # read most recent message from s_1 topic
        s_1_now = rospy.wait_for_message("/s_1", Float32)
        # chek if the lap was completed
        if s_1_now.data < s_1_prev.data:
            lap_count += 1
            print("Lap completed: ", lap_count-1)
        # update the previous value
        s_1_prev = s_1_now

    return time.time() - start_time





# save study
study_name = "optuna_study_results_ROS_" + MPC_algorithm + ".csv"
storage_name = "sqlite:///"+study_name+".db"  # SQLite database file

study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)

study.trials_dataframe().to_csv(study_name)

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

plt.show()