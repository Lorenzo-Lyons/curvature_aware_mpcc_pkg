import rospy
from dynamic_reconfigure.client import Client
import optuna
import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32
from optuna_study_functions import set_up_GUI_optuna

# change folder to where this script is located
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)








rospy.init_node("optuna_node")  # Initialize the node
pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)


# Create a client to communicate with the reconfigurable node
GUI_client_simulator = Client("/dart_simulator_node", timeout=5)  
GUI_mpc_node = Client("/mpc_node", timeout=5) 

max_laps = 3

# select algorithm to tune
MPC_algorithm = 'CAMPCC' #
single_layer_tag = True
ROS_study = True
set_up_GUI_optuna_obj = set_up_GUI_optuna(GUI_mpc_node,single_layer_tag,MPC_algorithm,ROS_study)


# set dart simulator parameters
# for now using just the dynamic bicycle model # NOTE update this with the SVGP model and the disturbance once I hav ethe new dataset
GUI_client_simulator.update_configuration({"disturbance": False})
GUI_client_simulator.update_configuration({"dynamic_model_choice": 2})




def reset_initial_position(GUI_client_simulator):
    GUI_client_simulator.update_configuration({"reset_state": True})
    GUI_client_simulator.update_configuration({"reset_state_x": -1.6})
    GUI_client_simulator.update_configuration({"reset_state_y": -2.2})
    GUI_client_simulator.update_configuration({"reset_state_theta": 0})
    

def set_mpc_node_GUI(trial,GUI_mpc_node):
    # GUI_mpc_node.update_configuration({"single_layer": True})
    # generate parameters for optuna study
    # #local_path_length,       q_con,      q_u,     q_acc,     qt_pos_high,      qt_rot_high,    lane_width,        qt_s_high,  q_v, labels_k
    q_con = trial.suggest_float("q_con", 0.001, 1, log=True)
    q_v = trial.suggest_float("q_v", 0.001, 0.2, log=True)
    q_u = trial.suggest_float("q_u", 0.001, 0.2, log=True)
    q_acc = trial.suggest_float("q_acc", 0.001, 0.2, log=True)


    # set GUI parameters
    GUI_mpc_node.update_configuration({"q_con": q_con})
    GUI_mpc_node.update_configuration({"q_v": q_v})
    GUI_mpc_node.update_configuration({"q_u": q_u})
    GUI_mpc_node.update_configuration({"q_acc": q_acc})





    


config_simulator = GUI_client_simulator.get_configuration()
#print(config_simulator)  # Print all available parameters
config_mpc = GUI_mpc_node.get_configuration()
#print(config_mpc)









# -------------------------------- simualtion loop --------------------------------
def objective(trial):
    pub_safety_value.publish(0.0)

    # reset initial position to before the start of the track
    reset_initial_position(GUI_client_simulator)
    set_mpc_node_GUI(trial,GUI_mpc_node)

    
    interrupt = False
    s_1_prev = rospy.wait_for_message("/s_1", Float32)
    lap_count = 0
    started_timer = False
    elapsed_time = 0
    start_time_trial = time.time()
    lane_bound_penalty = 0
    
    while lap_count <= max_laps and (time.time() - start_time_trial) < 45: # protect against stalling
        # wait 2 s before activating the controller
        if time.time() - start_time_trial > 2:
            pub_safety_value.publish(1.0)
        else:
            pub_safety_value.publish(0.0)


        # read most recent message from s_1 topic
        s_1_now = rospy.wait_for_message("/s_1", Float32)
        # chek if the lap was completed
        if s_1_now.data < s_1_prev.data and started_timer==False:
            started_timer = True
            start_time = time.time()
            lap_count += 1
            print("Lap completed: ", lap_count-1)
        elif s_1_now.data < s_1_prev.data and started_timer==True:
            lap_count += 1
            print("Lap completed: ", lap_count-1)
        # update the previous value
        s_1_prev = s_1_now
        # print started_timer
        #print("started_timer: ", started_timer)
        if started_timer:
            elapsed_time = time.time() - start_time
            distance_from_centerline_now = rospy.wait_for_message("/distance_from_centerline_1", Float32)
            if distance_from_centerline_now.data > set_up_GUI_optuna_obj.lane_width/2:
                
                lane_bound_penalty += set_up_GUI_optuna_obj.lane_violation_cost * (distance_from_centerline_now.data - set_up_GUI_optuna_obj.lane_width/2)

    # set safety to 0 immediately after the trial is completed
    pub_safety_value.publish(0.0)

    if elapsed_time < 6 * max_laps or distance_from_centerline_now.data - set_up_GUI_optuna_obj.lane_width/2 > set_up_GUI_optuna_obj.lane_width/2: # something went wrong, like exited lane or some strange behavior
        print('Trial aborted due to too short lap time or too large lane violation')
        elapsed_time = 35

    
    print('Elapsed time: ', elapsed_time)
    print('Lane bound penalty: ', lane_bound_penalty)


    return elapsed_time + lane_bound_penalty





# save study
study_name = set_up_GUI_optuna_obj.study_name #"optuna_studies/optuna_results_SINGLE_layer_ROS_" + MPC_algorithm
storage_name = "sqlite:///" + study_name + ".db"  # SQLite database file


from optuna.samplers import TPESampler
# Create a study with Bayesian Optimization (TPE)
study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage_name, load_if_exists=True,sampler=TPESampler())
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)

study.trials_dataframe().to_csv(study_name)


# try reloading study
study = optuna.load_study(study_name=study_name, storage=storage_name)


optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

plt.show()