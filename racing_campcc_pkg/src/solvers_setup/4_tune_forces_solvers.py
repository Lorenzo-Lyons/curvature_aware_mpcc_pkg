import rospy
from dynamic_reconfigure.client import Client
import optuna
import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32

# change folder to where this script is located
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


optuna_studies_folder = 'optuna_studies_th_07'

# select algorithm to tune
MPC_algorithm = 'MPCCPP' # 'MPCC' - 'CAMPCC' - 'MPCCPP'


rospy.init_node("optuna_node")  # Initialize the node
pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)


# Create a client to communicate with the reconfigurable node
GUI_client_simulator = Client("/drone_simulator_node", timeout=5)  
GUI_mpc_node = Client("/mpc_node", timeout=5) 

max_laps = 1

n_trials = 10



lane_violation_cost = 10
lane_radius = 0.5

# set up constant parameters
GUI_mpc_node.update_configuration({"lane_radius": lane_radius})
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

# define initial paraemter guess for optuna

initial_guess = {
"qt_s": 0.1,
"qt_pos": 10.0,
}

if MPC_algorithm == "CAMPCC":
    initial_guess = {
    "qt_v": 0.1,
    }
    # load previously found best parameters for MPCCPP as they will not be changed
    load_study_name = optuna_studies_folder+"/optuna_study_results_ROS_" + "MPCCPP"
    storage_name = "sqlite:///" + load_study_name + ".db"  # SQLite database file
    print('loading GUI parameters from: ', load_study_name)
    study = optuna.load_study(study_name=load_study_name, storage=storage_name)
    for key, value in study.best_params.items():
        GUI_mpc_node.update_configuration({key: value})
    






def reset_initial_position(GUI_client_simulator):
    GUI_client_simulator.update_configuration({"reset_state": True})
    GUI_client_simulator.update_configuration({"reset_state_x": -2.88})
    GUI_client_simulator.update_configuration({"reset_state_y": -1.2})
    GUI_client_simulator.update_configuration({"reset_state_z": 2.4})
    GUI_client_simulator.update_configuration({"reset_state_roll": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_pitch": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_yaw": 0.0})

def set_mpc_node_GUI(trial,GUI_mpc_node):
    # generate random parameters
    # q_roll_pitch = trial.suggest_float("q_roll_pitch", 0.01, 1, log=False) 
    # q_yaw = trial.suggest_float("q_yaw", 0.01, 1, log=False)
    #qt_s = trial.suggest_float("qt_s", 0.01, 100, log=True) 

    # algorithm specific parameters

    
    if MPC_algorithm == 'MPCC' or MPC_algorithm == 'MPCCPP':
        qt_s = trial.suggest_float("qt_s", 0.01, 10, log=True) 
        qt_pos = trial.suggest_float("qt_pos", 0.01, 100, log=True) #0.1
        qt_v = 0
        GUI_mpc_node.update_configuration({"qt_pos": qt_pos})
        GUI_mpc_node.update_configuration({"qt_s": qt_s})

    elif MPC_algorithm == 'CAMPCC':
        qt_v = trial.suggest_float("qt_v", 0.01, 2.5, log=False) 
        GUI_mpc_node.update_configuration({"qt_v": qt_v})


    #read config
    #config_mpc = GUI_mpc_node.get_configuration()
    #cange parameters of interest
    if MPC_algorithm == 'MPCC':
        algorithm_number = 0
    elif MPC_algorithm == 'CAMPCC':
        algorithm_number = 1
    elif MPC_algorithm == 'MPCCPP':
        algorithm_number = 2

    # set GUI params
    GUI_mpc_node.update_configuration({"controller_type": algorithm_number})


    



    

config_simulator = GUI_client_simulator.get_configuration()
#print(config_simulator)  # Print all available parameters
config_mpc = GUI_mpc_node.get_configuration()
#print(config_mpc)





# # # load optimal trajectory paraemters
# # import roslib
# # import sys
# # pkg_path = roslib.packages.get_pkg_dir('racing_campcc_pkg')
# # sys.path.append(os.path.join(pkg_path, 'src'))     # or whatever sub‑folder holds the module
# # from reference_path_handeling_functions import generate_path_data

# # track_choice = 'vicon_racetrack'
# # load_optimally_smoothed_path = True

# # s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
# # s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
# # dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, \
# # gates_coordinates, gates_s , time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, load_optimally_smoothed_path)


# # optimal_lap_timee = time_optimal_trajectory_4_warmstart[-1,15]  # last element is the total time of the optimal trajectory
optimal_lap_time = 8.3










# -------------------------------- simualtion loop --------------------------------
def objective(trial):
    pub_safety_value.publish(0.0)

    # reset initial position to before the start of the track
    reset_initial_position(GUI_client_simulator)
    set_mpc_node_GUI(trial,GUI_mpc_node)

    
    s_1_prev = rospy.wait_for_message("/s", Float32)
    lap_count = 0
    started_timer = False
    elapsed_time = 0
    start_time_trial = time.time()
    lane_bound_penalty = 0
    
    while lap_count <= max_laps and (time.time()-start_time_trial) < (optimal_lap_time)*max_laps*3.5: # protect against stalling
        # wait 2 s before activating the controller
        if time.time() - start_time_trial > 1:
            pub_safety_value.publish(1.0)
        else:
            pub_safety_value.publish(0.0)


        # read most recent message from s_1 topic
        s_1_now = rospy.wait_for_message("/s", Float32)
        # chek if the lap was completed
        if s_1_now.data < s_1_prev.data and started_timer==False:
            started_timer = True
            start_time = time.time()
            lap_count += 1
            print("Lap completed: ", lap_count-1)
        
        elif s_1_now.data - s_1_prev.data < -20 and started_timer==True:
            lap_count += 1
            print("Lap completed: ", lap_count-1)
        # update the previous value
        s_1_prev = s_1_now
        # print started_timer
        #print("started_timer: ", started_timer)
        if started_timer:
            elapsed_time = time.time() - start_time
            distance_from_centerline_now = rospy.wait_for_message("/distance_from_centerline", Float32)
            if distance_from_centerline_now.data > lane_radius:
                lane_bound_penalty += lane_violation_cost * (distance_from_centerline_now.data - lane_radius)

    # set safety to 0 immediately after the trial is completed
    pub_safety_value.publish(0.0)

    # if elapsed_time < 6 * max_laps or distance_from_centerline_now.data - lane_width/2 > lane_width/2: # something went wrong, like exited lane or some strange behavior
    #     print('Trial aborted due to too short lap time or too large lane violation')
    #     elapsed_time = 35

    
    print('Elapsed time: ', elapsed_time)
    print('Lane bound penalty: ', lane_bound_penalty)


    return elapsed_time + lane_bound_penalty





# save study
study_name = optuna_studies_folder +"/optuna_study_results_ROS_" + MPC_algorithm
storage_name = "sqlite:///" + study_name + ".db"  # SQLite database file



# --- define the sampler ---

n_startup_trials = 1

#from optuna.samplers import TPESampler
# sampler = TPESampler()
from optuna.integration import BoTorchSampler
sampler = BoTorchSampler(n_startup_trials=n_startup_trials)  # GP starts after 5 random trials


study = optuna.create_study(study_name=study_name,
                            direction="minimize",
                            storage=storage_name,
                            load_if_exists=True,
                            sampler=sampler)

# evaluate initial guess as first trial
#study.enqueue_trial(initial_guess)
# Add first 5 warm-start trials close to the initial guess
import copy
import numpy as np
for i in range(n_startup_trials):
    print('--------- trial ', i, '---------')
    perturbed = copy.deepcopy(initial_guess)
    for key in initial_guess:
        # Add small Gaussian noise (std = 5% of range or fixed small amount)
        noise = np.random.normal(loc=0.0, scale=0.0 * (10 if "qt" not in key else 1))
        # show key and noise
        print(f"Perturbing {key} by noise: {noise:.2f}")
        perturbed[key] = max(0.0, initial_guess[key] + noise)  # enforce non-negative
    study.enqueue_trial(perturbed)





study.optimize(objective, n_trials=n_trials)

print("Best hyperparameters:", study.best_params)

study.trials_dataframe().to_csv(study_name)


# try reloading study
study = optuna.load_study(study_name=study_name, storage=storage_name)


optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
# to visualize using dashboard, use the following command in terminal:
# optuna-dashboard sqlite:///optuna_study_results_ROS_MPCC.db  (use actual name of the database file)


plt.show()