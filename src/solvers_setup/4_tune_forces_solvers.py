import rospy
from dynamic_reconfigure.client import Client
import optuna
import logging
# Suppress info messages from Optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32
import copy
import numpy as np
import os
from solver_manager_classes import MPC_solver_handler
from itertools import product
from tqdm import tqdm




# NOTE 
# from here you need to run the rviz simulator and  ONLY RUN THE MPC NODE, without the safety toggle, otherwise you will be sending safety on also.


# change folder to where this script is located
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



# Because we tune the CAMPCC after the MPCCPP, we will now train one after the other





# select algorithm to tune
MPC_algorithms = ['MPCCPP','CAMPCC'] # 'MPCC' - 'CAMPCC' - 'MPCCPP'
time_horizon_vec = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # start from longer horizons to shorter ones
software = 'forcespro'  # 'acados' or 'forcespro'

optuna_studies_folder = 'optuna_studies'


# specify track you are training on (just for initial guess of parameters)
track = "analytic_circle"
# track = "vicon_racetrack"

rospy.init_node("optuna_node")  # Initialize the node
pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)


# Create a client to communicate with the reconfigurable node
GUI_client_simulator = Client("/drone_simulator_node", timeout=5)  
GUI_mpc_node = Client("/mpc_node", timeout=5) 

max_laps = 1

n_trials = 100



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

# define initial parameter guess for optuna
def produce_initial_guess(controller_type, previous_study_name, previous_storage_name):
    # check if previous study exists
    if previous_study_name != [] and previous_storage_name != []:
        #print('Previous study found for longer time horizon, using as initial guess.')
        #print('loading initial guess from: ', previous_study_name)
        study = optuna.load_study(study_name=previous_study_name, storage=previous_storage_name)
        initial_guess = {}
        for key, value in study.best_params.items():
            initial_guess[key] = value



    else:
        initial_guess = {
        "qt_s": 0.1,
        "qt_pos": 10.0,
        }

        if controller_type == "CAMPCC":
            initial_guess = {
            "qt_v": 1,
            }


    return initial_guess

    



def reset_initial_position(GUI_client_simulator, track):
    if track == "analytic_circle":
        GUI_client_simulator.update_configuration({"reset_state_x": 0.0})
        GUI_client_simulator.update_configuration({"reset_state_y": -3})
        GUI_client_simulator.update_configuration({"reset_state_z": 1})
    elif track == "vicon_racetrack":
        GUI_client_simulator.update_configuration({"reset_state_x": -2.88})
        GUI_client_simulator.update_configuration({"reset_state_y": -1.2})
        GUI_client_simulator.update_configuration({"reset_state_z": 2.4})
    
    GUI_client_simulator.update_configuration({"reset_state": True})
    GUI_client_simulator.update_configuration({"reset_state_roll": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_pitch": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_yaw": 0.0})

def set_mpc_node_GUI(trial,GUI_mpc_node, MPC_solver_handler_obj):
    # generate random parameters
    # q_roll_pitch = trial.suggest_float("q_roll_pitch", 0.01, 1, log=False) 
    # q_yaw = trial.suggest_float("q_yaw", 0.01, 1, log=False)
    #qt_s = trial.suggest_float("qt_s", 0.01, 100, log=True) 

    # algorithm specific parameters
    controller_type = MPC_solver_handler_obj.controller_type

    
    if controller_type == 'MPCC' or controller_type == 'MPCCPP':
        qt_s = trial.suggest_float("qt_s", 0.01, 100, log=True) 
        qt_pos = trial.suggest_float("qt_pos", 0.01, 100, log=True) #0.1
        qt_v = 0
        GUI_mpc_node.update_configuration({"qt_pos": qt_pos})
        GUI_mpc_node.update_configuration({"qt_s": qt_s})

    elif controller_type == 'CAMPCC':
        qt_v = trial.suggest_float("qt_v", 0.01, 100, log=False) 
        GUI_mpc_node.update_configuration({"qt_v": qt_v})


    #read config
    #config_mpc = GUI_mpc_node.get_configuration()
    #cange parameters of interest
    if controller_type == 'MPCC':
        algorithm_number = 0
    elif controller_type == 'CAMPCC':
        algorithm_number = 1
    elif controller_type == 'MPCCPP':
        algorithm_number = 2

    # set GUI params
    GUI_mpc_node.update_configuration({"controller_type": algorithm_number})
    GUI_mpc_node.update_configuration({"time_horizon": MPC_solver_handler_obj.time_horizon})


    



    

config_simulator = GUI_client_simulator.get_configuration()
#print(config_simulator)  # Print all available parameters
config_mpc = GUI_mpc_node.get_configuration()
#print(config_mpc)




# # optimal_lap_timee = time_optimal_trajectory_4_warmstart[-1,15]  # last element is the total time of the optimal trajectory

if track == "analytic_circle":
    optimal_lap_time = 4
elif track == "vicon_racetrack":
    optimal_lap_time = 8.3








# -------------------------------- simualtion loop --------------------------------
def objective(trial, MPC_solver_handler_obj, track):
    pub_safety_value.publish(0.0)

    # reset initial position to before the start of the track
    reset_initial_position(GUI_client_simulator, track)
    set_mpc_node_GUI(trial, GUI_mpc_node, MPC_solver_handler_obj)

    
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

        # TEMPORARY PRINTING of current and previous s values with 3 decimals
        #print(f"--- s_now: {s_1_now.data:.3f} | s_prev: {s_1_prev.data:.3f} ---")


        if s_1_now.data < s_1_prev.data and started_timer==False:
            started_timer = True
            start_time = time.time()
            lap_count += 1
            #print("Lap completed: ", lap_count-1)
        
        elif s_1_now.data - s_1_prev.data < -10 and started_timer==True:
            lap_count += 1
            #print("Lap completed: ", lap_count-1)
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

    
    #print('Elapsed time: ', elapsed_time)
    #print('Lane bound penalty: ', lane_bound_penalty)


    return elapsed_time + lane_bound_penalty









# --- define the sampler ---

n_startup_trials = 3

#from optuna.samplers import TPESampler
# sampler = TPESampler()
import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
from optuna.integration import BoTorchSampler





# tune all the algorithms and time horizons
previous_study_name_MPCCPP = []
previous_storage_name_MPCCPP = []
previous_study_name_CAMPCC = []
previous_storage_name_CAMPCC = []



# all this parafenallia is just to have a progress bar that works well with rospy

# # helper so you can keep using `log()` instead of `print()`
# log = tqdm.write

# pairs = list(product(time_horizon_vec, MPC_algorithms))

# for time_horizon, controller_type in tqdm(
#     pairs,
#     total=len(pairs),
#     desc="Full tuning progress",
#     dynamic_ncols=True,
#     leave=True,          # keep the bar after completion
#     position=0
# ):
#     # logs that don't disturb the bar:
#     log("\n" + "─" * 50)
#     MPC_solver_handler_obj = MPC_solver_handler(controller_type, time_horizon, software)
#     log(f"Tuning algorithm: {MPC_solver_handler_obj.solver_name_forcespro}")


print('')
print('')
print('')

# persistent top bar over all (time_horizon, controller_type) pairs
outer = tqdm(total=len(time_horizon_vec) * len(MPC_algorithms),
             desc="Overall progress", position=0, leave=True, dynamic_ncols=True)

for time_horizon in time_horizon_vec:
    for controller_type in MPC_algorithms:
        
        # define MPC algorithm and time horizon in the GUI
        MPC_solver_handler_obj = MPC_solver_handler(controller_type,time_horizon,software)
        #print('_________________________________________________')
        #print('Tuning algorithm: ', MPC_solver_handler_obj.solver_name_forcespro)


        # define study name and storage
        study_name = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track)
        storage_name = os.path.join("sqlite:///", study_name + ".db")
        #study_name = optuna_studies_folder +"/optuna_study_results_ROS_" + controller_type
        #storage_name = "sqlite:///" + study_name + ".db"  # SQLite database file
        
        # create fresh sampler
        sampler = BoTorchSampler(n_startup_trials=n_startup_trials)  # GP starts after 5 random trials

        study = optuna.create_study(study_name=study_name,
                                    direction="minimize",
                                    storage=storage_name,
                                    load_if_exists=True,
                                    sampler=sampler)


        # define initial guess
        if controller_type == 'MPCCPP':
            previous_study_name = previous_study_name_MPCCPP
            previous_storage_name = previous_storage_name_MPCCPP
        elif controller_type == 'CAMPCC':
            previous_study_name = previous_study_name_CAMPCC
            previous_storage_name = previous_storage_name_CAMPCC

        initial_guess = produce_initial_guess(controller_type, previous_study_name, previous_storage_name)


        # If algorithm is CAMPCC we use the same parameters as found for MPCCPP and then only tune the additional one
        if controller_type == 'CAMPCC':
            # load previously found best parameters for MPCCPP as they will not be changed
            #load_study_name = optuna_studies_folder+"/optuna_study_results_ROS_" + "MPCCPP"
            #storage_name = "sqlite:///" + load_study_name + ".db"  # SQLite database file
            MPC_solver_handler_obj_MPCCPP = MPC_solver_handler("MPCCPP",time_horizon,'forcespro')
            study_name_MPCCPP = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj_MPCCPP.solver_name_forcespro  + '_' + track)
            storage_name_MPCCPP = os.path.join("sqlite:///", study_name_MPCCPP + ".db")

            #print('Controller is CAMPCC, using the values for qt_pos and qt_s from previously optimized MPCCPP.')
            #print('loading GUI parameters from: ', study_name_MPCCPP)
            study_MPCC = optuna.load_study(study_name=study_name_MPCCPP, storage=storage_name_MPCCPP)
            for key, value in study_MPCC.best_params.items():
                GUI_mpc_node.update_configuration({key: value})



        # pre-load initial guess as first trial
        for i in range(n_startup_trials):
            #for i in tqdm(range(n_startup_trials), desc="Iterations", position=1, leave=False, dynamic_ncols=True):
            #print('--------- trial ', i, '---------')
            perturbed = copy.deepcopy(initial_guess)
            for key in initial_guess:
                # Add small Gaussian noise (std = 5% of range or fixed small amount)
                noise = np.random.normal(loc=0.0, scale=0.01 * (10 if "qt" not in key else 1))
                # show key and noise
                #print(f"Perturbing {key} by noise: {noise:.2f}")
                perturbed[key] = max(0.0, initial_guess[key] + noise)  # enforce non-negative
            study.enqueue_trial(perturbed)


        # inner bar for this study's trials
        inner = tqdm(total=n_trials,
                     desc=f"Trials: {controller_type} th={time_horizon}",
                     position=1, leave=False, dynamic_ncols=True)

        # callback: tick inner bar on every finished trial
        def _pb_callback(study, trial):
            inner.update(1)
            # optional: show latest value/state on the tail
            # inner.set_postfix(value=f"{trial.value:.3g}" if trial.value is not None else "—",
            #                   state=str(trial.state).split('.')[-1])



        # perform the optimization
        #study.optimize(lambda trial: objective(trial, MPC_solver_handler_obj, track),n_trials=n_trials)
        # run the study with the callback
        study.optimize(lambda t: objective(t, MPC_solver_handler_obj, track),
                       n_trials=n_trials,
                       callbacks=[_pb_callback])

        inner.close()
        

        
        #study.optimize(objective, n_trials=n_trials)
        #print("Best hyperparameters:", study.best_params)
        study.trials_dataframe().to_csv(study_name)


        # save study name for next iteration
        if controller_type == 'MPCCPP':
            previous_study_name_MPCCPP = study_name
            previous_storage_name_MPCCPP = storage_name
        elif controller_type == 'CAMPCC':
            previous_study_name_CAMPCC = study_name
            previous_storage_name_CAMPCC = storage_name

        #print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
        #print('')
    
    outer.update(1)

outer.close()



# print message that we are done
print('------------------------------')
print('All done with tuning!')
print('------------------------------')













# # try reloading study
# study = optuna.load_study(study_name=study_name, storage=storage_name)


# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_param_importances(study).show()
# # to visualize using dashboard, use the following command in terminal:
# # optuna-dashboard sqlite:///optuna_study_results_ROS_MPCC.db  (use actual name of the database file)


# plt.show()