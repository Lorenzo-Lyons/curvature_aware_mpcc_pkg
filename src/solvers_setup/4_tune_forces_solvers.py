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
import sys
from solver_manager_classes import MPC_solver_handler
from itertools import product
from tqdm import tqdm



# NOTE 
# from here you need to run the rviz simulator and  ONLY RUN THE MPC NODE, without the safety toggle, otherwise you will be sending safety on also.


# change folder to where this script is located

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from reference_path_handeling_functions import generate_path_data


# Because we tune the CAMPCC after the MPCCPP, we will now train one after the other





# select algorithm to tune
MPC_algorithms = ['CAMPCC'] # 'MPCC' - 'CAMPCC' - 'MPCCPP'
time_horizon_vec = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] # start from longer horizons to shorter ones  , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
software = 'forcespro'  # 'acados' or 'forcespro'

optuna_studies_folder = 'optuna_studies'

# training CAMPCC with qt_pos from  MPCCPP?
CAMPCC_qtpos_flags = [2]  # 0 set to 0, 1 set to previous best qt_pos from MPCCPP, 2 leave free to tune


# specify track you are training on (just for initial guess of parameters)
#track = "analytic_circle"
track = "vicon_racetrack"



# get optimal lap time from the offline solution
load_optimally_smoothed_path = True
optimal_lap_time = generate_path_data(track, load_optimally_smoothed_path)[-1]










rospy.init_node("optuna_node")  # Initialize the node
pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=1)


# Create a client to communicate with the reconfigurable node
GUI_client_simulator = Client("/drone_simulator_node", timeout=5)  
GUI_mpc_node = Client("/mpc_node", timeout=5) 

max_laps = 3

n_trials = 100 # must be more than number startup trials for the sampler to work well
n_startup_trials = 33


lane_violation_cost = 1 
lane_radius = 0.5
admissible_lane_violation = 1.5 * lane_radius # this value will abort the whole trail, it saves from spectacualr failure

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


    

def reset_initial_position(GUI_client_simulator, track):
    if track == "analytic_circle":
        GUI_client_simulator.update_configuration({"reset_state_x": 0.2})
        GUI_client_simulator.update_configuration({"reset_state_y": -3.2})
        GUI_client_simulator.update_configuration({"reset_state_z": 1})
    elif track == "vicon_racetrack":
        GUI_client_simulator.update_configuration({"reset_state_x": -2.4})
        GUI_client_simulator.update_configuration({"reset_state_y": -1.2})
        GUI_client_simulator.update_configuration({"reset_state_z": 1.2})
    
    GUI_client_simulator.update_configuration({"reset_state": True})
    GUI_client_simulator.update_configuration({"reset_state_roll": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_pitch": 0.0})
    GUI_client_simulator.update_configuration({"reset_state_yaw": 0.0})

def set_mpc_node_GUI(trial,GUI_mpc_node, MPC_solver_handler_obj,CAMPCC_qtpos_flag=0):
    # algorithm specific parameters
    controller_type = MPC_solver_handler_obj.controller_type

    max_qt_s = 1 if track == "vicon_racetrack" else 100
    min_qt_s = 0.01 if track == "vicon_racetrack" else 1

    if controller_type == 'MPCC' or controller_type == 'MPCCPP':
        qt_s = trial.suggest_float("qt_s", min_qt_s, max_qt_s, log=True) 
        qt_pos = trial.suggest_float("qt_pos", 1, 100, log=False) #0.1
        qt_v = 0

    elif controller_type == 'CAMPCC':
        qt_s = trial.suggest_float("qt_s", min_qt_s, max_qt_s, log=True)
        qt_v = trial.suggest_float("qt_v", 0.1, 100, log=False) 
        if CAMPCC_qtpos_flag == 0:
            qt_pos = 0
        elif CAMPCC_qtpos_flag == 1:
            # read best qt_pos from previous MPCCPP study
            # TEMPORARY UGLY COPY PASTE FROM ABOVE
            campcc_study_name = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track)
            # replace "MPCCPP" with "CAMPCC"    
            mpccpp_study_name = campcc_study_name.replace("CAMPCC", "MPCCPP")
            
            
            #mpccpp_study_name = os.path.join(optuna_studies_folder, "optuna_study_MPCCPP_forcespro_" + track)
            mpccpp_storage_name = os.path.join("sqlite:///", mpccpp_study_name + ".db")
            mpccpp_study = optuna.load_study(study_name=mpccpp_study_name, storage=mpccpp_storage_name)
            best_trial = mpccpp_study.best_trial
            best_qt_pos = best_trial.params["qt_pos"]
            qt_pos = best_qt_pos
            print('Using qt_pos from MPCCPP: ', qt_pos)
        elif CAMPCC_qtpos_flag == 2:
            qt_pos = trial.suggest_float("qt_pos", 0, 100, log=False)


    GUI_mpc_node.update_configuration({"qt_s": qt_s})
    GUI_mpc_node.update_configuration({"qt_pos": qt_pos})
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







# -------------------------------- simualtion loop --------------------------------
def objective(trial, MPC_solver_handler_obj, track):
    pub_safety_value.publish(0.0)


    # reset initial position to before the start of the track
    reset_initial_position(GUI_client_simulator, track)
    set_mpc_node_GUI(trial, GUI_mpc_node, MPC_solver_handler_obj,CAMPCC_qtpos_flag)

    
    s_1_prev = rospy.wait_for_message("/s", Float32)
    lap_count = 0
    started_timer = False
    elapsed_time = 0
    start_time_trial = time.time()
    lane_bound_penalty = 0 # this will accumelate lane violation cost
    time_limit = (optimal_lap_time)*max_laps*3.5
    last_lap_increase = time.time() # this is needed against false lap crossing detections
    delta_t_laps = 1 # minimum time between lap increases 
    prev_time = time.time()

    while True:
        #now = time.time()
        time_since_trial_start = (time.time()-start_time_trial)
        # --- explicit stop conditions ---
        if lap_count > max_laps:
            #print("laps_completed, exiting")
            exit_reason = "laps_completed"
            break

        if time_since_trial_start >= time_limit:
            #print("time_limit exceeded, exiting")
            exit_reason = "time_limit"
            break
        

        # wait 2 s before activating the controller
        if time.time() - start_time_trial > 1:
            pub_safety_value.publish(1.0)
        else:
            pub_safety_value.publish(0.0)

        # read most recent message from s_1 topic
        s_1_now = rospy.wait_for_message("/s", Float32)

        if s_1_now.data - s_1_prev.data < -1:
            if lap_count == 0: # first lap
                started_timer = True
                start_time = time.time()
                lap_count += 1
            else:
                if time.time() - last_lap_increase > delta_t_laps: #this protects against jittering over the line and counting multiple laps
                    lap_count += 1
                    last_lap_increase = time.time()
                    print("Lap completed: ", lap_count-1)


        # update the previous value
        s_1_prev = s_1_now

        if started_timer:
            elapsed_time = time.time() - start_time
            distance_from_centerline_now = rospy.wait_for_message("/distance_from_centerline", Float32)

            # if mega lane violation, exit
            if distance_from_centerline_now.data > admissible_lane_violation:
                exit_reason = "lane_violation"
                break

            # else penalise it
            dt = time.time() - prev_time
            prev_time = time.time()
            if distance_from_centerline_now.data > lane_radius:
                lane_bound_penalty += lane_violation_cost * (distance_from_centerline_now.data - lane_radius) * dt


    # set safety to 0 immediately after the trial is completed
    pub_safety_value.publish(0.0)

    if exit_reason == 'laps_completed': # check if too fast
        if elapsed_time < max_laps*optimal_lap_time*0.85 and started_timer == True:  # something went wrong, like exited lane or some strange behavior
            exit_reason = "too_fast"



    if exit_reason == "laps_completed":
        #print('Trial completed successfully.')
        print('average time: ', elapsed_time/max_laps)
        print('Lane bound penalty: ', lane_bound_penalty)
        objective_value = elapsed_time/max_laps + lane_bound_penalty
    else:
        print('Trial aborted due to: ' + exit_reason)
        objective_value = optimal_lap_time*max_laps*3.0 + lane_bound_penalty # adding lane penalty to help the solver distinguish


    # store time data and lane penalty data
    trial.set_user_attr("average_time", elapsed_time/max_laps)
    trial.set_user_attr("lane_bound_penalty", lane_bound_penalty)


    return objective_value









# --- define the sampler ---

#n_startup_trials = 3

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





print('')
print('')
print('')

# persistent top bar over all (time_horizon, controller_type) pairs
outer = tqdm(total=len(time_horizon_vec) * len(MPC_algorithms),
             desc="Overall progress", position=0, leave=True, dynamic_ncols=True)

for time_horizon in time_horizon_vec:
    for controller_type, CAMPCC_qtpos_flag in zip(MPC_algorithms,CAMPCC_qtpos_flags):  # add a dummy flag if training MPCCPP
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


        # 
        sampler = optuna.samplers.TPESampler(
            # Let TPE choose its well-tested default gamma: ~min(25, n/4)
            consider_prior=True,
            prior_weight=0.1,              # lower the smoothing
            consider_magic_clip=True,
            consider_endpoints=True,       # allow boundary exploration
            n_startup_trials=n_startup_trials, # rule of thumb: ~5–10 per dim (use your D)
            n_ei_candidates=100,           # more candidates -> better EI search
            multivariate=True,             # model interactions
            warn_independent_sampling=True,
            seed=0,                        # make behavior reproducible for debugging
        )


        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_name,
            load_if_exists=True,
            sampler=sampler
        )
        # define some warm start parameters to start from a feasible region


        qt_s_startup_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        # queue them BEFORE study.optimize
        for qt_s_startup_val in qt_s_startup_vals:
            print(f'Enqueuing startup trial with qt_s = {qt_s_startup_val:.2f}, qt_pos = 100, (qt_v=0 if applicable)', )
            startup_params_i = {"qt_s": qt_s_startup_val, "qt_pos": 100}
            # if controller is CAMPCC and we are training qt_pos, add it to the startup params
            if controller_type == 'CAMPCC':
                # add qt_v = 0
                startup_params_i["qt_v"] = 0.0

            study.enqueue_trial(startup_params_i, skip_if_exists=True)  # avoids duplicates if you re-run


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

        study.optimize(lambda t: objective(t, MPC_solver_handler_obj, track),
                       n_trials=n_trials,
                       callbacks=[_pb_callback])

        inner.close()
        

        
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



# reset drone position after tuning
reset_initial_position(GUI_client_simulator, track)


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