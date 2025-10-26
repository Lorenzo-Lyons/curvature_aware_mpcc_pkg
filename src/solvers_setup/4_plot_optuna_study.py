import os
import optuna

import rospy
from std_msgs.msg import Float32
from dynamic_reconfigure.client import Client
from solver_manager_classes import MPC_solver_handler
import matplotlib.pyplot as plt
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

plt.rcParams.update({'font.size': 20})

# select algorithm to tune
MPC_algorithms = ['MPCCPP','CAMPCC'] # 'MPCC' - 'CAMPCC' - 'MPCCPP'
time_horizon_vec = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] # start from longer horizons to shorter ones  , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
software = 'forcespro'  # 'acados' or 'forcespro'

optuna_studies_folder = 'optuna_studies'
track = "analytic_circle"
if track == "analytic_circle":
    optimal_lap_time = 3.919247413153508
elif track == "vicon_racetrack":
    optimal_lap_time = 8.3


# define colors fro plots
l = np.max([2,len(time_horizon_vec)])  # number of colors you want
cmap = plt.get_cmap("viridis_r")  # reversed viridis colormap
colors = [cmap(i / (l - 1)) for i in range(l)]  # l equally spaced colors



# Define figure and axis
fig, ax_mpccpp = plt.subplots(figsize=(8, 6))
fig2, ax_campcc = plt.subplots(figsize=(8, 6))
plt.ion()
plt.show()


for time_horizon, color in zip(time_horizon_vec, colors):
    for controller_type in MPC_algorithms:
        
        # define MPC algorithm and time horizon in the GUI
        MPC_solver_handler_obj = MPC_solver_handler(controller_type,time_horizon,software)

        # define study name
        study_name = os.path.join(optuna_studies_folder, "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track)
        storage_name = os.path.join("sqlite:///", study_name + ".db")

        study = optuna.load_study(study_name=study_name, storage=storage_name)

        if controller_type == 'MPCCPP':
            # Extract the completed trials
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value < optimal_lap_time*2]

            # Get parameter values
            qt_pos = [t.params["qt_pos"] for t in trials if "qt_pos" in t.params]
            qt_s = [t.params["qt_s"] for t in trials if "qt_s" in t.params]
            values = [t.value for t in trials]  # optional, if you want to color by objective value
            
            # get best parameters
            best_trial = study.best_trial
            best_qt_pos = best_trial.params["qt_pos"]
            best_qt_s = best_trial.params["qt_s"]

            # Scatter plot on the defined axis
            #sc = ax_mpccpp.scatter(qt_s, qt_pos, c=values, cmap="viridis_r", s=25, edgecolors='0.6')
            sc = ax_mpccpp.scatter(qt_s, qt_pos, color = color, s=25, edgecolors='0.6', alpha=0.5)

            # add a red star for the best parameters
            ax_mpccpp.scatter(best_qt_s, best_qt_pos, color=color,  s=75,  edgecolors='0.0',label='Best ' + f'(horizon = {time_horizon:.1f} s)',zorder=50) 

            # # Add colorbar and labels
            # if 'cbar_mpccpp' in locals():
            #     cbar_mpccpp.remove()
            # cbar_mpccpp = plt.colorbar(sc, ax=ax_mpccpp)
            # cbar_mpccpp.set_label("Objective Value")

            ax_mpccpp.set_xlabel(r"$q^t_s$")
            ax_mpccpp.set_ylabel(r"$q^t_{pos}$")
            ax_mpccpp.set_title("MPCC++")
            ax_mpccpp.grid(True)
            ax_mpccpp.set_xlim(0, 100*1.1)
            ax_mpccpp.set_ylim(0, 100*1.1)
            # Legend if best trial marked
            handles, labels = ax_mpccpp.get_legend_handles_labels()
            if handles:
                ax_mpccpp.legend(loc="best")

            #plt.pause(0.5) 

        elif controller_type == 'CAMPCC':
            import numpy as np

            # Extract completed trials (same filtering)
            trials = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value < optimal_lap_time*2]

            # Get parameter values
            qt_s = [t.params["qt_s"] for t in trials if "qt_s" in t.params]
            qt_v = [t.params["qt_v"] for t in trials if "qt_v" in t.params]
            values = [t.value for t in trials]

            # Handle empty case
            if len(qt_s) == 0 or len(qt_v) == 0:
                ax_campcc.clear()
                ax_campcc.set_title("CAMPCC (no completed trials with qt_s / qt_v)")
                #plt.pause(0.5)
            else:
                qt_s = np.asarray(qt_s)
                qt_v = np.asarray(qt_v)
                values = np.asarray(values)

                # Reuse the same colormap and normalization as MPCC++
                cmap = sc.cmap if 'sc' in locals() else plt.get_cmap("viridis_r")
                norm = getattr(sc, "norm", None) if 'sc' in locals() else None

                # Clear and scatter
                #ax_campcc.clear()
                # sc_campcc = ax_campcc.scatter(
                #     qt_s, qt_v,
                #     c=values, cmap=cmap, norm=norm,
                #     s=25, edgecolors='0.6'
                # )

                sc_campcc = ax_campcc.scatter(
                    qt_s, qt_v,
                    color = color,
                    s=25, alpha=0.5 ,edgecolors='0.6'
                )


                # Mark best trial
                best_trial = study.best_trial
                if "qt_s" in best_trial.params and "qt_v" in best_trial.params:
                    best_qt_s = best_trial.params["qt_s"]
                    best_qt_v = best_trial.params["qt_v"]
                    #ax_campcc.scatter(best_qt_s, best_qt_v, color='red', marker='+', s=200, label='Best Parameters')
                    ax_campcc.scatter(best_qt_s, best_qt_v, color=color, s=75,edgecolors='0.0' ,label='Best ' + f'(horizon = {time_horizon:.1f} s)',zorder=50) 

                # # Update colorbar
                # if 'cbar_campcc' in locals():
                #     cbar_campcc.remove()
                # cbar_campcc = plt.colorbar(sc_campcc, ax=ax_campcc)
                # cbar_campcc.set_label("Objective Value")

                # Axis labels and formatting
                ax_campcc.set_xlabel(r"$q^t_s$")
                ax_campcc.set_ylabel(r"$q^t_v$")
                ax_campcc.set_title("CAMPCC")
                ax_campcc.grid(True)

                # Apply same limits convention as MPCC++
                ax_campcc.set_xlim(0, 100 * 1.1)
                ax_campcc.set_ylim(0, 100 * 1.1)

                # Legend if best trial marked
                handles, labels = ax_campcc.get_legend_handles_labels()
                if handles:
                    ax_campcc.legend(loc="best")

                #plt.pause(0.5)



plt.ioff()



lw = 2
color_ca = 'dodgerblue'
color_pp = 'orangered'
colors = [color_pp, color_ca]
labels = ['MPCC++', 'rCA-MPCC']

# plot best lap time vs time horizon
fig3, ax_lap_time = plt.subplots(figsize=(10, 4))

# add horizontal line for optimal lap time
ax_lap_time.axhline(y=optimal_lap_time, color='gray', linestyle='--',linewidth = lw ,label='Optimal Lap Time')

time_horizon_vec_flipped = list(reversed(time_horizon_vec))




# number of best runs to average
n_best_to_average = 5  

for controller_type, color, label in zip(MPC_algorithms, colors, labels):
    best_lap_times = []

    for time_horizon in time_horizon_vec_flipped:
        # define MPC algorithm and time horizon in the GUI
        MPC_solver_handler_obj = MPC_solver_handler(controller_type, time_horizon, software)

        # define study name and storage
        study_name = os.path.join(
            optuna_studies_folder,
            "optuna_study_" + MPC_solver_handler_obj.solver_name_forcespro + '_' + track
        )
        storage_name = os.path.join("sqlite:///", study_name + ".db")

        # load the study
        study = optuna.load_study(study_name=study_name, storage=storage_name)

        # get all completed trials
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # sort them by objective value (ascending = better)
        sorted_trials = sorted(trials, key=lambda t: t.value)

        # take up to n_best_to_average (handle studies with fewer trials)
        top_k = min(n_best_to_average, len(sorted_trials))

        if top_k > 0:
            avg_best = np.mean([t.value for t in sorted_trials[:top_k]])
        else:
            avg_best = np.nan  # or skip / handle differently if no trials

        best_lap_times.append(avg_best)

    # plot results
    ax_lap_time.plot(
        time_horizon_vec_flipped,
        best_lap_times,
        marker='o',
        color=color,
        label=label,
        linewidth = lw,
    )

# add labels and legend
ax_lap_time.set_xlabel("MPC Time Horizon [s]")
ax_lap_time.set_ylabel(f"Lap Time [s]")
ax_lap_time.legend()
#ax_lap_time.set_ylim(0, 1.1 * max(best_lap_times ))

# adjust subplots
fig3.subplots_adjust(
top=1.0,
bottom=0.180,
left=0.1,
right=0.995,
hspace=0.2,
wspace=0.2
)

# use the horizon length as x ticks
ax_lap_time.set_xticks(time_horizon_vec_flipped)
ax_lap_time.invert_xaxis()

plt.show()