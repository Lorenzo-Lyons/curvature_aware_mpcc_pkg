from MPC_generate_solvers.functions_for_solver_generation import generate_high_level_path_planner_ocp, generate_high_level_MPCC_PP
import numpy as np
import os
import forcespro.nlp
import matplotlib.pyplot as plt
from mpc_node import path_handeling_utilities_class
import optuna
import time


MPC_algorithm = 'MPCC_PP' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'
study_name = "optuna_study_results" + MPC_algorithm + ".csv"
storage_name = "sqlite:///"+study_name+".db"  # SQLite database file

study = optuna.load_study(study_name=study_name, storage=storage_name)

# Print best parameters found so far
print("Best parameters:", study.best_params)

# visualize the optimization history
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

plt.show()

