import os
import optuna


ROS_study = True
MPC_algorithm = 'MPCC_PP' # 'MPCC' - 'CAMPCC' - 'MPCC_PP'



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

# visualize the optimization history
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()



