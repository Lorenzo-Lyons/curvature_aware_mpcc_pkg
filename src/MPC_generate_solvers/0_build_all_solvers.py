import numpy as np
import os
from acados_template import AcadosOcpSolver
from functions_for_solver_generation import generate_high_level_path_planner_ocp, generate_low_level_solver_ocp


# Build all solvers for both high level and low level planning 

# select the solver to build MPCC or CAMPCC
MPC_algorithms = ['CAMPCC','MPCC'] # 'MPCC' or 'CAMPCC'
dynamic_models = ['kinematic_bicycle','dynamic_bicycle'] # 'kinematic_bicycle', 'dynamic_bicycle'

# generate locations to where to build the solvers
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers_folder = os.path.join(current_script_dir,'solvers')




# --- high level solvers ---
for MPC_algorithm in MPC_algorithms:
    solver_maker_obj = generate_high_level_path_planner_ocp(MPC_algorithm)

    # build acados solver
    path_to_build_acados_solver = os.path.join(path_to_built_solvers_folder, solver_maker_obj.solver_name_acados)
    if not os.path.exists(path_to_build_acados_solver):
        os.makedirs(path_to_build_acados_solver)  # Creates the folder and any necessary parent folders
    os.chdir(path_to_build_acados_solver)
    ocp = solver_maker_obj.produce_ocp()
    solver = AcadosOcpSolver(ocp, json_file=solver_maker_obj.solver_name_acados + '.json') # this will regenerate the solver

    #build forces solver
    os.chdir(path_to_built_solvers_folder)
    model,codeoptions = solver_maker_obj.produce_FORCES_model_codeoptions()
    solver = model.generate_solver(codeoptions)
    print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

# --- low level solvers ---
for dynamic_model in dynamic_models:
    solver_maker_obj = generate_low_level_solver_ocp(dynamic_model)

    # build acados solver
    path_to_build_acados_solver = os.path.join(path_to_built_solvers_folder, solver_maker_obj.solver_name_acados)
    if not os.path.exists(path_to_build_acados_solver):
        os.makedirs(path_to_build_acados_solver)  # Creates the folder and any necessary parent folders
    os.chdir(path_to_build_acados_solver)
    ocp = solver_maker_obj.produce_ocp()
    solver = AcadosOcpSolver(ocp, json_file=solver_maker_obj.solver_name_acados + '.json') # this will regenerate the solver

    #build forces solver
    os.chdir(path_to_built_solvers_folder)
    model,codeoptions = solver_maker_obj.produce_FORCES_model_codeoptions()
    solver = model.generate_solver(codeoptions)
    print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')



print('------------------------------')
print('All done with solver building!')
print('------------------------------')
