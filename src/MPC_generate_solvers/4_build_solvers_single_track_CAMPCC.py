import numpy as np
import os
from functions_for_solver_generation import generate_single_layer_CAMPCC
from acados_template import AcadosOcpSolver


build_acados=False
build_FORCES=True

# select the solver to build MPCC or CAMPCC
dynamic_models = ['dynamic_bicycle_GP'] # 'kinematic_bicycle', 'dynamic_bicycle', 'dynamic_bicycle_GP'
actuator_dynamics = [False] # True, False

# select where to load the actuator dynamics from
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)


# # # Using GP and actuator dynamics from DART package
import importlib.resources
# import the GP parameters
with importlib.resources.path('DART_dynamic_models', 'SVGP_saved_parameters_slippery_floor') as data_path:
    GP_params_folder = str(data_path)

# # # import the actuator dynamics parameters
# # with importlib.resources.path('DART_dynamic_models', 'actuator_dynamics_saved_parameters') as data_path:
# #     actuator_dynamics_folder = str(data_path)
#GP_params_folder = os.path.join(current_script_dir, 'SVGP_saved_parameters') 

#GP_params_folder = os.path.join(current_script_dir, 'SVGP_saved_parameters_slippery_floor')
actuator_dynamics_folder = os.path.join(current_script_dir, 'actuator_dynamics_saved_parameters')



for dynamic_model in dynamic_models:
    for actuator_dynamics_tag in actuator_dynamics:
        print('Building solver for dynamic model: ', dynamic_model, ' with actuator dynamics: ', actuator_dynamics)
    # instantiate the class
    solver_maker_obj = generate_single_layer_CAMPCC(dynamic_model, actuator_dynamics_tag, actuator_dynamics_folder,GP_params_folder)

    # change current folder to be where the solvers need to be put

    path_to_built_solvers = os.path.join(current_script_dir,'solvers')
    os.chdir(path_to_built_solvers)

    # build acados solver
    print('')
    print('_________________________________________________')

    if build_acados:
        print('Building solver ', solver_maker_obj.solver_name_acados)
        ocp = solver_maker_obj.produce_ocp()
        solver = AcadosOcpSolver(ocp, json_file=solver_maker_obj.solver_name_acados + '.json') # this will regenerate the solver
    if build_FORCES:
        print('Building solver ', solver_maker_obj.solver_name_forces)
        model,codeoptions = solver_maker_obj.produce_FORCES_model_codeoptions()
        solver = model.generate_solver(codeoptions)
    print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
    print('')

print('------------------------------')
print('All done with solver building!')
print('------------------------------')








