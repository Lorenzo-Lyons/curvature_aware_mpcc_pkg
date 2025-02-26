import numpy as np
import os
from functions_for_solver_generation import generate_single_layer_CAMPCC
from acados_template import AcadosOcpSolver

# select the solver to build MPCC or CAMPCC
dynamic_model = 'kinematic_bicycle' # 'kinematic_bicycle', 'dynamic_bicycle'

# instantiate the class
solver_maker_obj = generate_single_layer_CAMPCC(dynamic_model)








# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers = os.path.join(current_script_dir,'solvers')
os.chdir(path_to_built_solvers)

# build acados solver
print('_________________________________________________')
print('Building solver ', solver_maker_obj.solver_name_acados)
print('')
ocp = solver_maker_obj.produce_ocp()
solver = AcadosOcpSolver(ocp, json_file=solver_maker_obj.solver_name_acados + '.json') # this will regenerate the solver
print('')

# build forces solver
print('_________________________________________________')
print('Building solver ', solver_maker_obj.solver_name_forces)
print('')
model,codeoptions = solver_maker_obj.produce_FORCES_model_codeoptions()
solver = model.generate_solver(codeoptions)

print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

print('------------------------------')
print('All done with solver building!')
print('------------------------------')








