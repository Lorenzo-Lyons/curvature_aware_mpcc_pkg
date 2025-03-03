import numpy as np
import os
from functions_for_solver_generation import generate_low_level_solver_ocp, generate_high_level_path_planner_ocp


# select the solver to build MPCC or CAMPCC
MPC_algorithm = 'MPCC' # 'MPCC' or 'CAMPCC'



# decide which solvers to build
build_acados=True
build_FORCES=True
# select the solver to build MPCC or CAMPCC
MPC_algorithms = ['MPCC','CAMPCC'] 



# instantiate the class
for MPC_algorithm in MPC_algorithms:
    solver_maker_obj = generate_high_level_path_planner_ocp(MPC_algorithm)

    # change current folder to be where the solvers need to be put
    current_script_path = os.path.realpath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    path_to_built_solvers = os.path.join(current_script_dir,'solvers')
    os.chdir(path_to_built_solvers)


    print('_________________________________________________')
    print('Building solver ', solver_maker_obj.solver_name_forces)
    print('')
    if build_acados:
        from acados_template import AcadosOcpSolver
        ocp = solver_maker_obj.produce_ocp()
        solver = AcadosOcpSolver(ocp, json_file=solver_maker_obj.solver_name_acados + '.json')
    if build_FORCES:
        model,codeoptions = solver_maker_obj.produce_FORCES_model_codeoptions()
        solver = model.generate_solver(codeoptions)

    print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')

print('------------------------------')
print('All done with solver building!')
print('------------------------------')








