import os
from solver_manager_classes import MPC_solver_handler


# this file builds the acados CAMPCC for dorne racing


# change current folder to be where the solvers need to be put
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers = os.path.join(current_script_dir, 'solvers')
os.chdir(path_to_built_solvers)

# instantiate the solver maker object
controller_type = ["CAMPCC"] # "MPCC", "MPCCPP","CAMPCC", "CAMPCC_EA", "CAMPCC_EA_2"
software_choice = ["forcespro"]  # "acados", "forcespro"
time_horizon_vec = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for controller in controller_type:
    for software in software_choice:
        for time_horizon in time_horizon_vec:
            print('_________________________________________________')
            MPC_solver_handler_obj = MPC_solver_handler(controller,time_horizon,software)
            if software == "acados":
                print('Building solver ', MPC_solver_handler_obj.solver_name)
                # move to directory called as the solver name
                if not os.path.exists(MPC_solver_handler_obj.solver_name):
                    os.makedirs(MPC_solver_handler_obj.solver_name)
                os.chdir(MPC_solver_handler_obj.solver_name)

                from acados_template import AcadosOcpSolver
                ocp = MPC_solver_handler_obj.produce_ocp()
                solver = AcadosOcpSolver(ocp, json_file=MPC_solver_handler_obj.solver_name + '.json')


            elif software == "forcespro":
                print('Building solver ', MPC_solver_handler_obj.solver_name_forcespro)
                model, codeoptions = MPC_solver_handler_obj.produce_FORCES_model_codeoptions()
                solver = model.generate_solver(codeoptions)


            print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾')
            print('')






print('------------------------------')
print('All done with solver building!')
print('------------------------------')
