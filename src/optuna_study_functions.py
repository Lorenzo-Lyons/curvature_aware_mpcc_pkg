class set_up_GUI_optuna(): # inherits from DART system identification

    def __init__(self,GUI_mpc_node,single_layer_tag,MPC_algorithm,ROS_study):

        # select algorithm to tune
        #MPC_algorithm = 'CAMPCC' #

        if MPC_algorithm == 'MPCC':
            algorithm_number = 0
        elif MPC_algorithm == 'CAMPCC':
            algorithm_number = 1
        elif MPC_algorithm == 'MPCC_PP':
            algorithm_number = 2

        # set mpc node parameters
        self.lane_violation_cost = 10
        self.lane_width = 0.6
        GUI_mpc_node.update_configuration({"lane_width": self.lane_width})
        GUI_mpc_node.update_configuration({"Solver_software": 1})
        GUI_mpc_node.update_configuration({"MPC_algorithm": algorithm_number})
        GUI_mpc_node.update_configuration({"Dynamic_model": 1})   # dynamic bicycle model
        GUI_mpc_node.update_configuration({"qt_pos_high": 50})
        GUI_mpc_node.update_configuration({"qt_rot_high": 10})
        if single_layer_tag:
            GUI_mpc_node.update_configuration({"single_layer": True})
            GUI_mpc_node.update_configuration({"V_target": 2}) # this is actually not used in the solver but only for the path params
            single_layer_name = 'SINGLE_layer_'
        else:
            GUI_mpc_node.update_configuration({"single_layer": False})
            GUI_mpc_node.update_configuration({"V_target": 2.5})
            single_layer_name = []

        if ROS_study:
            ROS_stufy_name = "ROS_"
        else:
            ROS_stufy_name = []

        self.study_name = "optuna_studies/optuna_results_" + single_layer_name + ROS_stufy_name + MPC_algorithm
        self.storage_name = "sqlite:///" + self.study_name + ".db"  # SQLite database file

