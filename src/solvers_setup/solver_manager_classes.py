import numpy as np
import casadi as ca
from acados_template import  AcadosOcp, AcadosModel
try:
    from drone_dynamic_model import drone
except:
    from .drone_dynamic_model import drone

class common_solver_parameters():
    def __init__(self):
        self.n_points_kernelized = 20 # number of points in the kernelized path (41 for reference)
        # stages

        self.kernel_choice = 'RBF'  # 'RBF', 'Matern2'
        self.path_lengthscale = 1.3/self.n_points_kernelized #1.3/self.n_points_kernelized
        lambda_val = 0.0001**2
        self.Kxx_inv, self.normalized_s_4_kernel_path = self.generate_fixed_path_quantities(self.path_lengthscale,
                                                                                            lambda_val,
                                                                                            self.n_points_kernelized)
        

    def unpack_derivatives(self, drone_state_dot):
        x_dot      = drone_state_dot[0]  # velocity in x
        y_dot      = drone_state_dot[1]  # velocity in y
        z_dot      = drone_state_dot[2]  # velocity in z
        acc_x      = drone_state_dot[3]  # acceleration in x
        acc_y      = drone_state_dot[4]  # acceleration in y
        acc_z      = drone_state_dot[5]  # acceleration in z
        roll_rate  = drone_state_dot[6]  # roll rate
        pitch_rate = drone_state_dot[7]  # pitch rate
        yaw_rate   = drone_state_dot[8]  # yaw rate

        return x_dot, y_dot, z_dot, acc_x, acc_y, acc_z, roll_rate, pitch_rate, yaw_rate



    def generate_fixed_path_quantities(self,lengthscale, lambda_val,n_points):
        normalized_s = np.linspace(0.0, 1.0, n_points)
        if self.kernel_choice == 'RBF':
            K_xx = self.K_RBF_kernel(normalized_s, normalized_s, lengthscale,n_points,n_points)
        elif self.kernel_choice == 'Matern2':
            K_xx = self.K_matern2_kernel(normalized_s, normalized_s, lengthscale,n_points,n_points)
        fixed_matrix = np.linalg.inv(K_xx + lambda_val * np.eye(len(normalized_s))) 
        return fixed_matrix, normalized_s


    def K_matern2_kernel(self, x1, x2, lengthscale,n,m):
        sqrt5 = np.sqrt(5)
        #check if x1 is SX casadi type
        if type(x1) == ca.SX:
            K = ca.SX.zeros((n, m))
            abs = lambda x: ca.fabs(x)

        elif type(x1) == ca.MX:
            K = ca.MX.zeros((n, m))
            abs = lambda x: ca.fabs(x)

        else:
            K = np.zeros((n, m))
            abs = lambda x: np.abs(x)

        for ii in range(n):
            for jj in range(m):
                r = abs((x1[ii]-x2[jj])) / lengthscale
                K[ii, jj] = (1 + sqrt5 * r + (5.0 / 3.0) * r**2) * np.exp(-sqrt5 * r)
        return K

    def K_RBF_kernel(self,x1, x2, lengthscale,n,m):
        #check if x1 is SX casadi type
        if type(x1) == ca.SX:
            K = ca.SX.zeros((n, m))
        elif type(x1) == ca.MX:
            K = ca.MX.zeros((n, m))
        else:
            K = np.zeros((n, m))

        for ii in range(n):
            for jj in range(m):
                K[ii, jj] = np.exp((-(x1[ii]-x2[jj])**2/(2*lengthscale**2)))
        return K

    def evaluate_kernelized_line_reg(self,s,local_path_length, labels):
        s_star = s / local_path_length # normalize s
        if self.kernel_choice == 'RBF':
            K_x_star = self.K_RBF_kernel(s_star, self.normalized_s_4_kernel_path,
                                self.path_lengthscale,1,self.n_points_kernelized)
        elif self.kernel_choice == 'Matern2':
            K_x_star = self.K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,
                                self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv

        return left_side @ labels



class MPC_solver_handler(common_solver_parameters): # inherits from DART system identification

    def __init__(self,controller_type, time_horizon, software_choice='acados'):
        super().__init__()
        self.time_horizon = time_horizon
        self.N = int(np.ceil(time_horizon / 0.1))  # number of control intervals, assuming 0.1s per interval
        self.controller_type = controller_type
        self.software_choice = software_choice  # 'acados' or 'forcespro'
        # self.solver_name = controller_type + '_' + str(time_horizon) + '_solver'
        # self.solver_name_forcespro = controller_type + '_' + str(time_horizon) + '_forcespro_solver'
        time_str = f"th{time_horizon:.1f}".replace('.', '')  # 0.9 → "09", 1.5 → "15"
        self.solver_name = f"{controller_type}_{time_str}_solver"
        self.solver_name_forcespro = f"{controller_type}_{time_str}_forcespro_solver"

        self.n_base_params = 17

        self.controller_type = controller_type 
        if controller_type == "MPCC":
            self.nx = 10 # [pos_x pos_y pos_z vx vy vz roll pitch yaw s]
            self.nu = 5 # desired roll, desired pitch, desired yaw, thrust, slack
            self.n_parameters = self.n_base_params + 6 * self.n_points_kernelized #

        elif controller_type == "CAMPCC":
            self.nx = 10 # [pos_x pos_y pos_z vx vy vz roll pitch yaw s]
            self.nu = 5 # desired roll, desired pitch, desired yaw, thrust,
            self.n_parameters = self.n_base_params + 10 * self.n_points_kernelized #
            #print('Number of parameters for CAMPCC: ', self.n_parameters * (self.N+1))

        elif controller_type == "MPCCPP":
            self.nx = 10 # [pos_x pos_y pos_z vx vy vz roll pitch yaw s]
            self.nu = 6 # desired roll, desired pitch, desired yaw, thrust, slack, s_dot
            self.n_parameters = self.n_base_params + 6 * self.n_points_kernelized #
        
        elif controller_type == "CAMPCC_EA":
            # this is the CAMPCC with Euler angles reference path integration instead of 9 separate spline objects
            self.nx = 16 # [pos_x pos_y pos_z vx vy vz roll pitch yaw s ref_x ref_y ref_z ref_roll ref_pitch ref_yaw]
            self.nu = 5 # desired roll, desired pitch, desired yaw, thrust,
            self.n_parameters = self.n_base_params + 2 * self.n_points_kernelized #

        elif controller_type == "CAMPCC_EA_2":
            # this is the CAMPCC with Euler angles reference path integration instead of 9 separate spline objects
            self.nx = 10 # [pos_x pos_y pos_z vx vy vz roll pitch yaw s]
            self.nu = 5 # desired roll, desired pitch, desired yaw, thrust,
            self.n_parameters = self.n_base_params + 7 * self.n_points_kernelized #

        # number of inequality constraints
        self.n_inequality_constraints = 1

        # define bounds for the normalized inputs
        self.normalized_lb_u = np.array([0, 0, 0, 0, 0, 0])  # lower bound for normalized inputs
        self.normalized_ub_u = np.array([1, 1, 1, 1, 1000, 10])  # upper bound for normalized inputs (slack has no upper bound) 
        
        
    
    def produce_ocp(self):


        # seting up ACADOS solver
        x = ca.MX.sym('x', self.nx)
        u = ca.MX.sym('u', self.nu)
        p = ca.MX.sym('p', self.n_parameters) # stage-wise parameters


        # Create model object
        model = AcadosModel()
        model.name = self.solver_name
        model.x = x
        model.u = u
        model.p = p

        # unpack state x
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(x)
        # unpack normalized inputs u
        if self.controller_type == 'MPCCPP':
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack, s_dot = self.unpack_u(u)
        else:
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack = self.unpack_u(u)
        
        # denormalize inputs
        u_denormalized = self.denormalize_u(u)
        # unpack denormalized inputs
        roll_desired, pitch_desired, yaw_desired, thrust, slack = self.unpack_u(u_denormalized)  # NOTE slack is always the same (it's not normalized)


        # unpack parameters
        #if self.controller_type == 'MPCC':
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s, lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)
        labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds = self.unpack_parameters_MPCC(model.p)
        if self.controller_type == 'CAMPCC':
            labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds, labels_d2x_ds2, labels_d2y_ds2, labels_d2z_ds2, labels_k =\
            self.unpack_parameters_CAMPCC(model.p)




        # define path quantities
        ref_x = self.evaluate_kernelized_line_reg(s, local_path_length, labels_x)
        ref_y = self.evaluate_kernelized_line_reg(s, local_path_length, labels_y)
        ref_z = self.evaluate_kernelized_line_reg(s, local_path_length, labels_z)
        ref_dxds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dxds)
        ref_dyds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dyds)
        ref_dzds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dzds)

        if self.controller_type == 'CAMPCC':
            ref_d2x_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2x_ds2)
            ref_d2y_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2y_ds2)
            ref_d2z_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2z_ds2)
            k = self.evaluate_kernelized_line_reg(s, local_path_length, labels_k)  # curvature of the path





        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'



        # --- set up the cost functions ---
        if self.controller_type == 'MPCC':
            s_dot = np.sqrt(vx**2 + vy**2 + vz**2+0.001)  # path progress rate is the speed of the drone
        elif self.controller_type == 'CAMPCC':
            s_dot = self.s_dot_CAMPCC(pos_x, pos_y, pos_z,vx,vy,vz,ref_x, ref_y, ref_z, ref_d2x_ds2,ref_d2y_ds2,ref_d2z_ds2,ref_dxds,ref_dyds,ref_dzds,k)  # path progress rate is the speed of the drone
        elif self.controller_type == 'MPCCPP':
            pass # s_dot is already unpacked from u


        model.f_expl_expr = self.continous_dynamics(roll_desired, pitch_desired, yaw_desired, thrust, vx, vy, vz, roll, pitch, yaw, s_dot)


        ocp.model.cost_expr_ext_cost  =  self.objective( roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, 
                                                                    pos_x, pos_y, pos_z, yaw ,s_dot,
                                                                    ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds,
                                                                    q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw,
                                                                    lane_radius, slack)

        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost_on_reference(pos_x, pos_y, pos_z, vx, vy, vz,\
                                                terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz,\
                                                qt_pos, qt_v,\
                                                qt_s, s, local_path_length)



        # constraints
        ocp.constraints.constr_type = 'BGH'

        # define bounds

        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4]) # contraining the inputs
        ocp.constraints.lbu =   np.array([0, 0, 0, 0, 0])
        ocp.constraints.ubu =   np.array([1, 1, 1, 1, 1000]) # slack has no upper bound



        # define lane boundary constraints
        #ocp.model.con_h_expr = self.lane_boundary_constraint(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, slack, lane_radius)  # Define h(x, u)
        #ocp.constraints.lh = np.array([0.0])  # Lower bound (h_min)
        #ocp.constraints.uh = np.array([100000])  # Upper bound (h_max)
        ocp.model.con_h_expr = self.lane_boundary_constraint_tube(  pos_x,pos_y,pos_z,
                                                                    ref_x,ref_y,ref_z,
                                                                    lane_radius,slack)

        ocp.constraints.lh = np.zeros(self.n_inequality_constraints)  # Lower bound (h_min)
        ocp.constraints.uh = 1000 * np.ones(self.n_inequality_constraints)  # Upper bound (h_max)

        # Initial state constraint
        ocp.constraints.x0 = np.zeros(self.nx)  # This is a default value, it will be updated at runtime

        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # GAUSS_NEWTON, EXACT

        ocp.solver_options.integrator_type = 'ERK' # IRK, ERK
        ocp.solver_options.sim_method_num_steps = 5  # e.g., 5 integration steps per interval
        ocp.solver_options.sim_method_num_stages = 1  # this is the order of the integrator, ERK1 (Euler), ERK2 ERK3 ERK4



        ocp.solver_options.nlp_solver_type = 'SQP' # SQP   SQP_RTI
        ocp.solver_options.tf = self.time_horizon  # time horizon in seconds

        # messing with the convergence criteria
        ocp.solver_options.qp_solver_warm_start = 1 # 0: no warm start, 1: warm start 2 : hot start
        ocp.solver_options.globalization = 'FIXED_STEP' # 'MERIT_BACKTRACKING', 'FIXED_STEP' # fixed is the default
        ocp.solver_options.nlp_solver_max_iter = 20  # Maximum SQP iterations 20
        ocp.solver_options.qp_solver_iter_max = 1
        ocp.solver_options.print_level = 0 # 0 no print

        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        ocp.solver_options.nlp_solver_tol_eq = 1e-3
        ocp.solver_options.nlp_solver_tol_ineq = 1e-3
        ocp.solver_options.nlp_solver_tol_comp = 1e-2


        # Initialize parameters with default values (this step is important to avoid dimension mismatch)
        ocp.parameter_values = np.zeros(self.n_parameters)
        return ocp
    



    def produce_FORCES_model_codeoptions(self):
        import forcespro.nlp


        model = forcespro.nlp.SymbolicModel(self.N+1) # this plus one is to keep the same output dimensions as the acados model that has 1 extra state

        model.xinitidx = np.array(range(self.nu, self.nu + self.nx))  # variables in these positions are affected by initial state constraint. (I.e. they cannot change in the first stage)
        
        #theese parameters are the same for all the solvers
        model.nvar = self.nu + self.nx         # number of stage variables
        model.neq = self.nx                    # number of equality constraints (dynamic model)
        model.npar = self.n_parameters         # number of parameters
        model.E = np.concatenate([np.zeros((self.nx, self.nu)), np.eye(self.nx)], axis=1)  # This extraction matrix tells forces what variables are states and what are inputs

        # set fixed input bounds since they will not change at runtime
        # generate inf upper and lower bounds for the inputs and states
        x_l = -1000 * np.ones(self.nx)
        x_u = +1000 * np.ones(self.nx)
        model.lb = np.array([*self.normalized_lb_u[:self.nu],*x_l])  # lower bound on inputs (slack is always >= 0)
        model.ub = np.array([*self.normalized_ub_u[:self.nu],*x_u])  # upper bound on inputs (slack has no upper bound)


        # --- Set objective ---

        for i in range(self.N):
            model.objective[i] = self.objective_forcespro  # eval_obj is a Python function
        #model.objective[self.N] = self.objective_terminal_cost_forcespro_on_reference
        model.objective[self.N] = self.objective_terminal_cost_forcespro_on_path


        # --- Set dynamic constraint ---
        model.continuous_dynamics = self.continous_dynamics_forcespro

        # # --- Set non linear constraints ---
        model.nh = 1
        model.ineq = self.lane_boundary_constraint_tube_forcespro

        model.hl = np.array([0])   # actually is already positive by construction
        model.hu = np.array([+1000])    # should be less than the lane radius

        # --- Define solver options ---
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver') #get standard options

        # continuous dynamics options
        codeoptions.nlp.integrator.Ts = self.time_horizon / (self.N+1)
        codeoptions.nlp.integrator.type = 'ERK4' #'ERK4' #'ForwardEuler' #'ERK4' #'IRK2' # 'ForwardEuler' #
        codeoptions.nlp.integrator.nodes = 1 # intermediate nodes for the integrator

        codeoptions.name = self.solver_name_forcespro
        codeoptions.printlevel = 0  #  1: summary line after each solve,   0: no prit
        codeoptions.BuildSimulinkBlock = 0  # disable simulink block generation because we don't need it
        codeoptions.maxit = 200  # maximum qp iterations (must be high otherwise it will crash)
        codeoptions.noVariableElimination = 1  # enable or disable variable simplification (like if first stage is constrained)
        codeoptions.nlp.stack_parambounds = True  # determines if the parameters can simply be stacked 


        # set solver type
        codeoptions.solvemethod = 'PDIP_NLP' # 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm (should not apply to sqp)

        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask         #set overwrite behviour

        # # # # set sqp parameters
        #codeoptions.sqp_nlp.maxSQPit = 2  # this seems to do nothing (?)
        #codeoptions.sqp_nlp.maxqps = 5    # maximum number of iterations
        # codeoptions.sqp_nlp.TolStat = 1e-3 # Tolerance on stationarity
        # codeoptions.sqp_nlp.TolEq = 1e-3 # Tolerance on equality constraints
        codeoptions.sqp_nlp.reg_hessian = 1e-5


        # codeoptions.nlp.TolStat = 1e-3  # Tolerance on gradient of Lagrangian
        # codeoptions.nlp.TolEq = 1e-3    # Tolerance on equality constraints
        # codeoptions.nlp.TolIneq = 1e-3  # Tolerance on inequality constraints
        # codeoptions.nlp.TolComp = 1e-3 # Tolerance on complementarity

        return model, codeoptions



    def action_state_bounds(self):
        # define the constraints on the state and action variables
        max_roll_desired = np.pi / 3
        max_pitch_desired = np.pi / 3
        max_yaw_desired = np.pi  # 2 * np.pi / 3 # relative to current yaw, so actually indirectly controlling the yaw rate with a first order controller
        max_thrust = 3.6   
        max_s_dot = 10.0  # maximum path progress rate (s_dot) in m/s for MPCCPP 
        # state bounds
        max_pos_x = 100
        max_pos_y = 100
        max_pos_z = 100
        max_vx = 10
        max_vy = 10
        max_vz = 10
        max_roll = 10 * np.pi # no need to constrain the roll
        max_pitch = 10 * np.pi # no need to constrain the pitch
        max_yaw = 10 * np.pi # no need to constrain the yaw
        max_s = 10

        # minum values
        min_roll_desired = -max_roll_desired
        min_pitch_desired = -max_pitch_desired
        min_yaw_desired = -max_yaw_desired
        min_thrust = -max_thrust
        min_s_dot = -max_s_dot

        # state bounds
        min_pos_x = -max_pos_x
        min_pos_y = -max_pos_y
        min_pos_z = -max_pos_z
        min_vx = -max_vx
        min_vy = -max_vy
        min_vz = -max_vz
        min_roll = -max_roll
        min_pitch = -max_pitch
        min_yaw = -max_yaw
        min_s = -0.1


        # create the bounds
        
        max_action_bounds = np.array([max_roll_desired, max_pitch_desired, max_yaw_desired, max_thrust])
        min_action_bounds = np.array([min_roll_desired, min_pitch_desired, min_yaw_desired, min_thrust])
        max_state_bounds = np.array([max_pos_x, max_pos_y, max_pos_z, max_vx, max_vy, max_vz, max_roll, max_pitch, max_yaw, max_s])
        min_state_bounds = np.array([min_pos_x, min_pos_y, min_pos_z, min_vx, min_vy, min_vz, min_roll, min_pitch, min_yaw, min_s])
        # if self.controller_type == 'MPCCPP':
        #     # add to control bounds the s_dot variable
        #     max_action_bounds = np.concatenate([max_action_bounds, [max_s_dot]])
        #     min_action_bounds = np.concatenate([min_action_bounds, [min_s_dot]])
        
        return min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds


    def normalize_state_action(self, z):
        min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds = self.action_state_bounds()
        
        # Normalize the action part
        z_normalized_action = (z[:self.nu] - min_action_bounds) / (max_action_bounds - min_action_bounds)
        
        # Normalize the state part
        z_normalized_state = (z[self.nu:] - min_state_bounds) / (max_state_bounds - min_state_bounds)
        
        # Concatenate the normalized action and state vectors
        z_normalized = np.concatenate([z_normalized_action, z_normalized_state])
        
        return z_normalized_action, z_normalized_state ,z_normalized

    def denormalize_u(self, u_normalized):
        min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds = self.action_state_bounds()

        denormalized_roll = u_normalized[0] * (max_action_bounds[0] - min_action_bounds[0]) + min_action_bounds[0]
        denormalized_pitch = u_normalized[1] * (max_action_bounds[1] - min_action_bounds[1]) + min_action_bounds[1]
        denormalized_yaw = u_normalized[2] * (max_action_bounds[2] - min_action_bounds[2]) + min_action_bounds[2]
        denormalized_thrust = u_normalized[3] * (max_action_bounds[3] - min_action_bounds[3]) + min_action_bounds[3]
        denormalized_slack = u_normalized[4] # slack is not nomralized 
        
        if self.controller_type == 'MPCCPP':
            denormalized_s_dot = u_normalized[5] # s_dot is not normalized 
            return denormalized_roll, denormalized_pitch, denormalized_yaw, denormalized_thrust, denormalized_slack, denormalized_s_dot
        else:
            return denormalized_roll, denormalized_pitch, denormalized_yaw, denormalized_thrust, denormalized_slack
        
    
    def normalize_u(self, u): # rempove the slack variable before passing the u to this function
        min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds = self.action_state_bounds()
        return (u - min_action_bounds) / (max_action_bounds - min_action_bounds)


    def denormalize_state_action(self, z_normalized):
        min_action_bounds, max_action_bounds, min_state_bounds, max_state_bounds = self.action_state_bounds()
        
        # Denormalize the action part
        z_action = z_normalized[:self.nu] * (max_action_bounds - min_action_bounds) + min_action_bounds
        
        # Denormalize the state part
        z_state = z_normalized[self.nu:] * (max_state_bounds - min_state_bounds) + min_state_bounds
        
        # Concatenate the denormalized action and state vectors
        z = ca.vertcat(z_action, z_state)
        
        return z_action, z_state, z



    def unpack_u(self, u):
        roll_desired  = u[0]   # desired roll
        pitch_desired = u[1]   # desired pitch
        yaw_desired   = u[2]   # desired yaw
        thrust  = u[3]   # thrust command
        slack = u[4]   # slack variable (not used in MPCC, but in CAMPCC)
        if self.controller_type == 'MPCCPP':
            s_dot = u[5]
            return roll_desired, pitch_desired, yaw_desired, thrust, slack, s_dot
        else :
            return roll_desired, pitch_desired, yaw_desired, thrust, slack

    def unpack_x(self, x):
        pos_x = x[0]         # position x
        pos_y = x[1]         # position y
        pos_z = x[2]         # position z

        vx = x[3]            # velocity in x
        vy = x[4]            # velocity in y
        vz = x[5]            # velocity in z

        roll  = x[6]         # actual roll
        pitch = x[7]         # actual pitch
        yaw   = x[8]         # actual yaw

        s = x[9]             # path progress variable

        return pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s

    def unpack_x_CAMPCC_EA(self, x):
        # only unpack the extra states (use normal funcion for the first 10 states)
        ref_x = x[10]       # reference path x at current s
        ref_y = x[11]       # reference path y at current s
        ref_z = x[12]       # reference path z at current s

        ref_roll = x[13]    # reference path roll at current s
        ref_pitch = x[14]
        ref_yaw = x[15]
        return ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw

    def unpack_base_parameters(self, p):
        q_sdot   = p[0]  # velocity tracking
        q_cont   = p[1]  # contouring error
        q_lag    = p[2]  # lag error
        q_roll_pitch   = p[3]  # desired roll
        q_yaw    = p[4]  # controls yaw dot indirectly
        q_thrust = p[5]  # desired thrust
        qt_pos   = p[6]  # terminal cost for position error
        qt_v =     p[7]  # terminal cost for velocity error
        qt_s     = p[8]  # terminal cost for path progress
        lane_radius = p[9]  # radius of the lane, used for the lane boundary constraint
        local_path_length = p[10]
        # terminal costs for position and velocity errors
        terminal_x = p[11]  # terminal cost for position error in x
        terminal_y = p[12]  # terminal cost for position error in y
        terminal_z = p[13]  # terminal cost for position error in z
        terminal_vx = p[14]  # terminal cost for velocity error in x
        terminal_vy = p[15]  # terminal cost for velocity error in y
        terminal_vz = p[16]  # terminal cost for velocity error in z

        return  q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
                terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz
    
    def unpack_parameters_MPCC(self, p):
        # unpack base parameters    
        # q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        # terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # -----------------------------
        # Reference path info
        # -----------------------------
        index_init = self.n_base_params

        N_pts = self.n_points_kernelized
        labels_x   = p[index_init                 :  index_init + N_pts]
        labels_y   = p[index_init + N_pts         :  index_init + 2*N_pts]
        labels_z   = p[index_init + 2*N_pts       :  index_init + 3*N_pts]
        labels_dxds      = p[index_init + 3*N_pts       :  index_init + 4*N_pts]
        labels_dyds      = p[index_init + 4*N_pts       :  index_init + 5*N_pts]
        labels_dzds      = p[index_init + 5*N_pts       :  index_init + 6*N_pts]


        return  labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds
            
    def unpack_parameters_CAMPCC(self, p):
        labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds = self.unpack_parameters_MPCC(p)

        index_init = index_init = self.n_base_params
        N_pts = self.n_points_kernelized

        labels_d2x_ds2 = p[index_init + 6*N_pts       :  index_init + 7*N_pts]
        labels_d2y_ds2 = p[index_init + 7*N_pts       :  index_init + 8*N_pts]
        labels_d2z_ds2 = p[index_init + 8*N_pts       :  index_init + 9*N_pts]
        labels_k = p[index_init + 9*N_pts       :  index_init + 10*N_pts] # kernelized path regression parameters

        return labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds, labels_d2x_ds2, labels_d2y_ds2, labels_d2z_ds2, labels_k


    def unpack_parameters_CAMPCC_EA(self, p):
        # # unpack base parameters    
        # q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        # terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # -----------------------------
        # Reference path info
        # -----------------------------
        index_init = self.n_base_params

        N_pts = self.n_points_kernelized
        labels_wz_ref   = p[index_init                 :  index_init + N_pts]
        labels_wx_ref  = p[index_init + N_pts         :  index_init + 2*N_pts]

        return  labels_wz_ref, labels_wx_ref

    def unpack_parameters_CAMPCC_EA_2(self, p):
        index_init = self.n_base_params
        N_pts = self.n_points_kernelized

        labels_x     = p[index_init                 :  index_init + N_pts]
        labels_y     = p[index_init + N_pts         :  index_init + 2*N_pts]
        labels_z     = p[index_init + 2*N_pts       :  index_init + 3*N_pts]

        labels_roll  = p[index_init + 3*N_pts       :  index_init + 4*N_pts]
        labels_pitch = p[index_init + 4*N_pts       :  index_init + 5*N_pts]
        labels_yaw   = p[index_init + 5*N_pts       :  index_init + 6*N_pts]

        labels_wz = p[index_init + 6*N_pts       :  index_init + 7*N_pts]

        return labels_x, labels_y, labels_z, labels_roll, labels_pitch, labels_yaw, labels_wz



    def unpack_parameters_MPCC(self, p):
        # unpack base parameters    
        # q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        # terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # -----------------------------
        # Reference path info
        # -----------------------------
        index_init = self.n_base_params

        N_pts = self.n_points_kernelized
        labels_x   = p[index_init                 :  index_init + N_pts]
        labels_y   = p[index_init + N_pts         :  index_init + 2*N_pts]
        labels_z   = p[index_init + 2*N_pts       :  index_init + 3*N_pts]
        labels_dxds      = p[index_init + 3*N_pts       :  index_init + 4*N_pts]
        labels_dyds      = p[index_init + 4*N_pts       :  index_init + 5*N_pts]
        labels_dzds      = p[index_init + 5*N_pts       :  index_init + 6*N_pts]


        return  labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds




    def continous_dynamics(self,roll_d, pitch_d, yaw_d, thrust, vx, vy, vz, roll, pitch, yaw,s_dot):

        # --- drone dynamics ---
        drone_state_dot = drone(roll_d, pitch_d, yaw_d, thrust, vx,vy,vz, roll, pitch, yaw)

        # concatenate the state derivatives and the path progress derivative
        state_dot = ca.vertcat(*drone_state_dot,s_dot)

        return state_dot
    
    def continous_dynamics_forcespro(self, x, u, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(x)
        if self.controller_type == 'CAMPCC_EA':
            ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(x)

        # denormalize inputs
        if self.controller_type != 'MPCCPP':
            roll_desired, pitch_desired, yaw_desired, thrust, slack = self.denormalize_u(u)
        else : # MPCCPP
            roll_desired, pitch_desired, yaw_desired, thrust, slack, s_dot = self.denormalize_u(u)
        
        # path parameters
        # stack x and u
        z = ca.vertcat(u, x)
        ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds, s_dot, ref_wz, ref_wx = self.evalaute_path_quantities_forcespro(z, p)

        # call the continous drone dynamics function
        state_dot = self.continous_dynamics(roll_desired, pitch_desired, yaw_desired, thrust, vx, vy, vz, roll, pitch, yaw, s_dot)

        if self.controller_type == 'CAMPCC_EA':
            ref_frame_dot = self.continuous_dynamics_reference_CAMPCC_EA(ref_roll, ref_pitch, ref_yaw, ref_wz, ref_wx, s_dot)
            state_dot = ca.vertcat(state_dot, ref_frame_dot)    


        return state_dot
        
    def continuous_dynamics_reference_CAMPCC_EA(self, ref_roll, ref_pitch, ref_yaw, ref_wz, ref_wx, s_dot):
        """
        Continuous dynamics of the reference frame (Z-Y-X Euler convention)
        Inputs:
            ref_roll, ref_pitch, ref_yaw : Euler angles of the reference frame
            ref_wz : rotation about body z-axis (yaw in osculating plane)
            ref_wx : rotation about body x-axis (roll of osculating plane)
            s_dot : forward speed along the reference frame x-axis
        Returns:
            dpx, dpy, dpz : derivatives of position in world frame
            droll, dpitch, dyaw : Euler-angle rates
        """
        # correct pitch to avoid singularities
        min_val = -np.pi/2 + 0.05
        max_val = np.pi/2 - 0.05
        ref_pitch = self.soft_clip(ref_pitch, min_val, max_val)

        # Simplified Euler-angle rates (wy = 0)
        droll  = s_dot * (ref_wx + ca.cos(ref_roll) * ca.tan(ref_pitch) * ref_wz )
        dpitch = s_dot * (-ca.sin(ref_roll) * ref_wz )
        dyaw   = s_dot * ((ca.cos(ref_roll)/ca.cos(ref_pitch)) * ref_wz )

        # Forward direction (body x-axis in world frame)
        fx = ca.cos(ref_pitch) * ca.cos(ref_yaw)
        fy = ca.cos(ref_pitch) * ca.sin(ref_yaw)
        fz = -ca.sin(ref_pitch)   # depends on convention

        # Position derivatives along body x-axis
        dpx = s_dot * fx
        dpy = s_dot * fy
        dpz = s_dot * fz

        return ca.vertcat(dpx, dpy, dpz, droll, dpitch, dyaw)



    def evalaute_path_quantities_forcespro(self, z, p):
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        if self.controller_type == 'CAMPCC_EA':
            ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(z[self.nu:])

        # unpack normalized inputs
        if self.controller_type == 'MPCCPP':
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack, s_dot = self.unpack_u(z[:self.nu])
        else:
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack = self.unpack_u(z[:self.nu])


        # --- unpack parameters ---
        # base parameters  
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # path parameters
        if self.controller_type == 'MPCC' or self.controller_type == 'MPCCPP': # they have the same parameters
            labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds = self.unpack_parameters_MPCC(p)
        elif self.controller_type == 'CAMPCC':
            labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds,\
            labels_d2x_ds2, labels_d2y_ds2, labels_d2z_ds2, labels_k = self.unpack_parameters_CAMPCC(p)
        elif self.controller_type == 'CAMPCC_EA':
            labels_wz_ref, labels_wx_ref = self.unpack_parameters_CAMPCC_EA(p)
        elif self.controller_type == 'CAMPCC_EA_2':
            labels_x, labels_y, labels_z, labels_roll, labels_pitch, labels_yaw, labels_wz = self.unpack_parameters_CAMPCC_EA_2(p)



        # --- evaluate path quantities ---
        if self.controller_type == 'MPCC' or self.controller_type == 'MPCCPP':
            ref_x = self.evaluate_kernelized_line_reg(s, local_path_length, labels_x)
            ref_y = self.evaluate_kernelized_line_reg(s, local_path_length, labels_y)
            ref_z = self.evaluate_kernelized_line_reg(s, local_path_length, labels_z)
            ref_dxds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dxds)
            ref_dyds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dyds)
            ref_dzds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dzds)
            ref_wz, ref_wx = [],[] # dummy values

        elif self.controller_type == 'CAMPCC':
            ref_x = self.evaluate_kernelized_line_reg(s, local_path_length, labels_x)
            ref_y = self.evaluate_kernelized_line_reg(s, local_path_length, labels_y)
            ref_z = self.evaluate_kernelized_line_reg(s, local_path_length, labels_z)
            ref_dxds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dxds)
            ref_dyds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dyds)
            ref_dzds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dzds)
            ref_d2x_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2x_ds2)
            ref_d2y_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2y_ds2)
            ref_d2z_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2z_ds2)
            k = self.evaluate_kernelized_line_reg(s, local_path_length, labels_k)
            ref_wz, ref_wx = [],[] # dummy values

        elif self.controller_type == 'CAMPCC_EA':
            ref_wz = self.evaluate_kernelized_line_reg(s, local_path_length, labels_wz_ref)
            ref_wx = self.evaluate_kernelized_line_reg(s, local_path_length, labels_wx_ref)

        elif self.controller_type == 'CAMPCC_EA_2':
            ref_x = self.evaluate_kernelized_line_reg(s, local_path_length, labels_x)
            ref_y = self.evaluate_kernelized_line_reg(s, local_path_length, labels_y)
            ref_z = self.evaluate_kernelized_line_reg(s, local_path_length, labels_z)
            ref_roll = self.evaluate_kernelized_line_reg(s, local_path_length, labels_roll)
            ref_pitch = self.evaluate_kernelized_line_reg(s, local_path_length, labels_pitch)
            ref_yaw = self.evaluate_kernelized_line_reg(s, local_path_length, labels_yaw)
            ref_wz = self.evaluate_kernelized_line_reg(s, local_path_length, labels_wz)
            ref_wx = [] # dummy value



        # define s_dot 
        if self.controller_type == 'MPCC':
            s_dot = np.sqrt(vx**2 + vy**2 + vz**2 + 0.001)  # path progress rate is the speed of the drone

        elif self.controller_type == 'MPCCPP':
            # s_dot is already provided as an input
            pass

        elif self.controller_type == 'CAMPCC':
            # define s_dot
            s_dot = self.s_dot_CAMPCC(pos_x, pos_y, pos_z, vx, vy, vz,
                                        ref_x, ref_y, ref_z, 
                                        ref_d2x_ds2, ref_d2y_ds2, ref_d2z_ds2, ref_dxds, ref_dyds, ref_dzds,k) # ref_dxds, ref_dyds, ref_dzds,       

        elif self.controller_type == 'CAMPCC_EA' or self.controller_type == 'CAMPCC_EA_2':
            # evaluate the reference frame axes
            x_axis_ref, y_axis_ref = self.CAMPCC_AE_xy_axis(ref_roll, ref_pitch, ref_yaw)
            # evalaute s_dot
            s_dot = self.s_dot_CAMPCC_EA(   pos_x, pos_y, pos_z,
                                            vx,vy,vz,
                                            ref_x, ref_y, ref_z,
                                            x_axis_ref, y_axis_ref,
                                            ref_wz) 
            # split up x axis ref to match the objective function inputs
            ref_dxds = x_axis_ref[0]
            ref_dyds = x_axis_ref[1]
            ref_dzds = x_axis_ref[2]
        

        return ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds, s_dot, ref_wz, ref_wx
    

    def slack_cost(self,slack):
        return 10000 * slack**2 + 1000 * slack
    

    def objective(self,    roll_desired, pitch_desired, yaw_desired, thrust,
                                pos_x, pos_y, pos_z, yaw, s_dot,
                                ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds,
                                q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw,
                                lane_radius, slack):

        # Error vectors
        err_vec = ca.vertcat(pos_x - ref_x, pos_y - ref_y, pos_z - ref_z)
        tan_vec = ca.vertcat(ref_dxds, ref_dyds, ref_dzds)

        # Lag error
        if self.controller_type == "CAMPCC" or self.controller_type == "CAMPCC_AE":
            lag_error = 0
            cont_error_sqrd = ca.sumsqr(err_vec)
        else:
            lag_error = ca.dot(err_vec, tan_vec)
            # Contour error
            proj_error_on_tan = tan_vec * lag_error
            cont_error_vector = err_vec - proj_error_on_tan
            cont_error_sqrd = ca.sumsqr(cont_error_vector)

        j = - q_sdot * s_dot \
            + q_cont * cont_error_sqrd/lane_radius**2 \
            + q_lag * lag_error**2 \
            + q_roll_pitch * (roll_desired-0.5) ** 2 \
            + q_roll_pitch * (pitch_desired-0.5) ** 2 \
            + q_yaw * (yaw_desired-0.5) ** 2 \
            + q_thrust * (thrust-0.5) ** 2 \
            + self.slack_cost(slack)
        return j
    



    def objective_forcespro(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        if self.controller_type == 'CAMPCC_EA':
            ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(z[self.nu:])

        # unpack normalized inputs
        if self.controller_type == 'MPCCPP':
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack, s_dot = self.unpack_u(z[:self.nu])
        else:
            roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, slack = self.unpack_u(z[:self.nu])

        # --- unpack parameters ---
        # base parameters  
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # path parameters
        ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds, s_dot, ref_wz, ref_wx = self.evalaute_path_quantities_forcespro(z, p)
            

        j = self.objective( roll_desired_norm, pitch_desired_norm, yaw_desired_norm, thrust_norm, 
                                                                    pos_x, pos_y, pos_z, yaw ,s_dot,
                                                                    ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds,
                                                                    q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw,
                                                                    lane_radius, slack)
        return j



    def objective_terminal_cost_MPCC(self,pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, ref_dxds,ref_dyds,ref_dzds, vx, vy, vz ,qt_pos, qt_v,qt_s, s, local_path_length):
        # Vector form of position error
        err_vec = ca.vertcat(pos_x - ref_x, pos_y - ref_y, pos_z - ref_z)

        # Squared Euclidean distance (contour error)
        err_squared = ca.sumsqr(err_vec)

        # allignment error as dot prod between velocity and tangent vector
        tan_vec = ca.vertcat(ref_dxds, ref_dyds, ref_dzds) / (ca.sqrt(ca.sumsqr(ref_dxds) + ca.sumsqr(ref_dyds) + ca.sumsqr(ref_dzds)) )  # normalize the tangent vector
        v_vec = ca.vertcat(vx, vy, vz)
        dot_prod = ca.dot(v_vec, tan_vec)
        #v_norm = ca.sqrt(ca.sumsqr(v_vec) + 0.001)  # add small value to avoid division by zero


        # Terminal cost
        j = qt_pos * err_squared +\
            qt_v * (1 - dot_prod)**2 +\
            qt_s * (1 - s/local_path_length)**2

        return j

    def objective_terminal_cost_MPCC_forcespro(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        #unpack parameters
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos,qt_v ,qt_s, lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz, \
        labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds =\
        self.unpack_parameters_MPCC(p)

        # evaluate path quantities
        ref_x = self.evaluate_kernelized_line_reg(s, local_path_length, labels_x)
        ref_y = self.evaluate_kernelized_line_reg(s, local_path_length, labels_y)
        ref_z = self.evaluate_kernelized_line_reg(s, local_path_length, labels_z)
        # evaluate tangent vectors
        ref_dxds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dxds)
        ref_dyds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dyds)
        ref_dzds = self.evaluate_kernelized_line_reg(s, local_path_length, labels_dzds)

        j = self.objective_terminal_cost_MPCC(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, ref_dxds,ref_dyds,ref_dzds, vx, vy, vz ,qt_pos,qt_v ,qt_s, s, local_path_length)

        return j
    
    def objective_terminal_cost_on_reference(self,pos_x, pos_y, pos_z, vx, vy, vz,\
                                                terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz,\
                                                qt_pos, qt_v,\
                                                qt_s, s, local_path_length):
        # Vector form of position error
        err_vec_pos = ca.vertcat(pos_x - terminal_x, pos_y - terminal_y, pos_z - terminal_z)
        err_vec_velocities = ca.vertcat(vx - terminal_vx, vy - terminal_vy, vz - terminal_vz)

        # Squared Euclidean distance 
        err_squared_pos = ca.sumsqr(err_vec_pos)
        error_squared_v = ca.sumsqr(err_vec_velocities)

        # Terminal cost
        j = qt_pos * err_squared_pos \
            + qt_v * error_squared_v \
            - qt_s * ((s/local_path_length)**2+s/local_path_length)
        return j


    def objective_terminal_cost_forcespro_on_reference(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        #unpack parameters
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v, qt_s, lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz, \
        labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds =\
        self.unpack_parameters_MPCC(p)



        j = self.objective_terminal_cost_on_reference(pos_x, pos_y, pos_z, vx, vy, vz,\
                                                terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz,\
                                                qt_pos, qt_v,\
                                                qt_s, s, local_path_length)

        return j


    def objective_terminal_cost_on_path(self,pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, qt_pos,qt_s, s, local_path_length):
        # Vector form of position error
        err_vec = ca.vertcat(pos_x - ref_x, pos_y - ref_y, pos_z - ref_z)

        # Squared Euclidean distance (contour error)
        err_squared = ca.sumsqr(err_vec)


        # Terminal cost
        # j = qt_pos * err_squared - qt_s * ((s/local_path_length)**2+s/local_path_length)
        j = qt_pos * err_squared - qt_s * s
        return j

    def objective_terminal_cost_forcespro_on_path(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        if self.controller_type == 'CAMPCC_EA':
            ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(z[self.nu:])

        #unpack parameters
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v,qt_s, lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # evalaute path quantities using function
        ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds, s_dot, ref_wz, ref_wx = self.evalaute_path_quantities_forcespro(z, p)


        j = self.objective_terminal_cost_on_path(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, qt_pos,qt_s, s, local_path_length)

        if self.controller_type == "CAMPCC":
            labels_x, labels_y, labels_z, labels_dxds, labels_dyds, labels_dzds,\
            labels_d2x_ds2, labels_d2y_ds2, labels_d2z_ds2, labels_k = self.unpack_parameters_CAMPCC(p)
            ref_d2x_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2x_ds2)
            ref_d2y_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2y_ds2)
            ref_d2z_ds2 = self.evaluate_kernelized_line_reg(s, local_path_length, labels_d2z_ds2)
            k = self.evaluate_kernelized_line_reg(s, local_path_length, labels_k)
            p_projected_r = (pos_x - ref_x) * ref_d2x_ds2 + (pos_y - ref_y) * ref_d2y_ds2 + (pos_z - ref_z) * ref_d2z_ds2
            denominator = (1 - p_projected_r * k)
            denominator_corrected = self.soft_min(denominator, 0.1)
            #projection_ratio = 1 / denominator_corrected
            #j = - qt_s * ((s/local_path_length)**2 + s/local_path_length) + qt_v * denominator**2
            #j = - qt_s * (s - 0.9 * denominator)/local_path_length  
            j = j + qt_v * denominator_corrected**2
            

        return j



    def s_dot_CAMPCC(self,pos_x, pos_y, pos_z,vx,vy,vz,ref_x, ref_y, ref_z,ref_d2x_ds2,ref_d2y_ds2,ref_d2z_ds2,ref_dxds,ref_dyds,ref_dzds,k): 

        # projection ratio 
        p_projected_r = (pos_x - ref_x) * ref_d2x_ds2 + (pos_y - ref_y) * ref_d2y_ds2 + (pos_z - ref_z) * ref_d2z_ds2
        denominator = (1 - p_projected_r * k)
        #denominator_corrected = denominator + np.exp(-10*denominator) # this will stop the denominator from going to 0 but will be very similar elswere
        denominator_corrected = self.soft_min(denominator, 0.33)
        #denominator_corrected = denominator + 0.167*ca.exp(-100*(denominator+0.11)**2)  # this will stop the denominator from going to 0 but will be very similar elswere
        #denominator_corrected = 0.01 + (np.tanh(10*(denominator)))*(denominator)
        #denominator_corrected = ca.fmax(denominator, 0.1)  # this will stop the denominator from going to 0 but will be very similar elswere
        projection_ratio = 1 / denominator_corrected

        # formulation without 1st order derivatives (rewriting of vt)
        # vn_sqrd = (vx*ref_d2x_ds2 + vy*ref_d2y_ds2 + vz*ref_d2z_ds2)**2 / (ref_d2x_ds2**2+ref_d2y_ds2**2 + ref_d2z_ds2**2+0.001)  # squared normal velocity
        # vt = np.sqrt((vx**2+ vy**2 + vz**2 + 0.0001)- vn_sqrd)  # add small value to avoid division by zero

        # projection on the tangent vector
        vt = ca.dot(ca.vertcat(vx, vy, vz), ca.vertcat(ref_dxds, ref_dyds, ref_dzds))

        s_dot = vt * projection_ratio  # s_dot is the tangential velocity times the projection ratio

        return s_dot
    
    def CAMPCC_AE_xy_axis(self, ref_roll, ref_pitch, ref_yaw):
        # ----- x-axis (forward / tangent) -----
        xx = ca.cos(ref_pitch) * ca.cos(ref_yaw)
        xy = ca.cos(ref_pitch) * ca.sin(ref_yaw)
        xz = -ca.sin(ref_pitch)
        x_axis_ref = ca.vertcat(xx, xy, xz)

        # ----- y-axis (lateral) This lies on the osculating plane -----
        # NOTE this is not the normal vector of the curve, as it may be pointing out 
        yx = -ca.cos(ref_yaw)*ca.sin(ref_roll) - ca.sin(ref_yaw)*ca.sin(ref_pitch)*ca.cos(ref_roll)
        yy = -ca.sin(ref_yaw)*ca.sin(ref_roll) + ca.cos(ref_yaw)*ca.sin(ref_pitch)*ca.cos(ref_roll)
        yz = ca.cos(ref_pitch)*ca.cos(ref_roll)
        y_axis_ref = ca.vertcat(yx, yy, yz)
        return x_axis_ref, y_axis_ref
    
    def s_dot_CAMPCC_EA(self,pos_x, pos_y, pos_z,vx,vy,vz,
                        ref_x, ref_y, ref_z,
                        x_axis_ref, y_axis_ref,
                        ref_wz): 
        # NOTE: the trick here is that ref_wz is actually the SIGNED curvature k according to the current frame

        # velocity vector  (drone state)
        vel_vec = ca.vertcat(vx, vy, vz)

        # project velocity on the tangent vector
        vt = ca.dot(vel_vec, x_axis_ref)

        # project the error vector on the lateral vector to get the "p" value
        error_vec = ca.vertcat(pos_x - ref_x, pos_y - ref_y, pos_z - ref_z)
        p = ca.dot(error_vec, y_axis_ref)
        den_corrected = self.soft_min(1-p*ref_wz,0.33)
        projection_ratio = 1 / den_corrected # (1-p*ref_wz) #
        
        
        s_dot = vt * projection_ratio


        # this is what we were doing for the 2D case (kept here as a reminder)
        # # # s_dot definition depending on the selected algorithm
        # # # for now we assume vy is small
        # # v_tan = vx * cos(yaw - ref_heading)
        # # p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -cos(ref_heading)
        # # den_corrected = self.soft_min(1+p*k,0.3)
        # # projection_ratio = 1 / den_corrected
        # # s_dot = v_tan * projection_ratio

        return s_dot




    def soft_min(self, x, min_val):
        sharpness = 10
        return min_val + 0.5*(1 + np.tanh(sharpness*(x-min_val)))*(x-min_val)
    
    def soft_max(self, x, max_val):
        """
        Smooth approximation of min(x, max_val).
        """
        sharpness = 10
        return max_val + 0.5 * (1 + np.tanh(sharpness * (-x + max_val))) * (x - max_val )

    def soft_clip(self, x, min_val, max_val):
        """
        Smooth approximation of clipping x between [min_val, max_val].
        """
        return self.soft_max(self.soft_min(x, min_val), max_val)
    

    def lane_boundary_constraint_tube(self,pos_x,pos_y,pos_z,ref_x,ref_y,ref_z,lane_radius,slack):
        error_sqrd = (pos_x - ref_x)**2 + (pos_y - ref_y)**2 + (pos_z - ref_z)**2 + 0.001  # distance from the reference path
        return  lane_radius - error_sqrd**0.5 + slack # this is the constraint that the drone should stay within the lane width, slack is a slack variable that allows the drone to go outside the lane if needed
    


    def lane_boundary_constraints_square_gate(self,
        pos_x, pos_y, pos_z,
        ref_x, ref_y, ref_z,
        dxds, dyds,  # normalized path direction
        lane_radius
    ):
        # 1. Perpendicular vector to the path direction in xy-plane
        nx = -dyds
        ny =  dxds

        # 2. Displacement in xy-plane
        dx = pos_x - ref_x
        dy = pos_y - ref_y

        # 3. Lateral offset in gate plane (perpendicular to path direction)
        lateral_offset = nx * dx + ny * dy

        # 4. Vertical offset from gate center
        vertical_offset = pos_z - ref_z

        # 5. Constraints: must stay within square gate bounds (all ≥ 0)
        c1 = lane_radius - lateral_offset
        c2 = lane_radius + lateral_offset
        c3 = lane_radius - vertical_offset
        c4 = lane_radius + vertical_offset

        return ca.vertcat(c1, c2, c3, c4)

    
    def lane_boundary_constraint_tube_forcespro(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        if self.controller_type == 'CAMPCC_EA':
            ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(z[self.nu:])
        # unpack inputs
        #roll_desired, pitch_desired, yaw_desired, thrust, slack = self.unpack_u(z[:self.nu])
        slack = z[4] # slack is the 5th element in the input vector in all controller settings

        # unpack base parameters
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v ,qt_s,lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz = self.unpack_base_parameters(p)

        # evaluate path quantities using function
        ref_x, ref_y, ref_z, ref_dxds, ref_dyds, ref_dzds, s_dot, ref_wz, ref_wx = self.evalaute_path_quantities_forcespro(z, p)

        h = self.lane_boundary_constraint_tube(     pos_x,pos_y,pos_z,
                                                    ref_x,ref_y,ref_z,
                                                    lane_radius,slack)
        return [h]


    def lane_boundary_constraint_tube_CAMPCC_EA_forcespro(self, z, p):
        # unpack state
        pos_x, pos_y, pos_z, vx, vy, vz, roll, pitch, yaw, s = self.unpack_x(z[self.nu:])
        ref_x, ref_y, ref_z, ref_roll, ref_pitch, ref_yaw = self.unpack_x_CAMPCC_EA(z[self.nu:])

        # unpack inputs
        #roll_desired, pitch_desired, yaw_desired, thrust, slack = self.unpack_u(z[:self.nu])
        slack = z[4] # slack is the 5th element in the input vector in all controller settings

        #unpack parameters   (this works in all cases because the CAMPCC just has a few extra parameters at the end for the second order derivatives)
        q_sdot, q_cont, q_lag, q_thrust, q_roll_pitch, q_yaw, qt_pos, qt_v,qt_s, lane_radius, local_path_length,\
        terminal_x, terminal_y, terminal_z, terminal_vx, terminal_vy, terminal_vz, \
        labels_wz_ref, labels_wx_ref =\
        self.unpack_parameters_CAMPCC_EA(p)

        h = self.lane_boundary_constraint_tube(     pos_x,pos_y,pos_z,
                                                    ref_x,ref_y,ref_z,
                                                    lane_radius,slack)
        return [h]



    def produce_X0(self,labels_s,labels_x,labels_y,labels_z):
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))

        # Index mapping for z:
        # z[ 0] = roll_desired
        # z[ 1] = pitch_desired
        # z[ 2] = yaw_desired
        # z[ 3] = thrust
        # z[ 4] = pos_x
        # z[ 5] = pos_y
        # z[ 6] = pos_z
        # z[ 7] = vx
        # z[ 8] = vy
        # z[9] = vz
        # z[10] = roll
        # z[11] = pitch
        # z[12] = yaw
        # z[13] = s

    
        # assign initial guess for the states by forward euler integration on th ereference path

        s_0_vec = np.linspace(0, self.time_horizon*5, self.N+1)

        # interpolate to get curvature values
        #normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)
        x_ref_0 = np.interp(s_0_vec, labels_s, labels_x)
        y_ref_0 = np.interp(s_0_vec, labels_s, labels_y)
        z_ref_0 = np.interp(s_0_vec, labels_s, labels_z)

        # set inputs to zero
        X0_array[:,0] = 0.5
        X0_array[:,1] = 0.5
        X0_array[:,2] = 0.5
        X0_array[:,3] = 0.5

        # set ref trajectory to the initial guess
        X0_array[:,4] = x_ref_0
        X0_array[:,5] = y_ref_0
        X0_array[:,6] = z_ref_0
        X0_array[:,13] = s_0_vec

        return X0_array
    
    def produce_X0_from_time_optimal_trajectory(self, time_optimal_trajectory, start_index, xyz_closest_point_on_ref_path,
                                                s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path ,
                                                roll_4_local_path, pitch_4_local_path ,yaw_4_local_path):
        # time_optimal_trajectory is ordered as follows:
        # Index : Description
        #   0   : desired_roll
        #   1   : desired_pitch
        #   2   : desired_yaw
        #   3   : thrust
        #   4   : pos_x
        #   5   : pos_y
        #   6   : pos_z
        #   7   : vx
        #   8   : vy
        #   9   : vz
        #  10   : roll
        #  11   : pitch
        #  12   : yaw
        #  13   : s         (arc-length or path parameter)
        #  14   : time



        # get the corresponding time index
        t_0 = time_optimal_trajectory[start_index, -1]
        t_1 = t_0 + self.time_horizon

        # find the index of the closest time to t_0 + time horizon
        index_t_finish = np.argmin(np.abs(time_optimal_trajectory[:,-1] - t_1))

        # extract that trajectory segment
        time_optimal_trajectory_slice = time_optimal_trajectory[start_index:index_t_finish+1,:]
        # linearly interpolate to have the right number of stages for the solver
        time_vec_0 = np.linspace(t_0, t_1, self.N+1)

        X0_interpolated = np.zeros((self.N+1, self.nu + self.nx))

        # assign the interpolated inputs (this avoids changing the slack variable)
        for i in range(4): # skip the slack variable (keep it at 0)
            X0_interpolated[:, i] = np.interp(time_vec_0, time_optimal_trajectory_slice[:, 14], time_optimal_trajectory_slice[:, i])
        
        # assign the interpolated inputs
        if self.controller_type == 'MPCCPP':
            index_shift = 2 # for MPCCPP we have an extra input s_dot
        else:
            index_shift = 1

        for i in range(4,14): # skip the slack variable (keep it at 0)
            X0_interpolated[:, i + index_shift] = np.interp(time_vec_0, time_optimal_trajectory_slice[:, 14], time_optimal_trajectory_slice[:, i])
        
        
        # !!!! X0 has the extra slack variable !!!!!  , so we need to shift the indices
        # subtract intial s for solver (it's always the last value)
        X0_interpolated[:, -1] -= X0_interpolated[0, -1]

        # rescale the inputs to be between 0 and 1
        for i in range(self.N+1):
            X0_interpolated[i, :4] = self.normalize_u(X0_interpolated[i, :4]) # only drone inputs are normalized, slack and s_dot are notis not

        #produce position relative to the closest point on the reference path  NOTE remember to shift for the slack variable
        X0_interpolated[:, self.nu] -= xyz_closest_point_on_ref_path[0]
        X0_interpolated[:, self.nu+1] -= xyz_closest_point_on_ref_path[1]
        X0_interpolated[:, self.nu+2] -= xyz_closest_point_on_ref_path[2]

        # add s_dot guess if using MPCCPP
        if  self.controller_type == 'MPCCPP':
            # calculate s_dot as the speed of the drone
            dt = self.time_horizon / self.N  # time step
            s_dot_guess = np.diff(X0_interpolated[:,-1]) / dt
            X0_interpolated[:, self.nu-1] = [*s_dot_guess, s_dot_guess[-1]]

        # add reference path states if using CAMPCC_EA
        if  self.controller_type == 'CAMPCC_EA':
            # extract the s values from X0 (10th position after the inputs)
            s_X0 = X0_interpolated[:, self.nu + 9] 
            # # get the values of s init and s finish
            # s_init = time_optimal_trajectory_slice[0, 13]
            # s_finish = time_optimal_trajectory_slice[-1, 13]
            # find the corresponding indices on the reference path
            start_index_path = np.argmin(np.abs(s_4_local_path - s_X0[0]))
            end_index_path = np.argmin(np.abs(s_4_local_path - s_X0[-1]))+1
            s_vec_path = s_4_local_path[start_index_path:end_index_path]
            x_vec_path = x_4_local_path[start_index_path:end_index_path]
            y_vec_path = y_4_local_path[start_index_path:end_index_path]
            z_vec_path = z_4_local_path[start_index_path:end_index_path]
            roll_vec_path = roll_4_local_path[start_index_path:end_index_path]
            pitch_vec_path = pitch_4_local_path[start_index_path:end_index_path]
            yaw_vec_path = yaw_4_local_path[start_index_path:end_index_path]


            X0_interpolated[:,self.nu-1+10] = np.interp(s_X0,s_vec_path, x_vec_path) # x values
            X0_interpolated[:,self.nu-1+11] = np.interp(s_X0,s_vec_path, y_vec_path) # y values
            X0_interpolated[:,self.nu-1+12] = np.interp(s_X0,s_vec_path, z_vec_path) # z values
            X0_interpolated[:,self.nu-1+13] = np.interp(s_X0,s_vec_path, roll_vec_path) # roll values
            X0_interpolated[:,self.nu-1+14] = np.interp(s_X0,s_vec_path, pitch_vec_path) # pitch values
            X0_interpolated[:,self.nu-1+15] = np.interp(s_X0,s_vec_path, yaw_vec_path) # yaw values

            # print(s_X0)
            # print(s_vec_path)
            # print(x_vec_path)
            # print(ref_path_X0[:,0])

            # stack horizontally
            #print(X0_interpolated.shape)
            #X0_interpolated = np.hstack((X0_interpolated, ref_path_X0))

            #print(time_optimal_trajectory.shape)

            
        return X0_interpolated


    # def set_up_solver_4_MPC_loop(self, solver, xinit, params_i, X0=[]):


    #     # assign initial state
    #     solver.set(0, "lbx", xinit)
    #     solver.set(0, "ubx", xinit)

    #     # assign parameers
    #     for i in range(self.N+1):
    #         solver.set(i, "p", params_i)

    #     # assign frist guess if supplied
    #     if len(X0) > 0:
    #         for i in range(self.N):
    #             solver.set(i, "u", X0[i, :self.nu])
    #             solver.set(i, "x", X0[i, self.nu:])
    #         solver.set(self.N, "x", X0[self.N, self.nu:])

    #     return solver

    def solve_mpc(self,software_choice, solver, xinit, params_i, X0=[]):

        # assign the runtime parameters
        if software_choice == "acados":
            # assign frist guess if supplied
            if len(X0) > 0:
                for i in range(self.N):
                    solver.set(i, "u", X0[i, :self.nu])
                    solver.set(i, "x", X0[i, self.nu:])

                solver.set(self.N, "x", X0[self.N, self.nu:])

            # assign initial state
            solver.set(0, "lbx", xinit)
            solver.set(0, "ubx", xinit)

            # assign parameers
            for i in range(self.N+1):
                solver.set(i, "p", params_i)

            # ---- solve the problem ----
            #solver.set('step_length', 0.1)  # Set step length for the solver, if needed
            status = solver.solve()
            if status != 0:
                print(f"Solver failed with status {status}")
                self.converged = False
            else:
                self.converged = True
            # # Retrieve solver time (in seconds)
            # solve_time = solver.get_stats('time_tot')
            # print(f"Solve time: {solve_time:.6f} seconds")

            # print('---')
            # solver.print_statistics()

            # ---- extract the solution ----
            state_traj = np.zeros((self.N + 1, self.nx))  
            input_traj = np.zeros((self.N, self.nu))      

            for i in range(self.N + 1):
                state_traj[i, :] = solver.get(i, "x")

                if i < self.N:
                    input_traj[i, :] = self.denormalize_u(solver.get(i, "u"))

        elif software_choice == "forcespro":

            # produce problem as a dictionary for forces
            param_array = np.tile(params_i, (self.N + 1, 1)).ravel()
            problem = {"xinit": xinit, "all_parameters": param_array}
            if len(X0) > 0:
                # if initial guess is provided, add it to the problem
                problem["x0"] = X0

            # # --- solve the problem ---
            # if reinitialize_solver:
            #     solver.xopt = None          # forget previous primal variables
            #     solver.last_problem = None  # optional, depending on version
            # else:
            #     pass


            output, exitflag, info = solver.solve(problem)
            

            if exitflag != 1:
                print(f"Solver failed with status {exitflag}")
                self.converged = False
            else:
                self.converged = True

            output_array = np.array(list(output.values()))

            state_traj = output_array[:,self.nu :]
            # denormalize input trajectory and extract
            input_traj = np.zeros((self.N, self.nu))
            for i in range(self.N):
                input_traj[i, :] = self.denormalize_u(output_array[i, :self.nu])
        
            

        return state_traj, input_traj
