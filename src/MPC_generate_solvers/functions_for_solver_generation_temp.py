import numpy as np
try:
    from .path_track_definitions import K_RBF_kernel, K_matern2_kernel
except:
    from path_track_definitions import K_RBF_kernel, K_matern2_kernel

import casadi

# This assumes that the DART system identification package is installed
from DART_dynamic_models.dart_dynamic_models import model_functions 





class generate_high_level_MPCC_PP(): # inherits from DART system identification

    def __init__(self,MPC_algorithm):
        
        self.MPC_algorithm = MPC_algorithm
        self.solver_name_acados = 'high_level_acados_MPCC_PP_' + MPC_algorithm
        self.solver_name_forces = 'high_level_forces_MPCC_PP_' + MPC_algorithm

        self.n_points_kernelized = 81 # number of points in the kernelized path (41 for reference)
        self.time_horizon = 1.5
        self.N = 30 # stages
        self.max_yaw_rate = 10 # based on w = V / R = k * V so 2 * V is the maximum yaw rate 
        self.nu = 3
        self.nx = 4
        self.n_parameters = 9 + 3 * self.n_points_kernelized
        self.n_inequality_constraints = 1

        # generate fixed path quantities
        #n points kernelized is the number of points used to kernelize the path but defined above to get the nparamters right
        try:
            from path_track_definitions import generate_fixed_path_quantities
        except:
            from .path_track_definitions import generate_fixed_path_quantities
        self.path_lengthscale = 1.3/self.n_points_kernelized
        lambda_val = 0.0001**2
        self.Kxx_inv, self.normalized_s_4_kernel_path = generate_fixed_path_quantities(self.path_lengthscale,
                                                                            lambda_val,
                                                                            self.n_points_kernelized)

        

    def produce_ocp(self):
        from casadi import vertcat, MX
        from acados_template import  AcadosOcp, AcadosModel


        # seting up ACADOS solver
        x = MX.sym('x', self.nx)
        u = MX.sym('u', self.nu)
        p = MX.sym('p', self.n_parameters) # stage-wise parameters


        # Create model object
        model = AcadosModel()
        model.name = self.solver_name_acados
        model.x = x
        model.u = u
        model.p = p

        #unpack states
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(vertcat(model.u,model.x))

        # unpack parameters
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(model.p)

        # assign dynamic constraint
        model.f_expl_expr = vertcat(*self.high_level_planner_continous_dynamics(V_target,u_yaw_dot,yaw,s_dot)) # make into vertical vector for casadi


        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(pos_x,pos_y,u_yaw_dot,slack,q_con,q_lag,q_u,s,local_path_length,labels_x, labels_y, labels_heading) 
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(yaw,pos_x,pos_y,qt_pos,qt_rot,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading)
                
        # constraints
        #ocp.constraints.constr_type = 'BGH'
        ocp.constraints.idxbu = np.array([0,1,2])
        ocp.constraints.lbu = np.array([-self.max_yaw_rate,0,0])
        ocp.constraints.ubu = np.array([+self.max_yaw_rate,100,1000]) 

        # define lane boundary constraints
        ocp.model.con_h_expr = self.lane_boundary_constraint(pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y)  # Define h(x, u)
        ocp.constraints.lh = np.array([0.0])  # Lower bound (h_min)
        ocp.constraints.uh = np.array([1000])  # Upper bound (h_max)

        # Initial state constraint
        ocp.constraints.x0 = np.zeros(self.nx)  # This is a default value, it will be updated at runtime

        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # GAUSS_NEWTON, EXACT
        ocp.solver_options.integrator_type = 'ERK' # IRK, ERK
        ocp.solver_options.sim_method_num_steps = 1  # Number of sub-steps in each interval for integration purpouses
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP   SQP_RTI
        ocp.solver_options.tf = self.time_horizon  # time horizon in seconds

        # messing with the convergence criteria
        ocp.solver_options.qp_solver_warm_start = 1 # 0: no warm start, 1: warm start 2 : hot start
        ocp.solver_options.globalization = 'FIXED_STEP' # 'MERIT_BACKTRACKING', 'FIXED_STEP' # fixed is the default
        #ocp.solver_options.nlp_solver_max_iter = 20  # Maximum SQP iterations
        #ocp.solver_options.qp_solver_iter_max = 5
        ocp.solver_options.print_level = 0 # no print
        #ocp.solver_options.tol = 0.001

        # Initialize parameters with default values (this step is important to avoid dimension mismatch)
        ocp.parameter_values = np.zeros(self.n_parameters)
        return ocp

    def produce_FORCES_model_codeoptions(self):
        import forcespro.nlp

        model = forcespro.nlp.SymbolicModel(self.N+1) # this plus one is to keep the same output dimensions as the acados model that has 1 extra state

        model.xinitidx = np.array(range(self.nu,self.nu + self.nx))  # variables in these positions are affected by initial state constraint. (I.e. they cannot change in the first stage)
        
        #theese parameters are the same for all the solvers
        model.nvar = self.nu + self.nx         # number of stage variables
        model.neq = self.nx                    # number of equality constraints (dynamic model)
        model.npar = self.n_parameters         # number of parameters
        model.E = np.concatenate([np.zeros((self.nx, self.nu)), np.eye(self.nx)], axis=1)  # This extraction matrix tells forces what variables are states and what are inputs

        # set fixed input bounds since they will not change at runtime
        # generate inf upper and lower bounds for the inputs and states
                            #  u_yaw_dot,       slack, s_dot, pos_x, pos_y,yaw,   s,  ref_x, ref_y, ref_heading
        model.lb = np.array([-self.max_yaw_rate, 0.0 ,   0.0, -1000, -1000,-1000,-0.2,-1000,-1000,-1000])  # lower bound on inputs
        model.ub = np.array([+self.max_yaw_rate, +100, +1000, +1000, +1000,+1000,+1000,+1000,+1000,+1000])  # upper bound on inputs

        # Set objective
        for i in range(self.N):
            model.objective[i] = self.objective_forces  # eval_obj is a Python function
        model.objective[self.N] = self.objective_terminal_forces
  
        # Set dynamic constraint
        model.continuous_dynamics = self.high_level_planner_continous_dynamics_forces

        # Set non linear constraints
        model.nh = self.n_inequality_constraints
        model.ineq = self.lane_boundary_constraint_forces
        model.hl = np.array([0.0])
        model.hu = np.array([1000.0])  # upper bound on inequality constraints
        


        # Define solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver') #get standard options
        # continuous dynamics options
        codeoptions.nlp.integrator.type = 'ERK4'
        codeoptions.nlp.integrator.Ts = self.time_horizon / (self.N+1)
        codeoptions.nlp.integrator.nodes = 1 # intermediate nodes for the integrator

        codeoptions.name = self.solver_name_forces
        codeoptions.printlevel = 0  #  1: summary line after each solve,   0: no prit
        codeoptions.BuildSimulinkBlock = 0  # disable simulink block generation because we don't need it
        codeoptions.maxit = 200  # maximum iterations
        codeoptions.noVariableElimination = 1  # enable or disable variable simplification (like if first stage is constrained)
        codeoptions.nlp.stack_parambounds = True  # determines if the parameters can simply be stacked (but not sure exactly what it does)

        # set tolerances
        codeoptions.nlp.TolStat = 1e-6  # inf norm tol. on stationarity
        codeoptions.nlp.TolEq = 1e-5  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1e-5  # tol. on inequality constraints
        codeoptions.nlp.TolComp = 1e-5  # tol. on complementarity

        # set warm start behaviour for dual variables (so always warm start from solver perspective, even if in practice you give it a vector of zeros)
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm

        #set overwrite behviour
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask

        codeoptions.solvemethod = 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        # NOTE that by default the solver uses a single sqp iteration so you need to increase the number of iterations
        codeoptions.sqp_nlp.maxqps = 3
        codeoptions.sqp_nlp.reg_hessian = 1e-5  # regularization of hessian (default is 5 * 10^(-9))
        # codeoptions.sqp_nlp.rti = 1
        # codeoptions.sqp_nlp.maxSQPit = 5


        return model,codeoptions

    def unpack_state(self,z):
        # control inputs
        u_yaw_dot = z[0]
        slack = z[1]
        s_dot = z[2]
        # state variables
        pos_x = z[2]
        pos_y = z[3]
        yaw =   z[4]
        s =     z[5]
        return u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s

    def unpack_parameters(self,p):
        V_target = p[0]
        local_path_length = p[1]
        q_con = p[2]
        q_lag = p[3]
        q_u = p[4]
        qt_pos = p[5]
        qt_rot = p[6]
        lane_width = p[7]
        qt_s_high = p[8]
        # prepare the kernelized path labels
        idx_start = 9
        idx_x_end = idx_start + self.n_points_kernelized
        idx_y_end = idx_x_end + self.n_points_kernelized
        idx_heading_end = idx_y_end + self.n_points_kernelized
        labels_x = p[idx_start:idx_x_end]
        labels_y = p[idx_x_end:idx_y_end]
        labels_heading = p[idx_y_end:idx_heading_end]
        return V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading
    

    def objective(self,pos_x,pos_y,u_yaw_dot,slack,q_con,q_lag,q_u,s,local_path_length,labels_x, labels_y, labels_heading):
        # produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y
        ref_heading = left_side @ labels_heading

        # stage cost
        err_lag_squared = ((pos_x - ref_x) *  np.cos(ref_heading)  + (pos_y - ref_y) * np.sin(ref_heading)) ** 2
        err_lat_squared = ((pos_x - ref_x) * -np.sin(ref_heading) + (pos_y - ref_y) * np.cos(ref_heading)) ** 2

        j = q_con * err_lat_squared +\
            q_lag * err_lag_squared +\
            q_u * u_yaw_dot ** 2 +\
            100 * slack**2 
        return j
    
    def objective_forces(self, z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.objective(pos_x,pos_y,u_yaw_dot,slack,q_con,q_lag,q_u,s,local_path_length,labels_x, labels_y, labels_heading)

    def objective_terminal_cost(self, yaw,pos_x,pos_y,qt_pos,qt_rot,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading):
        # produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y
        ref_heading = left_side @ labels_heading

        # terminal cost
        dot_direction = (np.cos(ref_heading) * np.cos(yaw)) + (np.sin(ref_heading) * np.sin(yaw)) # evaluate car angle relative to a straight path
        misalignment = -dot_direction # incentivise alligning with the path
        # higher penalty costs on v and path tracking, plus an dditional penalty for not alligning with the path at the end
        err_pos_squared_t = (pos_x - ref_x)**2 + (pos_y - ref_y)**2
        j_term_pos =    qt_pos * err_pos_squared_t + \
                        qt_rot * misalignment+\
                        - qt_s_high * (s/(self.time_horizon*V_target))**2
        
        return j_term_pos
    
    def objective_terminal_forces(self, z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.objective_terminal_cost(yaw,pos_x,pos_y,qt_pos,qt_rot,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading)


    def high_level_planner_continous_dynamics(self,V_target,u_yaw_dot,yaw,s_dot):

        # --- define the dynamic constraint ---
        # "robot" moving at constant speed
        x_dot = V_target * np.cos(yaw)
        y_dot = V_target * np.sin(yaw)
        yaw_dot = u_yaw_dot

        state_dot = [x_dot,y_dot, yaw_dot, s_dot]
        return state_dot
    
    
    def high_level_planner_continous_dynamics_forces(self, x, u, p):
        z = casadi.vertcat(u, x)
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return np.array(self.high_level_planner_continous_dynamics(V_target,u_yaw_dot,yaw,s_dot))

    def lane_boundary_constraint(self,pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y):
        # produce the path relative quantities
        # produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y

        return ((lane_width+slack)/2)**2 - ((pos_x - ref_x)**2  + (pos_y - ref_y)**2)  

    def lane_boundary_constraint_forces(self,z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return np.array([self.lane_boundary_constraint(pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y)])




    def produce_X0(self,V_target,local_path_length,labels_x, labels_y_params):
        # NOTE this needs to be updated to include the slack variable
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # z = yaw_dot slack x y yaw s ref_x ref_y ref_heading
        #     0       1     2 3 4   5 6     7     8  

        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + V_target * 1.5, N_0+1)

        # interpolate to get kurvature values
        normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        k_0_vals = np.interp(s_star_0, normalized_s_4_kernel_path, labels_k_params)
        x_ref_0 = np.zeros(N_0+1)
        y_ref_0 = np.zeros(N_0+1)
        ref_heading_0 = np.zeros(N_0+1)
        dt = self.time_horizon / N_0
        u_yaw_rate_0 = np.zeros(N_0+1)
        for i in range(1,N_0+1):
            x_ref_0[i] = x_ref_0[i-1] + V_target * dt * np.cos(ref_heading_0[i-1])
            y_ref_0[i] = y_ref_0[i-1] + V_target * dt * np.sin(ref_heading_0[i-1])
            ref_heading_0[i] = ref_heading_0[i-1] + k_0_vals[i-1] * V_target * dt

            u_yaw_rate_0[i-1] = (ref_heading_0[i] - ref_heading_0[i-1] )/ dt

        # now down sample to the N points
        s_0_vec = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), s_0_vec)
        x_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), x_ref_0)
        y_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), y_ref_0)
        ref_heading_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), ref_heading_0)
        u_yaw_rate_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), u_yaw_rate_0)



        # assign values to the array
        X0_array[:,0] = u_yaw_rate_0
        X0_array[:,1] = np.zeros(self.N+1) # slack variable should be zero
        X0_array[:,2] = x_ref_0
        X0_array[:,3] = y_ref_0
        X0_array[:,4] = ref_heading_0
        X0_array[:,5] = s_0_vec
        X0_array[:,6] = x_ref_0
        X0_array[:,7] = y_ref_0
        X0_array[:,8] = ref_heading_0

        return X0_array








