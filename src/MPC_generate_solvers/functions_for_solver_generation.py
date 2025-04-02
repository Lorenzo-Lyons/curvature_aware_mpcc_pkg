import numpy as np
try:
    from .path_track_definitions import K_RBF_kernel, K_matern2_kernel
except:
    from path_track_definitions import K_RBF_kernel, K_matern2_kernel

import casadi

# This assumes that the DART system identification package is installed
from DART_dynamic_models.dart_dynamic_models import model_functions 


class generate_high_level_path_planner_ocp(): # inherits from DART system identification

    def __init__(self,MPC_algorithm):
        
        self.MPC_algorithm = MPC_algorithm
        self.solver_name_acados = 'high_level_reference_generator_' + MPC_algorithm
        self.solver_name_forces = 'high_level_forces_reference_generator_' + MPC_algorithm

        self.n_points_kernelized = 41 # number of points in the kernelized path (41 for reference)
        self.time_horizon = 1.5
        self.N = 30 # stages
        self.max_yaw_rate = 10 # based on w = V / R = k * V so 2 * V is the maximum yaw rate 
        self.nx = 7 # pos_x, pos_y, yaw, s, ref_x, ref_y, ref_heading
        self.nu = 2 # u_yaw_dot, slack
        self.n_parameters = 9 + self.n_points_kernelized
        self.n_inequality_constraints = 1
        
    
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
        u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading = self.unpack_state(vertcat(model.u,model.x))

        # unpack parameters
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k = self.unpack_parameters(model.p)

        # assign dynamic constraint
        model.f_expl_expr = vertcat(*self.high_level_planner_continous_dynamics(s,local_path_length,labels_k,V_target,ref_x,ref_y,ref_heading,u_yaw_dot,pos_x,pos_y,yaw)) # make into vertical vector for casadi


        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(pos_x,pos_y,ref_x,ref_y,ref_heading,u_yaw_dot,slack,q_con,q_lag,q_u,s,qt_s_high) 
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high,V_target)
                
        # constraints
        #ocp.constraints.constr_type = 'BGH'
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.lbu = np.array([-self.max_yaw_rate,0])
        ocp.constraints.ubu = np.array([+self.max_yaw_rate,100]) 

        # define lane boundary constraints
        ocp.model.con_h_expr = self.lane_boundary_constraint(pos_x,pos_y,ref_x,ref_y,slack,lane_width)  # Define h(x, u)
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
                            #  u_yaw_dot,       slack, pos_x, pos_y,yaw,   s,  ref_x, ref_y, ref_heading
        model.lb = np.array([-self.max_yaw_rate, 0.0 , -1000, -1000,-1000,-0.2,-1000,-1000,-1000])  # lower bound on inputs
        model.ub = np.array([+self.max_yaw_rate, +100, +1000, +1000,+1000,+1000,+1000,+1000,+1000])  # upper bound on inputs

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
        codeoptions.nlp.TolStat = 1e-3  # inf norm tol. on stationarity
        codeoptions.nlp.TolEq = 1e-3  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1e-3  # tol. on inequality constraints
        codeoptions.nlp.TolComp = 1e-3  # tol. on complementarity

        # set warm start behaviour for dual variables (so always warm start from solver perspective, even if in practice you give it a vector of zeros)
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm

        #set overwrite behviour
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask

        codeoptions.solvemethod = 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        # NOTE that by default the solver uses a single sqp iteration so you need to increase the number of iterations
        #codeoptions.nlp.hessian_approximation = 'gauss-newton'
        #codeoptions.solver_timeout = 1  # Set a 40 ms time limit we assume the controller rate is 20Hz but you need some time to do other things in the control loop
        codeoptions.solver_exit_external = 1
        codeoptions.sqp_nlp.maxqps = 4
        codeoptions.sqp_nlp.maxSQPit = 10
        codeoptions.sqp_nlp.reg_hessian = 1e-6  # regularization of hessian (default is 5 * 10^(-9))
        #codeoptions.sqp_nlp.use_line_search = False  # Enable line search (default)

        codeoptions.parallel = 1 # this doesn't really do much


        return model,codeoptions

    def unpack_state(self,z):
        u_yaw_dot = z[0]
        slack = z[1]
        pos_x = z[2]
        pos_y = z[3]
        yaw =   z[4]
        s =     z[5]
        ref_x = z[6]       # path reference point x
        ref_y = z[7]       # path reference point y
        ref_heading = z[8] # path reference heading
        return u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading

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
        labels_k = p[9:]
        return V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k
    

    def objective(self,pos_x,pos_y,ref_x,ref_y,ref_heading,u_yaw_dot,slack,q_con,q_lag,q_u,s,qt_s_high):
        # stage cost
        if self.MPC_algorithm == 'MPCC':
            err_lag_squared = ((pos_x - ref_x) *  np.cos(ref_heading)  + (pos_y - ref_y) * np.sin(ref_heading)) ** 2
            err_lat_squared = ((pos_x - ref_x) * -np.sin(ref_heading) + (pos_y - ref_y) * np.cos(ref_heading)) ** 2

            j_path = q_con * err_lat_squared +\
                     q_lag * err_lag_squared
        else:
            # cost function for CAMPCC
            # penalize deviation from the path only since the s_dot integration is much more precise
            err_lat_squared = (pos_x - ref_x)**2 + (pos_y - ref_y)**2            
            j_path = q_con * err_lat_squared #- qt_s_high * s**2
        
        j = j_path + q_u * u_yaw_dot ** 2 + 100 * slack**2 

        return j
    
    def objective_forces(self, z, p):
        u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k = self.unpack_parameters(p)
        return self.objective(pos_x,pos_y,ref_x,ref_y,ref_heading,u_yaw_dot,slack,q_con,q_lag,q_u,s,qt_s_high)

    def objective_terminal_cost(self, ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high,V_target):
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
        u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k = self.unpack_parameters(p)
        return self.objective_terminal_cost(ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high,V_target)


    def high_level_planner_continous_dynamics(self,s,local_path_length,labels_k,V_target,ref_x,ref_y,ref_heading,u_yaw_dot,pos_x,pos_y,yaw):
        # Check if s is casadi or numpy
        if isinstance(s, casadi.MX) or isinstance(s, casadi.SX):
            cos = casadi.cos
            sin = casadi.sin
        else:
            cos = np.cos
            sin = np.sin
        
        try:
            from path_track_definitions import generate_fixed_path_quantities
        except:
            from .path_track_definitions import generate_fixed_path_quantities
                # evalaute curvature of the path as a function of s
        # n points kernelized is the number of points used to kernelize the path but defined above to get the nparamters right
        path_lengthscale = 1.3/self.n_points_kernelized
        lambda_val = 0.0001**2
        Kxx_inv, normalized_s_4_kernel_path = generate_fixed_path_quantities(path_lengthscale,
                                                                            lambda_val,
                                                                            self.n_points_kernelized)
        s_star = s / local_path_length # normalize s

        K_x_star = K_matern2_kernel(s_star, normalized_s_4_kernel_path,
                                path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ Kxx_inv
        k = left_side @ labels_k

        # --- define the dynamic constraint ---
        # "robot" moving at constant speed
        x_dot = V_target * cos(yaw)
        y_dot = V_target * sin(yaw)
        yaw_dot = u_yaw_dot

        # s_dot definition depending on the selected algorithm
        if self.MPC_algorithm == 'MPCC':
            s_dot = V_target
        else:
            v_tan = V_target * cos(yaw - ref_heading)
            p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -np.cos(ref_heading)
            den_corrected = self.soft_min(1+p*k,0.3)
            projection_ratio = 1 / den_corrected
            #projection_ratio = 1 / (1+p*k)

            s_dot = v_tan * projection_ratio

        x_ref_dot = s_dot * cos(ref_heading) 
        y_ref_dot = s_dot * sin(ref_heading)
        ref_heading_dot = k * s_dot

        state_dot = [x_dot,y_dot, yaw_dot, s_dot ,x_ref_dot, y_ref_dot, ref_heading_dot]
        return state_dot
    
    def soft_min(self, x, min_val):
        sharpness = 10
        return min_val + 0.5*(1 + np.tanh(sharpness*(x-min_val)))*(x-min_val)
    
    def high_level_planner_continous_dynamics_forces(self, x, u, p):
        z = casadi.vertcat(u, x)
        u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k = self.unpack_parameters(p)
        return self.high_level_planner_continous_dynamics(s,local_path_length,labels_k,V_target,ref_x,ref_y,ref_heading,u_yaw_dot,pos_x,pos_y,yaw)

    def lane_boundary_constraint(self,pos_x,pos_y,ref_x,ref_y,slack,lane_width):
        return ((lane_width+slack)/2)**2 - ((pos_x - ref_x)**2  + (pos_y - ref_y)**2)  

    def lane_boundary_constraint_forces(self,z, p):
        u_yaw_dot,slack, pos_x,pos_y,yaw,s, ref_x, ref_y, ref_heading = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, qt_pos, qt_rot, lane_width, qt_s_high ,labels_k= self.unpack_parameters(p)
        return [self.lane_boundary_constraint(pos_x,pos_y,ref_x,ref_y,slack,lane_width)]




    def produce_X0(self,V_target,local_path_length,labels_k,labels_s,labels_x,labels_y,labels_heading):
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # z = yaw_dot slack pos_x, pos_y, yaw, s, ref_x, ref_y, ref_heading
        #     0       1     2      3       4   5  6      7      8

        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, V_target * self.time_horizon, N_0+1)

        # interpolate to get curvature values
        #normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        k_0_vals = np.interp(s_star_0, labels_s, labels_k)
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
        # z = yaw_dot slack pos_x, pos_y, yaw, s, ref_x, ref_y, ref_heading
        #     0       1     2      3       4   5  6      7      8

        X0_array[:,0] = u_yaw_rate_0
        X0_array[:,1] = np.zeros(self.N+1) # slack variable should be zero
        X0_array[:,2] = x_ref_0 # s_dot can be around V_target
        X0_array[:,3] = y_ref_0
        X0_array[:,4] = ref_heading_0
        X0_array[:,5] = s_0_vec
        X0_array[:,6] = x_ref_0
        X0_array[:,7] = y_ref_0
        X0_array[:,8] = ref_heading_0


        return X0_array



class generate_high_level_MPCC_PP(): # inherits from DART system identification

    def __init__(self):
        
        self.solver_name_acados = 'high_level_acados_MPCC_PP'
        self.solver_name_forces = 'high_level_forces_MPCC_PP'

        self.n_points_kernelized = 41 # number of points in the kernelized path (41 for reference)
        self.time_horizon = 1.5
        self.N = 30 # stages
        self.max_yaw_rate = 10 # based on w = V / R = k * V so 2 * V is the maximum yaw rate 
        self.nu = 3
        self.nx = 4
        self.n_parameters = 10 + 3 * self.n_points_kernelized
        self.n_inequality_constraints = 1

        self.max_sdot = 6.0

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
        V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(model.p)

        # assign dynamic constraint
        model.f_expl_expr = vertcat(*self.high_level_planner_continous_dynamics(V_target,u_yaw_dot,yaw,s_dot)) # make into vertical vector for casadi


        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(pos_x,pos_y,u_yaw_dot,slack,s_dot,q_con,q_lag,q_u, q_sdot,s,local_path_length,labels_x, labels_y, labels_heading) 
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(yaw,pos_x,pos_y,qt_pos,qt_rot,q_lag,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading)
        
                
        # constraints
        #ocp.constraints.constr_type = 'BGH'
        ocp.constraints.idxbu = np.array([0,1,2])
        ocp.constraints.lbu = np.array([-self.max_yaw_rate,0,0.0])
        ocp.constraints.ubu = np.array([+self.max_yaw_rate,100,self.max_sdot]) 

        # define lane boundary constraints
        ocp.model.con_h_expr = self.lane_boundary_constraint(pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y ,labels_heading)  # Define h(x, u)
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
        model.lb = np.array([-self.max_yaw_rate, 0.0 ,   0.0, -1000, -1000,-1000,-0.2])  # lower bound on inputs
        model.ub = np.array([+self.max_yaw_rate, +100, self.max_sdot, +1000, +1000,+1000,+1000])  # upper bound on inputs

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
        codeoptions.nlp.TolStat = 1e-3  # inf norm tol. on stationarity
        codeoptions.nlp.TolEq = 1e-3  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1e-3  # tol. on inequality constraints
        codeoptions.nlp.TolComp = 1e-3  # tol. on complementarity

        # set warm start behaviour for dual variables (so always warm start from solver perspective, even if in practice you give it a vector of zeros)
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm

        #set overwrite behviour
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask

        codeoptions.solvemethod = 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        # NOTE that by default the solver uses a single sqp iteration so you need to increase the number of iterations
        #codeoptions.nlp.hessian_approximation = 'gauss-newton'
        #codeoptions.solver_timeout = 1  # Set a 40 ms time limit we assume the controller rate is 20Hz but you need some time to do other things in the control loop
        codeoptions.solver_exit_external = 1
        codeoptions.sqp_nlp.maxqps = 4
        codeoptions.sqp_nlp.maxSQPit = 10
        codeoptions.sqp_nlp.reg_hessian = 1e-6  # regularization of hessian (default is 5 * 10^(-9))
        #codeoptions.sqp_nlp.use_line_search = False  # Enable line search (default)

        codeoptions.parallel = 1 # this doesn't really do much


        return model,codeoptions

    def unpack_state(self,z):
        # control inputs
        u_yaw_dot = z[0]
        slack = z[1]
        s_dot = z[2]
        # state variables
        pos_x = z[3]
        pos_y = z[4]
        yaw =   z[5]
        s =     z[6]
        return u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s

    def unpack_parameters(self,p):
        V_target = p[0]
        local_path_length = p[1]
        q_con = p[2]
        q_lag = p[3]
        q_u = p[4]
        q_sdot = p[5]
        qt_pos = p[6]
        qt_rot = p[7]
        lane_width = p[8]
        qt_s_high = p[9]
        # prepare the kernelized path labels
        idx_start = 10
        idx_x_end = idx_start + self.n_points_kernelized
        idx_y_end = idx_x_end + self.n_points_kernelized
        idx_heading_end = idx_y_end + self.n_points_kernelized
        labels_x = p[idx_start:idx_x_end]
        labels_y = p[idx_x_end:idx_y_end]
        labels_heading = p[idx_y_end:idx_heading_end]
        return V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading
    

    def objective(self,pos_x,pos_y,u_yaw_dot,slack,s_dot,q_con,q_lag,q_u, q_sdot,s,local_path_length,labels_x, labels_y, labels_heading):
        # produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y
        ref_heading = left_side @ labels_heading

        # stage cost
        err_lag_squared = ((pos_x - ref_x) *  casadi.cos(ref_heading)  + (pos_y - ref_y) * casadi.sin(ref_heading)) ** 2
        err_lat_squared = ((pos_x - ref_x) * -casadi.sin(ref_heading) + (pos_y - ref_y) * casadi.cos(ref_heading)) ** 2

        j = q_con * err_lat_squared +\
            q_lag * err_lag_squared +\
            q_u * u_yaw_dot ** 2 +\
            q_sdot * s_dot ** 2 +\
            100 * slack**2 
        return j
    

    def objective_forces(self, z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.objective(pos_x,pos_y,u_yaw_dot,slack,s_dot,q_con,q_lag,q_u, q_sdot,s,local_path_length,labels_x, labels_y, labels_heading)

    def objective_terminal_cost(self, yaw,pos_x,pos_y,qt_pos,qt_rot,q_lag,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading):
        #produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y
        ref_heading = left_side @ labels_heading

        err_lag_squared = ((pos_x - ref_x) *  casadi.cos(ref_heading)  + (pos_y - ref_y) * casadi.sin(ref_heading)) ** 2

        # terminal cost
        dot_direction = (casadi.cos(ref_heading) * casadi.cos(yaw)) + (casadi.sin(ref_heading) * casadi.sin(yaw)) # evaluate car angle relative to a straight path
        misalignment = - dot_direction # incentivise alligning with the path
        # higher penalty costs on v and path tracking, plus an dditional penalty for not alligning with the path at the end
        err_pos_squared_t = (pos_x - ref_x)**2 + (pos_y - ref_y)**2
        j_term_pos =    qt_pos * err_pos_squared_t + \
                        qt_rot * misalignment+\
                        q_lag * err_lag_squared+\
                        - qt_s_high * (s/(self.time_horizon*V_target))**2     
        return j_term_pos
    
    def objective_terminal_forces(self, z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.objective_terminal_cost(yaw,pos_x,pos_y,qt_pos,qt_rot,q_lag,s,qt_s_high,V_target,local_path_length,labels_x, labels_y, labels_heading)


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
        V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.high_level_planner_continous_dynamics(V_target,u_yaw_dot,yaw,s_dot)

    def lane_boundary_constraint(self,pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y, labels_heading):
        # produce the path relative quantities
        # produce x, y, heading of the path
        s_star = s / local_path_length # normalize s
        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        ref_x = left_side @ labels_x
        ref_y = left_side @ labels_y
        ref_heading = left_side @ labels_heading

        err_lat_squared = ((pos_x - ref_x) * -casadi.sin(ref_heading) + (pos_y - ref_y) * casadi.cos(ref_heading)) ** 2

        return ((lane_width+slack)/2)**2 - err_lat_squared #((pos_x - ref_x)**2  + (pos_y - ref_y)**2)  

    def lane_boundary_constraint_forces(self,z, p):
        u_yaw_dot,slack,s_dot,pos_x,pos_y,yaw,s = self.unpack_state(z)
        V_target, local_path_length, q_con, q_lag, q_u, q_sdot, qt_pos, qt_rot, lane_width, qt_s_high ,labels_x, labels_y, labels_heading = self.unpack_parameters(p)
        return self.lane_boundary_constraint(pos_x,pos_y,s,slack,lane_width,local_path_length,labels_x, labels_y,labels_heading)



    def produce_X0(self,V_target,local_path_length,labels_x, labels_y, labels_heading):
        # NOTE this needs to be updated to include the slack variable
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # z = yaw_dot slack s_dot x y yaw s 
        #     0       1     2     3 4   5 6 

        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + V_target * 1.5, N_0+1)

        # interpolate to get kurvature values
        normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        
        x_ref_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_x)
        y_ref_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_y)
        ref_heading_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_heading)

        # now down sample to the N points
        s_0_vec = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), s_0_vec)
        x_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), x_ref_0)
        y_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), y_ref_0)
        ref_heading_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), ref_heading_0)
        u_yaw_rate_0 = np.diff(ref_heading_0) / (self.time_horizon / self.N)
        u_yaw_rate_0 = np.append(u_yaw_rate_0, u_yaw_rate_0[-1])         # append last value again to make the array the same size

        # assign values to the array
        X0_array[:,0] = u_yaw_rate_0
        X0_array[:,1] = np.zeros(self.N+1) # slack variable should be zero
        X0_array[:,2] = V_target # s_dot can be around V_target
        X0_array[:,3] = x_ref_0
        X0_array[:,4] = y_ref_0
        X0_array[:,5] = ref_heading_0
        X0_array[:,6] = s_0_vec


        return X0_array



class generate_low_level_solver_ocp(model_functions): # inherits from DART system identification

    def __init__(self,dynamic_model):
        # features to add:
        # 1. actuator delay
        # 2. inequality constraints

        # store for later
        self.dynamic_model = dynamic_model
        
        # chose options
        self.N = 30 # this should match the high level planner (at least for now)
        self.time_horizon = 1.5
        self.solver_name_acados = 'low_level_acados_' + dynamic_model # 'dynamic_bicycle', 'kinematic_bicycle', 'SVGP'
        self.nx = 6
        self.nu = 3
        self.n_parameters = 14
        self.n_inequality_constraints = 1

        self.solver_name_forces = 'low_level_forces_' + dynamic_model # 'dynamic_bicycle', 'kinematic_bicycle', 'SVGP'
        #self.max_centrifugal_force = 2.5 # maximum lateral force in m/s^2

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


        # -----  extract named variables for function definitions ----
        z = vertcat(model.u,model.x)
        th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy, w = self.unpack_state(z)
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width = self.unpack_parameters(model.p)
        # -----------------------------------------------------------

        #model.f_expl_expr = self.dynamic_constraint(th_input,st_input,yaw,vx) # now just kinematic bicycle

        if self.dynamic_model == "kinematic_bicycle":
            model.f_expl_expr = casadi.vertcat(*self.kinematic_bicycle_continuous_dynamics(th_input,st_input,vx,yaw))
        elif self.dynamic_model == "dynamic_bicycle":
            model.f_expl_expr = casadi.vertcat(*self.dynamic_bicycle_continuous_dynamics(th_input,st_input,vx,vy,w,yaw))
        else:
            print('Dynamic_constraint: Invalid dynamic model setting')

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # stage-wise cost
        ocp.model.cost_expr_ext_cost  =  self.objective(th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                                                                x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u,q_acc)
        # terminal cost
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(pos_x,pos_y,yaw,vx,x_ref, y_ref, yaw_ref,V_target,q_v, qt_pos, qt_rot)
        
        # constraints
        ocp.constraints.constr_type = 'BGH'
        # u = [throttle, steer, slack_var]
        ocp.constraints.lbu = np.array([0,-1, 0])
        ocp.constraints.ubu = np.array([+1,+1, 100]) # high value for slack variable
        ocp.constraints.idxbu = np.array([0, 1, 2])

        # define lane boundary constraints
        ocp.model.con_h_expr = self.lane_boundary_constraint(pos_x,pos_y,x_path,y_path,slack,lane_width)  # Define h(x, u)
        ocp.constraints.lh = np.array([0.0])  # Lower bound (h_min)
        ocp.constraints.uh = np.array([1000])  # Upper bound (h_max)

        # define lane boundary constraints
        # ocp.model.con_h_expr = vertcat(self.lane_boundary_constraint(pos_x,pos_y,x_path,y_path,slack,lane_width), self.max_centrifugal_force_constraint(vx,w,slack,st_input))  # Define h(x, u)
        # ocp.constraints.lh = np.array([0.0,0.0])  # Lower bound (h_min)
        # ocp.constraints.uh = np.array([1000,1000])  # Upper bound (h_max)


        # Initial state constraint
        ocp.constraints.x0 = np.zeros(self.nx)  # This is a default value, it will be updated at runtime

        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # GAUSS_NEWTON, EXACT
        ocp.solver_options.integrator_type = 'ERK' # IRK, ERK
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP   SQP_RTI
        ocp.solver_options.tf = 1.5  # time horizon in seconds
        
        # messing with the convergence criteria
        ocp.solver_options.qp_solver_warm_start = 1 # 0: no warm start, 1: warm start 2 : hot start
        ocp.solver_options.globalization = 'FIXED_STEP' # 'MERIT_BACKTRACKING', 'FIXED_STEP' # fixed is the default
        ocp.solver_options.print_level = 0 # no print
        #ocp.solver_options.tol = 0.001
        if self.dynamic_model == "kinematic_bicycle":
            ocp.solver_options.sim_method_num_steps = 1  # Number of sub-steps in each interval for integration purpouses
        else:#if using the dynamic model use more intermediate integration steps
            ocp.solver_options.sim_method_num_steps = 10

        #ocp.solver_options.qp_solver_iter_max = 10
        #ocp.solver_options.nlp_solver_max_iter = 1000

        # Initialize parameters with default values (this step is important to avoid dimension mismatch)
        ocp.parameter_values = np.zeros(self.n_parameters)
        return ocp
    
    def produce_FORCES_model_codeoptions(self):
        import forcespro.nlp

        model = forcespro.nlp.SymbolicModel(self.N)

        model.xinitidx = np.array(range(self.nu,self.nu + self.nx))  # variables in these positions are affected by initial state constraint. (I.e. they cannot change in the first stage)
        
        #theese parameters are the same for all the solvers
        model.nvar = self.nu + self.nx         # number of stage variables
        model.neq = self.nx                    # number of equality constraints (dynamic model)
        model.npar = self.n_parameters         # number of parameters
        model.E = np.concatenate([np.zeros((self.nx, self.nu)), np.eye(self.nx)], axis=1)  # This extraction matrix tells forces what variables are states and what are inputs

        # set fixed input bounds since they will not change at runtime
        # generate inf upper and lower bounds for the states
        model.lb = np.array([0, -1.0, 0,    -1000,-1000,-1000,0,-1,-1000])  # lower bound on inputs
        model.ub = np.array([1, +1.0, +100, +1000,+1000,+1000,+5,+1,+1000])  # upper bound on inputs

        # Set objective
        #model.objective = self.objective_forces
        for i in range(self.N - 1):
            model.objective[i] = self.objective_forces  # eval_obj is a Python function
        model.objective[self.N-1] = self.objective_terminal_forces

  
        # Set dynamic constraint
        if self.dynamic_model == "kinematic_bicycle":
            model.continuous_dynamics = self.kinematic_bicycle_continous_dynamics_forces
            #model.eq = self.kinematic_bicycle_dynamic_constraint_forces
        elif self.dynamic_model == "dynamic_bicycle":
            model.continuous_dynamics = self.dynamic_bicycle_continous_dynamics_forces
        else:
            print('Dynamic_constraint: Invalid dynamic model setting')
        

        # Set non linear constraints
        model.nh = self.n_inequality_constraints
        model.ineq = self.lane_boundary_constraint_forces
        model.hl = np.array([0.0])
        model.hu = np.array([1000.0])  # upper bound on inequality constraints
            
        # Set non linear constraints
        # model.nh = self.n_inequality_constraints
        # model.ineq = self.non_lin_constraint_forces
        # model.hl = np.array([0.0,  0.0])
        # model.hu = np.array([1000.0, 1000.0])  # upper bound on inequality constraints

        # Define solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver') #get standard options
        # continuous dynamics options
        codeoptions.nlp.integrator.type = 'ERK4'
        codeoptions.nlp.integrator.Ts = self.time_horizon / self.N
        codeoptions.nlp.integrator.nodes = 5 # intermediate nodes for the integrator

        codeoptions.name = self.solver_name_forces
        codeoptions.printlevel = 0  #  1: summary line after each solve,   0: no prit
        codeoptions.BuildSimulinkBlock = 0  # disable simulink block generation because we don't need it
        codeoptions.maxit = 200  # maximum iterations
        codeoptions.noVariableElimination = 1  # enable or disable variable simplification (like if first stage is constrained)
        codeoptions.nlp.stack_parambounds = True  # determines if the parameters can simply be stacked (but not sure exactly what it does)

        # set tolerances
        # codeoptions.nlp.TolStat = 1e-4  # inf norm tol. on stationarity
        # codeoptions.nlp.TolEq = 1e-5  # tol. on equality constraints
        # codeoptions.nlp.TolIneq = 1e-5  # tol. on inequality constraints
        # codeoptions.nlp.TolComp = 1e-5  # tol. on complementarity

        # set warm start behaviour for dual variables (so always warm start from solver perspective, even if in practice you give it a vector of zeros)
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm

        #set overwrite behviour
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask

        codeoptions.solvemethod = 'PDIP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'

        return model,codeoptions







    def unpack_state(self, z):
        th_input = z[0]
        st_input = z[1]    
        slack    = z[2]    
        pos_x = z[3]
        pos_y = z[4]
        yaw = z[5]
        vx = z[6]
        vy = z[7]
        w = z[8]
        return th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w


    def unpack_parameters(self,p):
        V_target =   p[0]      # target longitudinal_speed/s_dot depending on the formulation MPCC/CAMPCC respectively            # time step
        q_v =        p[1]              # mpc stage cost function tuning weights
        q_pos =      p[2]  # position tracking 
        q_rot =      p[3]  # orientation tracking
        q_u =        p[4]
        qt_pos =     p[5]             # terminal cost (position relative to final path direction)
        qt_rot =     p[6]             # (orientation relative to final path direction)
        q_acc =      p[7]            # acceleration cost tuning weight
        x_ref =      p[8]            # reference position x
        y_ref =      p[9]           # reference position y
        yaw_ref =    p[10]         # reference yaw
        x_path  =    p[11]         # path x
        y_path  =    p[12]         # path y
        lane_width = p[13]         # lane width

        return V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width


    def produce_xdot(self,yaw,vx,vy,w,acc_x,acc_y,acc_w):
        xdot1 = vx * np.cos(yaw) - vy * np.sin(yaw)
        xdot2 = vx * np.sin(yaw) + vy * np.cos(yaw)
        xdot3 = w
        xdot4 = acc_x  
        xdot5 = acc_y  
        xdot6 = acc_w
        return [xdot1,xdot2,xdot3,xdot4,xdot5,xdot6]


    def kinematic_bicycle_continuous_dynamics(self,th_input,st_input,vx,yaw):
        # evaluate longitudinal forces
        Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)

        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration

        # convert steering to steering angle
        steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
        # evaluate lateral velocity and yaw rate
        w = vx * np.tan(steering_angle) / (self.lr_self + self.lf_self) # angular velocity
        vy = self.l_COM_self * w

        # assemble derivatives
        xdot = self.produce_xdot(yaw,vx,vy,w,acc_x,0,0) # vy, acc y and acc w are 0 in this case
        return xdot

    def dynamic_bicycle_continuous_dynamics(self,th_input,st_input,vx,vy,w,yaw):
        

        # #evaluate steering angle 
        # steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)

        # # # evaluate longitudinal forces
        # Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
        #             + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)\
        #             + self.F_friction_due_to_steering(steering_angle,vx,self.a_stfr_self,self.b_stfr_self,self.d_stfr_self,self.e_stfr_self)

        # c_front = (self.m_front_wheel_self)/self.m_self
        # c_rear = (self.m_rear_wheel_self)/self.m_self

        # # redistribute Fx to front and rear wheels according to normal load
        # Fx_front = Fx_wheels * c_front
        # Fx_rear = Fx_wheels * c_rear

        # #evaluate slip angles
        # alpha_f,alpha_r = self.evaluate_slip_angles(vx,vy,w,self.lf_self,self.lr_self,steering_angle)

        # #lateral forces
        # Fy_wheel_f = self.lateral_tire_force(alpha_f,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self)
        # Fy_wheel_r = self.lateral_tire_force(alpha_r,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self)

        # acc_x,acc_y,acc_w = self.solve_rigid_body_dynamics(vx,vy,w,steering_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)
        
        acc_x,acc_y,acc_w = self.dynamic_bicycle(th_input, st_input, vx, vy, w )

        xdot = self.produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w)

        return xdot
    




    
    def kinematic_bicycle_continous_dynamics_forces(self,x,u):
        # extract control inputs
        th_input = u[0]
        st_input = u[1]
        slack = u[2]
        # extract states
        pos_x = x[0]
        pos_y = x[1]
        yaw = x[2]
        vx = x[3]
        vy = x[4]
        w = x[5]

        #return np.array([vx,0,w,th_input,0,st_input])
        return self.kinematic_bicycle_continuous_dynamics(th_input,st_input,vx,yaw)

    def dynamic_bicycle_continous_dynamics_forces(self,x,u):
        # extract control inputs
        th_input = u[0]
        st_input = u[1]
        slack = u[2]
        # extract states
        pos_x = x[0]
        pos_y = x[1]
        yaw = x[2]
        vx = x[3]
        vy = x[4]
        w = x[5]

        #return np.array([vx,0,w,th_input,0,st_input])
        return self.dynamic_bicycle_continuous_dynamics(th_input,st_input,vx,vy,w,yaw)









    def objective(self,th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                            x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u, q_acc):
        
        error_position_sqrd = (pos_x - x_ref)**2 + (pos_y - y_ref)**2
        error_heading_sqrd = (yaw - yaw_ref)**2

        # evalaute acceleration to penalize it

        # from kinemaitc bicycle model
        Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
                    
        acc_x =  Fx_wheels / self.m_self 


        j = q_v * (vx - V_target) ** 2 + \
            q_pos * error_position_sqrd +\
            q_rot * error_heading_sqrd +\
            q_u * st_input ** 2 +\
            q_u * th_input ** 2 +\
            100 * slack ** 2+\
            q_acc * acc_x ** 2
        
        return j 

    def objective_terminal_cost(self,pos_x,pos_y,yaw,vx,x_ref, y_ref, yaw_ref,V_target,q_v, qt_pos, qt_rot):
        # control input can't be used in terminal cost for ACADOS
        error_position_sqrd = (pos_x - x_ref)**2 + (pos_y - y_ref)**2
        error_heading_sqrd = (yaw - yaw_ref)**2
        j_term = q_v * (vx - V_target) ** 2 + \
                qt_pos * error_position_sqrd +\
                qt_rot * error_heading_sqrd
        
        return  j_term

    def objective_forces(self,z,p):
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width = self.unpack_parameters(p) 
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
        return self.objective(th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                            x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u,q_acc)


    def objective_terminal_forces(self,z,p):
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width = self.unpack_parameters(p) 
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
        return self.objective_terminal_cost(pos_x,pos_y,yaw,vx, x_ref, y_ref, yaw_ref,V_target,q_v, qt_pos, qt_rot)
    
    def lane_boundary_constraint(self,pos_x,pos_y,x_path,y_path,slack,lane_width):
        return ((lane_width+slack)/2)**2 - ((pos_x - x_path)**2  + (pos_y - y_path)**2)  

    def lane_boundary_constraint_forces(self,z, p):
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width = self.unpack_parameters(p)
        return [self.lane_boundary_constraint(pos_x,pos_y,x_path,y_path,slack,lane_width)]

    # def max_centrifugal_force_constraint(self,vx,w,slack,st_input):
    #     if self.dynamic_model == "kinematic_bicycle":
    #         steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
    #         w_kin = vx * np.tan(steering_angle) / (self.lf_self+self.lr_self)
    #         h = - (vx * w_kin)**2 + (self.max_centrifugal_force + slack)**2
    #     elif self.dynamic_model == "dynamic_bicycle":
    #         h =  - (vx * w)**2 + (self.max_centrifugal_force + slack)**2
    #     return h

    # def non_lin_constraint_forces(self,z, p):
    #     V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, q_acc, x_ref, y_ref, yaw_ref, x_path, y_path, lane_width = self.unpack_parameters(p) 
    #     th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
    #     return [self.lane_boundary_constraint(pos_x,pos_y,x_path,y_path,slack,lane_width),self.max_centrifugal_force_constraint(vx,w,slack,st_input)]



    def produce_X0(self,V_target,
                    x_high_level,
                    y_high_level,
                    yaw_high_level,
                    yaw_rate_high_level):
        # Initial guess 
        X0_array = np.zeros((self.N+1,self.nu + self.nx))
        # z = th st slack x y yaw vx vy w
        #     0  1  2     3 4 5   6  7  8
        # Evaluate throttle to keep the constant velocity
        throttle_search_vec = np.linspace(0,1,30)
        # evalaute FX on the throttle search vec
        Fx_wheels = + self.motor_force(throttle_search_vec,V_target,self.a_m_self,self.b_m_self,self.c_m_self)\
                + self.rolling_friction(V_target,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration
        #find the throttle that gives the closest acceleration to 0
        throttle_0 = throttle_search_vec[np.argmin(np.abs(acc_x))]

        # assign initial guess for solver
        X0_array[:,0] = throttle_0
        X0_array[:,3] = x_high_level
        X0_array[:,4] = y_high_level
        X0_array[:,5] = yaw_high_level
        X0_array[:,6] = V_target # assign target speed as first guess 
        X0_array[:,8] = yaw_rate_high_level # input of high level is the yaw rate

        return X0_array



class generate_single_layer_CAMPCC(generate_low_level_solver_ocp): # in the end inherits from DART system identification
    # here we need the dynamic constraints of teh low level controller

    def __init__(self,dynamic_model,actuator_dynamics,path_2_actuator_dynamics,GP_params_folder):
        
        self.dynamic_model = dynamic_model
        self.actuator_dynamics = actuator_dynamics
        if actuator_dynamics:
            actuator_dynamics_name_tag = '_act_dyn'
        else:
            actuator_dynamics_name_tag = ''

        self.solver_name_acados = 'single_layer_acados_CAMPCC_' + dynamic_model + actuator_dynamics_name_tag
        self.solver_name_forces = 'single_layer_forces_CAMPCC_' + dynamic_model + actuator_dynamics_name_tag

        self.n_points_kernelized = 41 # number of points in the kernelized path (41 for reference)
        self.time_horizon = 1.0 #1.5 * 0.5
        self.N = 20 # stages 30
        self.nx_base = 10 # pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading
        self.nu = 3 # throttle, stteering, slack


        # if actuator dynamics are enabled we must add extra states
        if actuator_dynamics:
            # load the weights from the actuator dynamics saved parameters
            self.load_actuator_dynamics(path_2_actuator_dynamics)
            self.act_FIR_states = len(self.weights_th_FIR_solver) + len(self.weights_st_FIR_solver) - 2 # minus 2 because the last value is the input at time now (u)
            self.nx = self.nx_base + self.act_FIR_states
            # no need to add the FRI since it will be baked into the dynamics
        else:
            self.nx = self.nx_base

        if dynamic_model == "dynamic_bicycle_GP":
            from DART_dynamic_models.dart_dynamic_models import SVGP_unified_analytic
            self.SVGP_unified_analytic_obj = SVGP_unified_analytic()
            self.SVGP_unified_analytic_obj.load_parameters(GP_params_folder)

            





        self.n_parameters = 9 + self.n_points_kernelized
        self.n_inequality_constraints = 1 # non linear inequality constraints

        # set operational limits on the centrifugal force
        #self.max_centrifugal_force = 30 #6.5 # m/s^2
        # upper / lower bound on the control inputs
                        #  th_input,st_input,slack,
        from DART_dynamic_models.dart_dynamic_models import model_functions
        mf = model_functions()

        self.u_l = np.array([-mf.c_m_self,-1, 0])
        self.u_u = np.array([1.0,+1, 100])
        # upper- lower bound on the states
                            # pos_x ,pos_y, yaw, vx,    vy,  w,  s,    ref_x,ref_y,ref_heading
        #self.x_l = np.array([-1000,-1000,-1000,-100,    -100,  -100,-100,-1000,-1000,-1000])
        #self.x_u = np.array([ 1000, 1000,1000,  100,     100,   100,1000,1000,1000,1000])
        self.x_l = -np.infty * np.ones(self.nx)
        self.x_u = +np.infty * np.ones(self.nx)
        


        # evalaute curvature of the path as a function of s
        try:
            from path_track_definitions import generate_fixed_path_quantities
        except:
            from .path_track_definitions import generate_fixed_path_quantities

        self.path_lengthscale = 1.3/self.n_points_kernelized
        lambda_val = 0.0001**2
        self.Kxx_inv, self.normalized_s_4_kernel_path = generate_fixed_path_quantities(self.path_lengthscale,
                                                                            lambda_val,
                                                                            self.n_points_kernelized)
    
    def load_actuator_dynamics(self,path_2_folder):
        print('loading actuator dynamics from folder: ', path_2_folder)
        # load the actuator dynamics parameters
        dt = np.load(path_2_folder + '/dt.npy').item()
        n_past_actions = np.load(path_2_folder + '/n_past_actions.npy').item()
        weights_th = np.load(path_2_folder + '/weights_throttle.npy')
        weights_st = np.load(path_2_folder + '/weights_steering.npy')


        self.dt_FIR = dt
        self.n_past_actions_FRI = n_past_actions
        self.weights_th_FIR = weights_th
        self.weights_st_FIR = weights_st

        # find the first value from the end of the weights that is larger than 10-6
        n_past_actions_th = np.where(np.abs(weights_th) > 10**-6)[0][-1] + 2
        n_past_actions_st = np.where(np.abs(weights_st) > 10**-6)[0][-1] + 2

        
        time_vec_th = np.arange(0,dt*n_past_actions_th,dt)
        time_vec_st = np.arange(0,dt*n_past_actions_st,dt)
        dt_solver = self.time_horizon / self.N
        n_past_actions_th_solver = int(np.ceil(dt*n_past_actions_th/dt_solver)) 
        n_past_actions_st_solver = int(np.ceil(dt*n_past_actions_st/dt_solver)) 

        time_vec_th_solver = np.arange(0,dt_solver*n_past_actions_th_solver+dt_solver*0.5,dt_solver)
        time_vec_st_solver = np.arange(0,dt_solver*n_past_actions_st_solver+dt_solver*0.5,dt_solver)

        # now interpolate the weights to the solver time horizon
        weights_th_solver = np.interp(time_vec_th_solver,time_vec_th,np.squeeze(weights_th[:n_past_actions_th]),right=0)
        weights_st_solver = np.interp(time_vec_st_solver,time_vec_st,np.squeeze(weights_st[:n_past_actions_st]),right=0)

        # set small values to 0 to simplyfy things
        threshold = 10**-6
        weights_th_solver[np.abs(weights_th_solver) < threshold] = 0
        weights_st_solver[np.abs(weights_st_solver) < threshold] = 0

        # assign to self
        self.weights_th_FIR_solver = weights_th_solver[:-1] / np.sum(weights_th_solver[:-1]) # skip last value that will be 0 (this was needed to interpolate correctly)
        self.weights_st_FIR_solver = weights_st_solver[:-1] / np.sum(weights_st_solver[:-1])

        # # # #  VERY TEMPORARY for debugging
        # # print('TEMPORARY: setting weights to 0 except for the first element')
        # # # replace with zeros except a one for the first element
        # # self.weights_th_FIR_solver = np.zeros_like(self.weights_th_FIR_solver)
        # # self.weights_st_FIR_solver = np.zeros_like(self.weights_st_FIR_solver)
        # # self.weights_th_FIR_solver[2] = 1
        # # self.weights_st_FIR_solver[2] = 1





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
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(vertcat(model.u,model.x))

        # unpack parameters
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(model.p)

        # assign dynamic constraint
        model.f_expl_expr = vertcat(*self.single_layer_continous_dynamics(local_path_length,labels_k,
                                        th_input,st_input,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading)) # make into vertical vector for casadi


        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(th_input,st_input,slack,pos_x,pos_y,ref_x,ref_y,q_con,q_u,vx,q_acc,q_v,s,local_path_length,labels_k,yaw,ref_heading) 
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high)
                
        # constraints
        ocp.constraints.constr_type = 'BGH'
        # u = [throttle, steer, slack_var]
        ocp.constraints.lbu = self.u_l
        ocp.constraints.ubu = self.u_u
        ocp.constraints.idxbu = np.array([0, 1, 2])
        # add constraints on the state
        ocp.constraints.lbx = self.x_l
        ocp.constraints.ubx = self.x_u
        ocp.constraints.idxbx = np.array(range(self.nx))


        # define lane boundary constraints
        ocp.model.con_h_expr = vertcat(self.lane_boundary_constraint(pos_x,pos_y,ref_x,ref_y,slack,lane_width), self.max_centrifugal_force_constraint(vx,w,slack,st_input))  # Define h(x, u)
        ocp.constraints.lh = np.zeros(2)#np.array([0.0, 0.0])  # Lower bound (h_min)
        ocp.constraints.uh = np.infty*np.ones(2)  # Upper bound (h_max)

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
                            #  th_input,st_input,slack,pos_x ,pos_y, yaw, vx,    vy,  w,    s,    ref_x,ref_y,ref_heading
        model.lb = np.array([*self.u_l,*self.x_l])  # lower bound on inputs
        model.ub = np.array([*self.u_u,*self.x_u])  # upper bound on inputs

        # Set objective
        for i in range(self.N):
            model.objective[i] = self.objective_forces  # eval_obj is a Python function
        model.objective[self.N] = self.objective_terminal_forces
        
        #model.LSobjective = self.objective_LS_forces  # Assign the LSobjective function
    

        # Set dynamic constraint
        if self.actuator_dynamics:
            self.discrete_dynamics_intermediate_shooting_vehicle = 10
            self.discrete_dynamics_intermediate_shooting_path = 4
            model.eq = self.single_layer_discrete_dynamics_actuators_forces
        else:
            model.continuous_dynamics = self.single_layer_planner_continous_dynamics_forces

        

    
        # Set non linear constraints
        model.nh = self.n_inequality_constraints
        model.ineq = self.non_lin_constraint_forces
        model.hl = np.zeros(self.n_inequality_constraints)   #np.array([0.0,0.0,0.0])
        model.hu = np.ones(self.n_inequality_constraints)*np.infty  #np.array([1000.0,1000.0,1000.0])  # upper bound on inequality constraints
        
        # Define solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver') #get standard options

        if self.actuator_dynamics == False:
            # continuous dynamics options
            codeoptions.nlp.integrator.type = 'ERK4' #'ERK4' #'ForwardEuler' #'ERK4' #'IRK2' # 'ForwardEuler' #
            codeoptions.nlp.integrator.Ts = self.time_horizon / (self.N+1)
            codeoptions.nlp.integrator.nodes = 2 # intermediate nodes for the integrator


        codeoptions.name = self.solver_name_forces
        codeoptions.printlevel = 0  #  1: summary line after each solve,   0: no prit
        codeoptions.BuildSimulinkBlock = 0  # disable simulink block generation because we don't need it
        codeoptions.maxit = 200  # maximum iterations
        codeoptions.noVariableElimination = 1  # enable or disable variable simplification (like if first stage is constrained)
        codeoptions.nlp.stack_parambounds = True  # determines if the parameters can simply be stacked (but not sure exactly what it does)

        # 
        codeoptions.parallel = 1 # this doesn't really do much
        codeoptions.solvemethod = 'SQP_NLP' # 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm (should not apply to sqp)
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask         #set overwrite behviour


        # set sqp parameters
        #codeoptions.sqp_nlp.rti = 0 # 0: no RTI, 1: RTI
        codeoptions.sqp_nlp.maxSQPit = 1  # this seems to do nothing (?)
        #codeoptions.sqp_nlp.qp_timeout = 0 # 0 no timeout, 1 yes timeout
        codeoptions.sqp_nlp.maxqps = 3   # this seems to do nothing (?)
        #codeoptions.sqp_nlp.use_line_search = 1 # 0: no line search, 1: line search (only with LS objective ecc)
        codeoptions.sqp_nlp.TolStat = 1e-3 # Tolerance on stationarity
        codeoptions.sqp_nlp.TolEq = 1e-3 # Tolerance on equality constraints
        codeoptions.sqp_nlp.reg_hessian = 1e-6
        # codeoptions.sqp_nlp.qpinit = 0 # 0: cold start, 1: centered
        # codeoptions.sqp_nlp.qp_method = 'general' # (??)
        # codeoptions.sqp_nlp.use_diagonal_hessian = -1 # (???)
        #codeoptions.sqp_nlp.autotune = 1 #  (???)
        # # also there are some 
        # codeoptions.sqp_nlp.tuning.qp_tuning.tuning0 = None
        # codeoptions.sqp_nlp.tuning.qp_tuning.tuning1 = None
        # codeoptions.sqp_nlp.tuning.qp_tuning.tuning2 = None
        # codeoptions.sqp_nlp.tuning.qp_tuning.tuning3 = None

        
 

        


        return model,codeoptions

    def unpack_state(self,z):
        th_input = z[0]
        st_input = z[1]    
        slack = z[2]
        pos_x = z[3]
        pos_y = z[4]
        yaw =   z[5]
        vx = z[6]
        vy = z[7]
        w = z[8]
        s = z[9]
        ref_x = z[10]       # path reference point x
        ref_y = z[11]       # path reference point y
        ref_heading = z[12] # path reference heading
        
        if self.actuator_dynamics:
            th_past = z[13:13+len(self.weights_th_FIR_solver)-1] # past throttle actions  (minus 1 because the first one is the current one)
            st_past = z[13+len(self.weights_th_FIR_solver)-1:]
        else:
            th_past = []
            st_past = []

        return th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past

    def unpack_parameters(self,p):
        local_path_length = p[0] # length of the path segment
        q_con = p[1]  # position tracking
        q_u = p[2]  # control input tracking
        q_acc   = p[3]  # acceleration tracking
        qt_pos = p[4]  # terminal cost (position relative to final path direction)
        qt_rot = p[5]  # (orientation relative to final path direction)
        lane_width = p[6]  # lane width
        qt_s_high = p[7]  # terminal cost on s
        q_v = p[8]  # velocity tracking
        labels_k = p[9:] # kernelized path labels
        return local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k
    

    def objective(self,th_input,st_input,slack,pos_x,pos_y,ref_x,ref_y,q_con,q_u,vx,q_acc,q_v,s,local_path_length,labels_k,yaw,ref_heading):
        # Check if s is casadi or numpy
        if isinstance(s, casadi.MX) or isinstance(s, casadi.SX):
            cos = casadi.cos
            sin = casadi.sin
        else:
            cos = np.cos
            sin = np.sin


        s_star = s / local_path_length # normalize s

        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,
                                self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        k = left_side @ labels_k

        # s_dot definition depending on the selected algorithm
        # for now we assume vy is small
        v_tan = vx * cos(yaw - ref_heading)
        p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -cos(ref_heading)
        den_corrected = self.soft_min(1+p*k,0.1)
        projection_ratio = 1 / den_corrected
        s_dot = v_tan * projection_ratio


        # from kinemaitc bicycle model
        Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration

        err_lat_squared = (pos_x - ref_x)**2 + (pos_y - ref_y)**2            
        j_path = q_con * err_lat_squared

        j = j_path\
            + q_u * st_input ** 2\
            + q_u * th_input**2\
            + q_acc * acc_x ** 2\
            + 100 * slack**2\
            + q_v * (s_dot-10)**2  # much better like this than - q_v * s_dot**2\
            #+ q_v * (vx-4)**2\
            
            
        return j
    
    def objective_forces(self, z, p):
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)
        return self.objective(th_input,st_input,slack,pos_x,pos_y,ref_x,ref_y,q_con,q_u,vx,q_acc,q_v,s,local_path_length,labels_k,yaw,ref_heading)


    def objective_LS(self,th_input,st_input,slack,pos_x,pos_y,ref_x,ref_y,q_con,q_u,vx,q_acc,q_v,s,local_path_length,labels_k,yaw,ref_heading):
        # Check if s is casadi or numpy
        if isinstance(s, casadi.MX) or isinstance(s, casadi.SX):
            cos = casadi.cos
            sin = casadi.sin
        else:
            cos = np.cos
            sin = np.sin


        s_star = s / local_path_length # normalize s

        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,
                                self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        k = left_side @ labels_k

        # s_dot definition depending on the selected algorithm
        # for now we assume vy is small
        v_tan = vx * cos(yaw - ref_heading)
        p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -cos(ref_heading)
        den_corrected = self.soft_min(1+p*k,0.1)
        projection_ratio = 1 / den_corrected
        s_dot = v_tan * projection_ratio


        # from kinemaitc bicycle model
        Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration

        err_lat_squared = (pos_x - ref_x)**2 + (pos_y - ref_y)**2            
        #j_path = q_con * err_lat_squared

        j1 = np.sqrt(q_con) * np.sqrt(err_lat_squared)
        j2 = np.sqrt(q_u)  * st_input
        j3 = np.sqrt(q_u)  * (th_input-1)
        j4 = np.sqrt(q_acc)  * acc_x
        j5 = np.sqrt(100)  * slack
        j6 = np.sqrt(q_v)  * (s_dot-15)  # much better like this than - q_v * s_dot**2\
            #+ q_v * (vx-4)**2\
            
            
        return casadi.vertcat(j1,j2,j3,j4,j5,j6)

    def objective_LS_forces(self,z,p):
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past= self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)
        return self.objective_LS(th_input,st_input,slack,pos_x,pos_y,ref_x,ref_y,q_con,q_u,vx,q_acc,q_v,s,local_path_length,labels_k,yaw,ref_heading)




    def objective_terminal_cost(self, ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high):
        # terminal cost
        dot_direction = (np.cos(ref_heading) * np.cos(yaw)) + (np.sin(ref_heading) * np.sin(yaw)) # evaluate car angle relative to a straight path
        misalignment = -dot_direction # incentivise alligning with the path
        # higher penalty costs on v and path tracking, plus an dditional penalty for not alligning with the path at the end
        err_pos_squared_t = (pos_x - ref_x)**2 + (pos_y - ref_y)**2
        j_term_pos =    qt_pos * err_pos_squared_t + \
                        qt_rot * misalignment #- qt_s_high * (s/(self.time_horizon*4)) ** 2
        
        return j_term_pos
    
    def objective_terminal_forces(self, z, p):
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)
        return self.objective_terminal_cost(ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high)


    def single_layer_continous_dynamics(self,local_path_length,labels_k,
                                        th_input,st_input,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading):
        # Check if s is casadi or numpy
        if isinstance(s, casadi.MX) or isinstance(s, casadi.SX):
            cos = casadi.cos
            sin = casadi.sin
        else:
            cos = np.cos
            sin = np.sin
        
        # --- vehicle dynamics constraint ---
        if self.dynamic_model == "kinematic_bicycle":
            x_dot, y_dot, yaw_dot, vx_dot, vy_dot, w_dot = self.kinematic_bicycle_continuous_dynamics(th_input,st_input,vx,yaw)
        elif self.dynamic_model == "dynamic_bicycle":
            x_dot, y_dot, yaw_dot, vx_dot, vy_dot, w_dot = self.dynamic_bicycle_continuous_dynamics(th_input,st_input,vx,vy,w,yaw)
        elif self.dynamic_model == "dynamic_bicycle_GP":
            # evaluate GP contribution
                                                                                    # x_star = [th st vx vy w]


            x_dot, y_dot, yaw_dot, vx_dot, vy_dot, w_dot = self.SVGP_continuous_dynamics(th_input,st_input,vx,vy,w,yaw,
                                                                                             self.SVGP_unified_analytic_obj.use_nominal_model.item())

        else:
            print('')
            print('Dynamic_constraint: Invalid dynamic model setting')

        # s_star = s / local_path_length # normalize s

        # K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,
        #                         self.path_lengthscale,1,self.n_points_kernelized)      
        # left_side = K_x_star @ self.Kxx_inv
        # k = left_side @ labels_k

        # # s_dot definition depending on the selected algorithm
        # # for now we assume vy is small
        # v_tan = vx * cos(yaw - ref_heading)
        # p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -cos(ref_heading)
        # den_corrected = self.soft_min(1+p*k,0.3)
        # projection_ratio = 1 / den_corrected
        # s_dot = v_tan * projection_ratio

        # # forwards integrate the reference path
        # x_ref_dot = s_dot * cos(ref_heading) 
        # y_ref_dot = s_dot * sin(ref_heading)
        # ref_heading_dot = k * s_dot
        s_dot,x_ref_dot,y_ref_dot,ref_heading_dot = self.kernelized_path_derivatives(local_path_length,labels_k,s,ref_x,ref_y,ref_heading,vx,yaw,pos_x,pos_y)



        # state is  pos_x, pos_y,  yaw,    vx,     vy,     w,      s,     ref_x,     ref_y,     ref_heading 
        state_dot = [x_dot,y_dot, yaw_dot, vx_dot, vy_dot, w_dot,  s_dot ,x_ref_dot, y_ref_dot, ref_heading_dot]
        return state_dot
    
    def kernelized_path_derivatives(self,local_path_length,labels_k,s,ref_x,ref_y,ref_heading,vx,yaw,pos_x,pos_y):
        if isinstance(s, casadi.MX) or isinstance(s, casadi.SX):
            cos = casadi.cos
            sin = casadi.sin
        else:
            cos = np.cos
            sin = np.sin
        s_star = s / local_path_length # normalize s

        K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,
                                self.path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ self.Kxx_inv
        k = left_side @ labels_k

        # s_dot definition depending on the selected algorithm
        # for now we assume vy is small
        v_tan = vx * cos(yaw - ref_heading)
        p = (pos_x - ref_x) * sin(ref_heading)  + (pos_y - ref_y) * -cos(ref_heading)
        den_corrected = self.soft_min(1+p*k,0.3)
        projection_ratio = 1 / den_corrected
        s_dot = v_tan * projection_ratio

        # forwards integrate the reference path
        x_ref_dot = s_dot * cos(ref_heading) 
        y_ref_dot = s_dot * sin(ref_heading)
        ref_heading_dot = k * s_dot

        return s_dot,x_ref_dot,y_ref_dot,ref_heading_dot
    

    def SVGP_continuous_dynamics(self,th_input,st_input,vx,vy,w,yaw,use_nominal_model):

        x_star = casadi.horzcat(th_input,st_input,vx,vy,w)
        mean_x, mean_y, mean_w = self.SVGP_unified_analytic_obj.predictive_mean_only(x_star)
        
        if use_nominal_model:
            #evaluate steering angle 
            steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)

            # # evaluate longitudinal forces
            Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                        + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)\
                        + self.F_friction_due_to_steering(steering_angle,vx,self.a_stfr_self,self.b_stfr_self,self.d_stfr_self,self.e_stfr_self)

            c_front = (self.m_front_wheel_self)/self.m_self
            c_rear = (self.m_rear_wheel_self)/self.m_self

            # redistribute Fx to front and rear wheels according to normal load
            Fx_front = Fx_wheels * c_front
            Fx_rear = Fx_wheels * c_rear

            #evaluate slip angles
            alpha_f,alpha_r = self.evaluate_slip_angles(vx,vy,w,self.lf_self,self.lr_self,steering_angle)

            #lateral forces
            Fy_wheel_f = self.lateral_tire_force(alpha_f,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self)
            Fy_wheel_r = self.lateral_tire_force(alpha_r,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self)

            acc_x_dyn_bike,acc_y_dyn_bike,acc_w_dyn_bike = self.solve_rigid_body_dynamics(vx,vy,w,steering_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)
            
            acc_x = mean_x + acc_x_dyn_bike
            acc_y = mean_y + acc_y_dyn_bike
            acc_w = mean_w + acc_w_dyn_bike
        else:
            acc_x = mean_x
            acc_y = mean_y
            acc_w = mean_w

        xdot = self.produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w)
        
        return xdot



    def soft_min(self, x, min_val):
        sharpness = 10
        return min_val + 0.5*(1 + np.tanh(sharpness*(x-min_val)))*(x-min_val)
    
    def single_layer_planner_continous_dynamics_forces(self, x, u, p):
        z = casadi.vertcat(u, x)
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past= self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)
        return self.single_layer_continous_dynamics(local_path_length,labels_k,
                                                    th_input,st_input,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading)
    


    

    # # def single_layer_discrete_dynamics_actuators_forces(self,z, p):
    # #     #z = casadi.vertcat(u, x)
    # #     th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(z)
    # #     local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)

    # #     # evaluate the th and st using FIR response
    # #     th_4_model,st_4_model,th_past_next,st_past_next = self.act_dynamics(th_input,st_input,th_past, st_past)

    # #     u = casadi.vertcat(th_input,st_input,slack)  # TEMP DEBUGGING
    # #     x = casadi.vertcat(pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past)

    # #     import forcespro
    # #     dt_solver = self.time_horizon / self.N
    # #     x_next_state = forcespro.nlp.integrate(self.single_layer_planner_continous_dynamics_forces_4_integrator, x, u, p,
    # #                                         integrator=forcespro.nlp.integrators.RK4,
    # #                                         stepsize=dt_solver)
    # #     # unpack the new state
    # #     pos_x_next = x_next_state[0]
    # #     pos_y_next = x_next_state[1]
    # #     yaw_next = x_next_state[2]
    # #     vx_next = x_next_state[3]
    # #     vy_next = x_next_state[4]
    # #     w_next = x_next_state[5]
    # #     s_next = x_next_state[6]
    # #     ref_x_next = x_next_state[7]
    # #     ref_y_next = x_next_state[8]
    # #     ref_heading_next = x_next_state[9]

    # #     # assemble new state
    # #     x_next = casadi.vertcat(pos_x_next,
    # #                             pos_y_next,
    # #                             yaw_next,
    # #                             vx_next,
    # #                             vy_next,
    # #                             w_next,
    # #                             s_next,
    # #                             ref_x_next,
    # #                             ref_y_next,
    # #                             ref_heading_next,
    # #                             th_past_next,
    # #                             st_past_next)
    # #     return x_next


    def single_layer_discrete_dynamics_actuators_forces(self,z, p):
        #z = casadi.vertcat(u, x)
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)

        return self.single_layer_discrete_dynamics_with_actuators(local_path_length,labels_k,
                                                    th_input,st_input,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past)




    def act_dynamics(self,th_input,st_input,th_past, st_past):
        # evaluate the th and st 
        th_4_model = th_input * self.weights_th_FIR_solver[0] + \
                    np.expand_dims(self.weights_th_FIR_solver[1:],0) @ th_past 
        st_4_model = st_input * self.weights_st_FIR_solver[0] + \
                    np.expand_dims(self.weights_st_FIR_solver[1:],0) @ st_past

        # update past actions
        th_past_next = casadi.vertcat(th_input,th_past[1:])
        st_past_next = casadi.vertcat(st_input,st_past[1:])

        return th_4_model,st_4_model,th_past_next,st_past_next




    def single_layer_discrete_dynamics_with_actuators(self,local_path_length,labels_k,
                                                    th_input,st_input,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past):
        # evaluate the dynamics with the FIR-based actions
        th_4_model,st_4_model,th_past_next,st_past_next = self.act_dynamics(th_input,st_input,th_past, st_past)

        # now evaluate the dynamics with the FIR-based actions
        dt_solver = self.time_horizon / self.N
        dt_vehicle = dt_solver / self.discrete_dynamics_intermediate_shooting_vehicle
        
        for _ in range(self.discrete_dynamics_intermediate_shooting_vehicle):
            
            pos_x_dot, pos_y_dot, yaw_dot, vx_dot, vy_dot, w_dot = self.dynamic_bicycle_continuous_dynamics(th_input,st_input,vx,vy,w,yaw)
            # integrating with simple Euler
            # th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_pas
            pos_x = pos_x + pos_x_dot * dt_vehicle
            pos_y = pos_y + pos_y_dot * dt_vehicle
            yaw = yaw + yaw_dot * dt_vehicle
            vx = vx + vx_dot * dt_vehicle
            vy = vy + vy_dot * dt_vehicle
            w = w + w_dot * dt_vehicle


        # update other states
        dt_path = dt_solver / self.discrete_dynamics_intermediate_shooting_path
        for _ in range(self.discrete_dynamics_intermediate_shooting_path):
            s_dot,x_ref_dot,y_ref_dot,ref_heading_dot = self.kernelized_path_derivatives(local_path_length,labels_k,s,ref_x,ref_y,ref_heading,vx,yaw,pos_x,pos_y)
            s = s + s_dot * dt_path
            ref_x = ref_x + x_ref_dot * dt_path
            ref_y = ref_y + y_ref_dot * dt_path
            ref_heading = ref_heading + ref_heading_dot * dt_path


        # assemble new state
        x_next = casadi.vertcat(pos_x,
                                pos_y,
                                yaw,
                                vx,
                                vy,
                                w,
                                s,
                                ref_x,
                                ref_y,
                                ref_heading,
                                th_past_next,
                                st_past_next)
        return x_next




    def lane_boundary_constraint(self,pos_x,pos_y,ref_x,ref_y,slack,lane_width):
        return ((lane_width+slack)/2)**2 - ((pos_x - ref_x)**2  + (pos_y - ref_y)**2)  

    def max_centrifugal_force_constraint(self,vx,w,slack,st_input):
        if self.dynamic_model == "kinematic_bicycle":
            steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
            w_constr = vx * np.tan(steering_angle) / (self.lf_self+self.lr_self)
        elif self.dynamic_model == "dynamic_bicycle" or self.dynamic_model == "dynamic_bicycle_GP":
            w_constr = w
        # evaluate linear contraint on the maximum centrifugal force
        # vx = 4.6 --> w = 0 (max vx)
        # vx = 2.5 --> w = -3.8 (observed point when car lifts wheels off the ground)
        slope = 3.8 / (2.5 - 4.6)
        w0 = - slope * 4.6
        w0 = w0 * 0.5

        h1 = vx*slope - w_constr + w0 + slack
        h2 = vx*slope + w_constr + w0 + slack
        return [h1,h2]


    def non_lin_constraint_forces(self,z, p):
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading, th_past, st_past = self.unpack_state(z)
        local_path_length, q_con, q_u, q_acc, qt_pos, qt_rot, lane_width, qt_s_high, q_v,labels_k = self.unpack_parameters(p)
        # ,*self.max_centrifugal_force_constraint(vx,w,slack,st_input)
        return [self.lane_boundary_constraint(pos_x,pos_y,ref_x,ref_y,slack,lane_width)]



    def produce_X0(self,V_target,local_path_length,labels_k,labels_s):
        # V_target in this case is the estimated speed of the vehicle 

        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # 0        1        2     3     4     5   6  7  8 9 10    11    12 
        # th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading

        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, V_target * self.time_horizon, N_0+1)

        # interpolate to get curvature values
        #normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        k_0_vals = np.interp(s_star_0, labels_s, labels_k)
        x_ref_0 = np.zeros(N_0+1)
        y_ref_0 = np.zeros(N_0+1)
        ref_heading_0 = np.zeros(N_0+1)
        dt = self.time_horizon / N_0
        yaw_rate_0 = np.zeros(N_0+1)
        for i in range(1,N_0+1):
            x_ref_0[i] = x_ref_0[i-1] + V_target * dt * np.cos(ref_heading_0[i-1])
            y_ref_0[i] = y_ref_0[i-1] + V_target * dt * np.sin(ref_heading_0[i-1])
            ref_heading_0[i] = ref_heading_0[i-1] + k_0_vals[i-1] * V_target * dt

            yaw_rate_0[i-1] = (ref_heading_0[i] - ref_heading_0[i-1] )/ dt

        # get throttle value
        throttle_search_vec = np.linspace(0,1,30)
        # evalaute FX on the throttle search vec
        Fx_wheels = + self.motor_force(throttle_search_vec,V_target,self.a_m_self,self.b_m_self,self.c_m_self)\
                + self.rolling_friction(V_target,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration
        #find the throttle that gives the closest acceleration to 0
        throttle_0 = throttle_search_vec[np.argmin(np.abs(acc_x))]


        # now down sample to the N points
        s_0_vec = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), s_0_vec)
        x_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), x_ref_0)
        y_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), y_ref_0)
        ref_heading_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), ref_heading_0)
        yaw_rate_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), yaw_rate_0)


        # assign values to the array
        # 0        1        2     3     4     5   6  7  8 9 10    11    12 
        # th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w,s,ref_x,ref_y,ref_heading

        X0_array[:,0] = throttle_0
        X0_array[:,1] = 0 # steering is 0 for now
        X0_array[:,2] = 0 # slack variable should be zero
        
        X0_array[:,3] = x_ref_0 
        X0_array[:,4] = y_ref_0
        X0_array[:,5] = ref_heading_0
        X0_array[:,6] = V_target
        X0_array[:,7] = 0
        X0_array[:,8] = yaw_rate_0
        X0_array[:,9] = s_0_vec
        X0_array[:,10] = x_ref_0
        X0_array[:,11] = y_ref_0
        X0_array[:,12] = ref_heading_0



        return X0_array






class generate_path_labels(): # inherits from DART system identification

    def __init__(self):
        
        self.solver_name_acados = 'path_labels_generator_acados'
        self.solver_name_forces = 'path_labels_generator_forces'

        self.N = 41 # number of points in the kernelized path
        self.nu = 1 
        self.nx = 4
        self.n_parameters = 3

        # generate fixed path quantities
        #n points kernelized is the number of points used to kernelize the path but defined above to get the nparamters right
        try:
            from path_track_definitions import generate_fixed_path_quantities
        except:
            from .path_track_definitions import generate_fixed_path_quantities
        self.path_lengthscale = 1.3/self.N
        lambda_val = 0.0001**2
        self.Kxx_inv, self.normalized_s_4_kernel_path = generate_fixed_path_quantities(self.path_lengthscale,
                                                                            lambda_val,
                                                                            self.N)

        

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


        # unpack parameters
        path_x, path_y, path_yaw = model.p[0], model.p[1], model.p[2]

        # assign dynamic constraint
        model.f_expl_expr = vertcat(*self.continous_dynamics(x, u)) # make into vertical vector for casadi

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(pos_x,pos_y,ref_x,ref_y,ref_heading,u_yaw_dot,slack,q_con,q_lag,q_u,s,qt_s_high) 
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(ref_heading, yaw,pos_x,pos_y,ref_x,ref_y,qt_pos,qt_rot,s,qt_s_high,V_target)



        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of points in the kernelized path
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost functions ---
        ocp.model.cost_expr_ext_cost  =  self.objective(x, path_x, path_y, path_yaw) 

                
        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # GAUSS_NEWTON, EXACT
        ocp.solver_options.integrator_type = 'ERK' # IRK, ERK
        ocp.solver_options.sim_method_num_steps = 1  # Number of sub-steps in each interval for integration purpouses
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP   SQP_RTI
        ocp.solver_options.tf = 1  # time horizon in seconds

        # messing with the convergence criteria
        ocp.solver_options.qp_solver_warm_start = 1 # 0: no warm start, 1: warm start 2 : hot start
        ocp.solver_options.globalization = 'FIXED_STEP' # 'MERIT_BACKTRACKING', 'FIXED_STEP' # fixed is the default
        ocp.solver_options.nlp_solver_max_iter = 20  # Maximum SQP iterations
        ocp.solver_options.qp_solver_iter_max = 5
        ocp.solver_options.print_level = 0 # no print
        ocp.solver_options.tol = 0.001

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
        model.lb = np.array([-self.max_yaw_rate, 0.0 ,   0.0, -1000, -1000,-1000,-0.2])  # lower bound on inputs
        model.ub = np.array([+self.max_yaw_rate, +100, self.max_sdot, +1000, +1000,+1000,+1000])  # upper bound on inputs

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
        codeoptions.nlp.TolStat = 1e-3  # inf norm tol. on stationarity
        codeoptions.nlp.TolEq = 1e-3  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1e-3  # tol. on inequality constraints
        codeoptions.nlp.TolComp = 1e-3  # tol. on complementarity

        # set warm start behaviour for dual variables (so always warm start from solver perspective, even if in practice you give it a vector of zeros)
        codeoptions.init = 2  # 0 cold, 1 centered, 2 warm

        #set overwrite behviour
        codeoptions.overwrite = 1 # 0 never, 1 always, 2 (Defaul) ask

        codeoptions.solvemethod = 'SQP_NLP' # 'PDIP_NLP' # changing to non linear primal dual method  'SQP_NLP'
        # NOTE that by default the solver uses a single sqp iteration so you need to increase the number of iterations
        #codeoptions.nlp.hessian_approximation = 'gauss-newton'
        #codeoptions.solver_timeout = 1  # Set a 40 ms time limit we assume the controller rate is 20Hz but you need some time to do other things in the control loop
        codeoptions.solver_exit_external = 1
        codeoptions.sqp_nlp.maxqps = 4
        codeoptions.sqp_nlp.maxSQPit = 10
        codeoptions.sqp_nlp.reg_hessian = 1e-6  # regularization of hessian (default is 5 * 10^(-9))
        #codeoptions.sqp_nlp.use_line_search = False  # Enable line search (default)

        codeoptions.parallel = 1 # this doesn't really do much


        return model,codeoptions

    def objective(self, u, x,y,yaw, path_x, path_y, path_yaw):
        # evalaute the cost
        q_k = 1
        q_pos = 1
        q_yaw = 1

        j =  q_pos * (path_x - x) ** 2 +\
                q_pos * (path_y - y) ** 2 +\
                q_yaw * (path_yaw - yaw) ** 2+\
                q_k * u ** 2
        
        return j
    
    def continous_dynamics(self, x, u):
        # K_x_star = K_matern2_kernel(s_star, normalized_s_4_kernel_path,
        #                         path_lengthscale,1,self.n_points_kernelized)      
        # left_side = K_x_star @ Kxx_inv
        # k = left_side @ labels_k

        # unpack state
        x,y,yaw,s = x[0],x[1],x[2],x[3]
        # unpack input
        u_yaw_dot = u[0]
        # solve the "dynamics"
        # ---
        # set up initial state to 0
        x_dot = np.cos(yaw)
        y_dot = np.sin(yaw)
        yaw_dot = u_yaw_dot
        s_dot = 1
        # ---
        return [x_dot,y_dot,yaw_dot,s_dot]
    

    def objective_static(self, u_labels_k, path_x, path_y, path_yaw):
        # produce k values
        # if not using RK4 this does not make sense because the k values are exactly the same as the u_labels_k
        ds = 1/self.nx
        # s_star = np.linspace(0,1,self.nx) # normalize s
        #K_x_star = K_matern2_kernel(s_star, self.normalized_s_4_kernel_path,self.path_lengthscale,1,self.nx)      
        # left_side = K_x_star @ self.Kxx_inv
        # k_vals = left_side @ u_labels_k

        k_vals = u_labels_k

        # solve the "dynamics"
        # ---
        # set up initial state to 0
        if type(u_labels_k) == casadi.SX:
            x = casadi.SX.zeros(self.nx)
            y = casadi.SX.zeros(self.nx)
            yaw = casadi.SX.zeros(self.nx)
        elif type(u_labels_k) == casadi.MX:
            x = casadi.MX.zeros(self.nx)
            y = casadi.MX.zeros(self.nx)
            yaw = casadi.MX.zeros(self.nx)
        else:
            x = np.zeros(self.nx)
            y = np.zeros(self.nx)
            yaw = np.zeros(self.nx)



        for i in range(1,self.nx):
            x[i] = x[i-1] + ds * np.cos(yaw[i-1]) 
            y[i] = y[i-1] + ds * np.sin(yaw[i-1])
            yaw[i] = yaw[i-1] + k_vals[i-1] * ds
        # ---
            
        # evalaute the cost
        q_k = 1
        q_pos = 1
        q_yaw = 1

        loss =  q_pos * (path_x - x) ** 2 +\
                q_pos * (path_y - y) ** 2 +\
                q_yaw * (path_yaw - yaw) ** 2+\
                q_k * k_vals ** 2
        
        return loss


    
    def objective_forces(self, z, p):
        path_x, path_y, path_yaw = p[0], p[1], p[2]
        return self.objective(z, path_x, path_y, path_yaw)


    def produce_X0(self,V_target,local_path_length,labels_x, labels_y, labels_heading):
        # NOTE this needs to be updated to include the slack variable
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # z = yaw_dot slack s_dot x y yaw s 
        #     0       1     2     3 4   5 6 

        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + V_target * 1.5, N_0+1)

        # interpolate to get kurvature values
        normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        
        x_ref_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_x)
        y_ref_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_y)
        ref_heading_0 = np.interp(s_star_0, normalized_s_4_kernel_path, labels_heading)

        # now down sample to the N points
        s_0_vec = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), s_0_vec)
        x_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), x_ref_0)
        y_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), y_ref_0)
        ref_heading_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), ref_heading_0)
        u_yaw_rate_0 = np.diff(ref_heading_0) / (self.time_horizon / self.N)
        u_yaw_rate_0 = np.append(u_yaw_rate_0, u_yaw_rate_0[-1])         # append last value again to make the array the same size

        # assign values to the array
        X0_array[:,0] = u_yaw_rate_0
        X0_array[:,1] = np.zeros(self.N+1) # slack variable should be zero
        X0_array[:,2] = V_target # s_dot can be around V_target
        X0_array[:,3] = x_ref_0
        X0_array[:,4] = y_ref_0
        X0_array[:,5] = ref_heading_0
        X0_array[:,6] = s_0_vec


        return X0_array
