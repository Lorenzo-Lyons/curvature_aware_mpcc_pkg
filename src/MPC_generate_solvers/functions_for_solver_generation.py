import numpy as np
import forcespro.nlp
try:
    from .path_track_definitions import K_RBF_kernel, K_matern2_kernel
except:
    from path_track_definitions import K_RBF_kernel, K_matern2_kernel

import casadi
import math


# This assumes that the DART system identification package is installed
from DART_dynamic_models.dart_dynamic_models import model_functions 



class generate_high_level_path_planner_ocp(): # inherits from DART system identification

    def __init__(self,MPC_algorithm):
        
        self.MPC_algorithm = MPC_algorithm
        self.solver_name = 'high_level_reference_generator_' + MPC_algorithm

        self.n_points_kernelized = 41 # number of points in the kernelized path (41 for reference)
        self.time_horizon = 1.5
        self.N = 15 # stages
        self.max_yaw_rate = 10 # based on w = V / R = k * V so 2 * V is the maximum yaw rate 
        self.nx = 7
        self.nu = 1
        self.n_parameters = 8 + self.n_points_kernelized
        
    
    def produce_ocp(self):

        try:
            from path_track_definitions import generate_fixed_path_quantities
        except:
            from .path_track_definitions import generate_fixed_path_quantities

        from casadi import vertcat, MX
        from acados_template import  AcadosOcp, AcadosModel

        # seting up ACADOS solver
        x = MX.sym('x', self.nx)
        u = MX.sym('u', self.nu)
        p = MX.sym('p', self.n_parameters) # stage-wise parameters


        # Create model object
        model = AcadosModel()
        model.name = self.solver_name
        model.x = x
        model.u = u
        model.p = p

        #unpack states
        u_yaw_dot = model.u[0]
        pos_x = model.x[0]
        pos_y = model.x[1]
        yaw = model.x[2]
        s = model.x[3]
        ref_x = model.x[4] # path reference point x
        ref_y = model.x[5] # path reference point y
        ref_heading = model.x[6] # path reference heading

        # unpack parameters
        V_target = model.p[0]
        local_path_length = model.p[1]
        q_con = model.p[2]
        q_lag = model.p[3]
        q_u = model.p[4]
        qt_pos = model.p[5]
        qt_rot = model.p[6]
        lane_width = model.p[7]
        labels_k = model.p[8:]

        # evalaute curvature of the path as a function of s
        # n points kernelized is the number of points used to kernelize the path but defined above to get the nparamters right
        path_lengthscale = 1.3/self.n_points_kernelized
        lambda_val = 0.0001**2
        Kxx_inv, normalized_s_4_kernel_path = generate_fixed_path_quantities(path_lengthscale,
                                                                            lambda_val,
                                                                            self.n_points_kernelized)
        s_star = s / local_path_length # normalize s
        try:
            from path_track_definitions import K_matern2_kernel
        except:
            from .path_track_definitions import K_matern2_kernel
            
        K_x_star = K_matern2_kernel(s_star, normalized_s_4_kernel_path,
                                path_lengthscale,1,self.n_points_kernelized)      
        left_side = K_x_star @ Kxx_inv
        k = left_side @ labels_k

        # --- define the dynamic constraint ---
        # "robot" moving at constant speed
        x_dot = V_target * np.cos(yaw)
        y_dot = V_target * np.sin(yaw)
        yaw_dot = u_yaw_dot

        # s_dot definition depending on the selected algorithm
        if self.MPC_algorithm == 'MPCC':
            s_dot = V_target
        else:
            v_tan = V_target * np.cos(yaw - ref_heading)
            p = (pos_x - ref_x) * np.sin(ref_heading)  + (pos_y - ref_y) * -np.cos(ref_heading)

            projection_ratio = 1 / (1+p*k )

            s_dot = v_tan * projection_ratio

        x_ref_dot = s_dot * np.cos(ref_heading) 
        y_ref_dot = s_dot * np.sin(ref_heading)
        ref_heading_dot = k * s_dot

        state_dot = [x_dot,y_dot, yaw_dot, s_dot ,x_ref_dot, y_ref_dot, ref_heading_dot]
        model.f_expl_expr = vertcat(*state_dot) # make into vertical vector for casadi


        # generate optimal control problem
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # number of stages
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # --- set up the cost function ---
        # stage cost
        if self.MPC_algorithm == 'MPCC':
            err_lag_squared = ((pos_x - ref_x) *  np.cos(ref_heading)  + (pos_y - ref_y) * np.sin(ref_heading)) ** 2
            err_lat_squared = ((pos_x - ref_x) * -np.sin(ref_heading) + (pos_y - ref_y) * np.cos(ref_heading)) ** 2

            j = q_con * err_lat_squared +\
                     q_lag * err_lag_squared +\
                     q_u * u_yaw_dot ** 2
        else:
            # cost function for CAMPCC
            # penalize deviation from the path only since the s_dot integration is much more precise
            err_lat_squared = (pos_x - ref_x)**2 + (pos_y - ref_y)**2
            j = q_con * err_lat_squared +\
                     q_u * u_yaw_dot ** 2

        # terminal cost
        dot_direction = (np.cos(ref_heading) * np.cos(yaw)) + (np.sin(ref_heading) * np.sin(yaw)) # evaluate car angle relative to a straight path
        misalignment = -dot_direction # incentivise alligning with the path
        # higher penalty costs on v and path tracking, plus an dditional penalty for not alligning with the path at the end
        err_pos_squared_t = (pos_x - ref_x)**2 + (pos_y - ref_y)**2
        j_term =    qt_pos * err_pos_squared_t + \
                    qt_rot * misalignment

        # asign cost functions
        ocp.model.cost_expr_ext_cost  =  j 
        ocp.model.cost_expr_ext_cost_e =  j_term
                

        # constraints
        #ocp.constraints.constr_type = 'BGH'
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([-self.max_yaw_rate])
        ocp.constraints.ubu = np.array([+self.max_yaw_rate]) 

        # # define lane boundary constraints
        # h = (pos_x - ref_x)**2  + (pos_y - ref_y)**2
        # # add constraint on projection ratio
        # ocp.model.con_h_expr = h  # Define h(x, u)
        # ocp.constraints.lh = np.array([0.0])  # Lower bound (h_min)
        # ocp.constraints.uh = np.array([(1/2)**2])  # Upper bound (h_max)

        # # copy for terminal
        # ocp.constraints.uh_e = ocp.constraints.uh
        # ocp.constraints.lh_e = ocp.constraints.lh
        # ocp.model.con_h_expr_e = ocp.model.con_h_expr

        # Initial state constraint
        ocp.constraints.x0 = np.zeros(self.nx)  # This is a default value, it will be updated at runtime

        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'EXACT' # GAUSS_NEWTON, EXACT
        ocp.solver_options.integrator_type = 'ERK' # IRK, ERK
        ocp.solver_options.sim_method_num_steps = 1  # Number of sub-steps in each interval for integration purpouses
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP   SQP_RTI
        ocp.solver_options.tf = self.time_horizon  # time horizon in seconds

        # messing with the convergence criteria
        ocp.solver_options.qp_solver_warm_start = 1 # 0: no warm start, 1: warm start 2 : hot start
        ocp.solver_options.globalization = 'FIXED_STEP' # 'MERIT_BACKTRACKING', 'FIXED_STEP' # fixed is the default
        #ocp.solver_options.nlp_solver_max_iter = 20  # Maximum SQP iterations
        #ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.print_level = 0 # no print
        ocp.solver_options.tol = 0.001
        #ocp.solver_options.globalization_fixed_step_length = 0.001

        # Initialize parameters with default values (this step is important to avoid dimension mismatch)
        ocp.parameter_values = np.zeros(self.n_parameters)
        return ocp
    
    def produce_X0(self,V_target,local_path_length,labels_k_params):
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N+1,self.nu +  self.nx))
        # z = yaw_dot x y yaw s ref_x ref_y ref_heading
        #     0       1 2 3   4 5     6     7  

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
        X0_array[:,1] = x_ref_0
        X0_array[:,2] = y_ref_0
        X0_array[:,3] = ref_heading_0
        X0_array[:,4] = s_0_vec
        X0_array[:,5] = x_ref_0
        X0_array[:,6] = y_ref_0
        X0_array[:,7] = ref_heading_0

        return X0_array


class generate_low_level_solver_ocp(model_functions): # inherits from DART system identification

    def __init__(self,dynamic_model):
        # features to add:
        # 1. actuator delay
        # 2. inequality constraints

        # store for later
        self.dynamic_model = dynamic_model
        
        # chose options
        self.N = 15 # this should match the high level planner (at least for now)
        self.time_horizon = 1.5
        self.solver_name_acados = 'low_level_acados_' + dynamic_model # 'dynamic_bicycle', 'kinematic_bicycle', 'SVGP'
        self.nx = 6
        self.nu = 3
        self.n_parameters = 12

        self.solver_name_forces = 'low_level_forces_' + dynamic_model # 'dynamic_bicycle', 'kinematic_bicycle', 'SVGP'
        

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
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, slack_p_1, q_acc,x_ref, y_ref, yaw_ref = self.unpack_parameters(model.p)
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
        ocp.model.cost_expr_ext_cost  =  self.evalaute_objective(th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                                                                x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u, slack_p_1,q_acc)
        # terminal cost
        ocp.model.cost_expr_ext_cost_e =  self.objective_terminal_cost(pos_x,pos_y,yaw,vx,x_ref, y_ref, yaw_ref,V_target,q_v, qt_pos, qt_rot)
        
        # constraints
        ocp.constraints.constr_type = 'BGH'
        # u = [throttle, steer, slack_var]
        ocp.constraints.lbu = np.array([0,-1, 0])
        ocp.constraints.ubu = np.array([+1,+1, 100]) # high value for slack variable
        ocp.constraints.idxbu = np.array([0, 1, 2])

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

        ocp.solver_options.qp_solver_iter_max = 10
        ocp.solver_options.nlp_solver_max_iter = 1000

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
        model.nh = 0
        model.hu = np.array([])
        model.hl = np.array([])


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
        slack_p_1 =  p[7]       # slack cost tuning weight
        q_acc =      p[8]            # acceleration cost tuning weight
        x_ref =      p[9]            # reference position x
        y_ref =      p[10]           # reference position y
        yaw_ref =    p[11]         # reference yaw

        return V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, slack_p_1, q_acc, x_ref, y_ref, yaw_ref


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

        acc_x,acc_y,acc_w = self.solve_rigid_body_dynamics(vx,vy,w,steering_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)
        
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
        return np.array(self.kinematic_bicycle_continuous_dynamics(th_input,st_input,vx,yaw))

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
        return np.array(self.dynamic_bicycle_continuous_dynamics(th_input,st_input,vx,vy,w,yaw))



    def evalaute_objective(self,th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                            x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u, slack_p_1, q_acc):
        
        error_position_sqrd = (pos_x - x_ref)**2 + (pos_y - y_ref)**2
        error_heading_sqrd = (yaw - yaw_ref)**2

        # evalaute acceleration to penalize it

        # from kinemaitc bicycle model
        steering_angle = self.steering_2_steering_angle(st_input,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
        Fx_wheels = self.motor_force(th_input,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
                    #+ self.F_friction_due_to_steering(steering_angle,vx,self.a_stfr_self,self.b_stfr_self,self.d_stfr_self,self.e_stfr_self)
        
        acc_x =  Fx_wheels / self.m_self 



        j = q_v * (vx - V_target) ** 2 + \
            q_pos * error_position_sqrd +\
            q_rot * error_heading_sqrd +\
            q_u * st_input ** 2 +\
            q_u * th_input ** 2 +\
            slack_p_1 * slack ** 2+\
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
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, slack_p_1, q_acc,x_ref, y_ref, yaw_ref = self.unpack_parameters(p) 
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
        return self.evalaute_objective(th_input,st_input,slack, pos_x,pos_y,yaw,vx,vy,w,\
                            x_ref, y_ref, yaw_ref,V_target,q_v, q_pos, q_rot, q_u, slack_p_1,q_acc)


    def objective_terminal_forces(self,z,p):
        V_target, q_v, q_pos, q_rot, q_u, qt_pos, qt_rot, slack_p_1, q_acc, x_ref, y_ref, yaw_ref = self.unpack_parameters(p) 
        th_input,st_input,slack,pos_x,pos_y,yaw,vx,vy,w = self.unpack_state(z)
        return self.objective_terminal_cost(pos_x,pos_y,yaw,vx, x_ref, y_ref, yaw_ref,V_target,q_v, qt_pos, qt_rot)


    def produce_X0(self,V_target, output_array_high_level):
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
        X0_array[:,3] = output_array_high_level[:,1]
        X0_array[:,4] = output_array_high_level[:,2]
        X0_array[:,5] = output_array_high_level[:,3]
        X0_array[:,6] = V_target # assign target speed as first guess 
        X0_array[:,8] = output_array_high_level[:,0] # input of high level is the yaw rate

        return X0_array



