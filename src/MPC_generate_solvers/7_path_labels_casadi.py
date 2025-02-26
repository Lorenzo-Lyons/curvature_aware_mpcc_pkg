import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
try:
    from .path_track_definitions import K_RBF_kernel, K_matern2_kernel
except:
    from path_track_definitions import K_RBF_kernel, K_matern2_kernel




# this script builds the high level MPC solver for the MPCC++ algorithm 
solver_name = 'path_labels_generator_casadi'
# generate locations to where to build the solvers
current_script_path = os.path.realpath(__file__)
current_script_dir = os.path.dirname(current_script_path)
path_to_built_solvers_folder = os.path.join(current_script_dir,'solvers')





N = 41 # number of points in the kernelized path
n_parameters = 3 * N + 1 

# generate fixed path quantities
#n points kernelized is the number of points used to kernelize the path but defined above to get the nparamters right
try:
    from path_track_definitions import generate_fixed_path_quantities
except:
    from .path_track_definitions import generate_fixed_path_quantities
path_lengthscale = 1.3/N
lambda_val = 0.0001**2
Kxx_inv, normalized_s_4_kernel_path = generate_fixed_path_quantities(path_lengthscale,
                                                                    lambda_val,
                                                                    N)


# define the cost function  
# maybe later have them as parameters
#ds = 1/N #* path_length

q_pos = 10
q_yaw = 1
q_k = 0.01
q_k_dev = 0.01

# define decreasing weights for the k values to favour fitting the path well closer to the origin
Q_u = np.diag(np.linspace(10,0.01,N))


def objective_static(u_labels_k, path_x, path_y, path_yaw, path_k, path_length_p):
    # produce k values
    # if not using RK4 this does not make sense because the k values are exactly the same as the u_labels_k
    ds_int = path_length_p * 1/N
    # solve the "dynamics"
    x_vec = ca.MX.zeros(N)
    y_vec = ca.MX.zeros(N)
    yaw_vec = ca.MX.zeros(N)

    s_star = ca.linspace(0,1,N) # normalize s


    for i in range(1,N):

        # add intermediate shooting nodes to add precision (in acados-forces it will be done with RK4)
        intermediate_shooting_nodes = 1
        s_star_jj = s_star[i-1]
        x_jj = x_vec[i-1]
        y_jj = y_vec[i-1]
        yaw_jj = yaw_vec[i-1]

        for jj in range(intermediate_shooting_nodes):
            # evaluate the dynamics
            K_x_star = K_matern2_kernel(s_star_jj, normalized_s_4_kernel_path,path_lengthscale,1,N)      
            left_side = K_x_star @ Kxx_inv
            k = left_side @ u_labels_k

            s_star_jj = s_star_jj + ds_int/intermediate_shooting_nodes
            x_jj = x_jj + ds_int/intermediate_shooting_nodes *ca.cos(yaw_vec[i-1]) 
            y_jj = y_jj + ds_int/intermediate_shooting_nodes * ca.sin(yaw_vec[i-1])
            yaw_jj = yaw_jj + k * ds_int/intermediate_shooting_nodes

        # save the values
        x_vec[i] = x_jj 
        y_vec[i] = y_jj 
        yaw_vec[i] = yaw_jj 
            


        # # x_vec[i] = x_vec[i-1] + ds_int *ca.cos(yaw_vec[i-1]) 
        # # y_vec[i] = y_vec[i-1] + ds_int * ca.sin(yaw_vec[i-1])
        # # yaw_vec[i] = yaw_vec[i-1] + k * ds_int
        
    # ---
    # evalaute the k values derivative
    dk_ds = (u_labels_k[1:] - u_labels_k[:-1]) / (ds_int)


    # compute loss
    loss =  q_pos * ca.transpose(path_x - x_vec) @ Q_u @ (path_x - x_vec) +\
            q_pos * ca.transpose(path_y - y_vec) @ Q_u @ (path_y - y_vec) +\
            q_yaw * ca.transpose(path_yaw - yaw_vec) @ Q_u @ (path_yaw - yaw_vec) +\
            q_k * ca.transpose(u_labels_k) @ (u_labels_k) +\
            q_k_dev   * ca.transpose(dk_ds) @ (dk_ds)
    
    return loss















# Define optimization variable
u_labels_k = ca.MX.sym('u_labels_k', N)  # Control variables

# Define path reference variables (these should be given as parameters)
path_x = ca.MX.sym('path_x', N)
path_y = ca.MX.sym('path_y', N)
path_yaw = ca.MX.sym('path_yaw', N)
path_k = ca.MX.sym('path_k', N)
path_length_p = ca.MX.sym('path_length', 1)

# Compute objective
cost = objective_static(u_labels_k, path_x, path_y, path_yaw, path_k,path_length_p)

# the upper and lower bounds you can actually set at runtime


# Setup NLP
nlp = {'x': u_labels_k, 'f': cost, 'p': ca.vertcat(path_x, path_y, path_yaw, path_k,path_length_p)}

# Create solver
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Generate C code for the solver
solver.generate_dependencies(solver_name + ".c")
print("C code generated: " + solver_name + ".c")



print('------------------------------')
print('All done with solver building!')
print('------------------------------')















# call the solver as a test
from path_track_definitions import generate_path_data

# racetrack generation
track_choice = 'racetrack_vicon_2' 
n_checkpoints = 100

s_vals_global_path,\
x_vals_global_path,\
y_vals_global_path,\
s_4_local_path,\
x_4_local_path,\
y_4_local_path,\
dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
k_vals_global_path,\
k_4_local_path,\
heading_angle_4_local_path = generate_path_data(track_choice,n_checkpoints)



# plot the path 4 local path
figure1, ax = plt.subplots(3,1)
ax[0].plot(x_4_local_path, y_4_local_path,label='path')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel('y [m]')
ax[0].axis('equal')
# now plot the heading 
ax[1].plot(s_4_local_path, heading_angle_4_local_path,label='heading')
ax[1].set_xlabel('s [m]')
ax[1].set_ylabel('heading [rad]')
# now plot the curvature
ax[2].plot(s_4_local_path, k_4_local_path,label='k')
ax[2].set_xlabel('s [m]')
ax[2].set_ylabel('k')

# define path length and k_vec
s_init = 6 #2.75 
path_length = 4.5 # 3 m/s for 1.5s time horizon 


# find the index of the point closest to the initial point
s_init_index = np.argmin(np.abs(s_4_local_path - s_init))
# find the index of the point closest to the final point
s_final_index = np.argmin(np.abs(s_4_local_path - (s_init + path_length)))
len_s = s_final_index - s_init_index




# define k_vec to be 0 until 1.5, then 1.33, then 0 again
s_vec = s_4_local_path[s_init_index:s_final_index] - s_4_local_path[s_init_index] # set it to 0
k_vec = k_4_local_path[s_init_index:s_final_index]


# integrate the k_vec to get the angle and then again to get the position
angle = np.zeros(len_s)
x_path = np.zeros(len_s)
y_path = np.zeros(len_s)

# integrate using the real k values
for i in range(1,len_s):
    angle[i] = angle[i-1] + k_vec[i-1]*(s_vec[i]-s_vec[i-1])
    x_path[i] = x_path[i-1] + np.cos(angle[i-1])*(s_vec[i]-s_vec[i-1])
    y_path[i] = y_path[i-1] + np.sin(angle[i-1])*(s_vec[i]-s_vec[i-1])




# plot the path
figure3, ax_p = plt.subplots()
ax_p.plot(x_path, y_path,label='path')
ax_p.set_xlabel('x [m]')
ax_p.set_ylabel('y [m]')
ax_p.axis('equal')

# plot the k_vec
figure2, ax_k = plt.subplots()
ax_k.plot(s_vec, k_vec, label='k',color='gray',alpha=0.3)
ax_k.set_xlabel('s [m]')
ax_k.set_ylabel('k_vec')













# Downsample path to N points
path_x_vals = np.interp(np.linspace(0, path_length, N), s_vec, x_path)
path_y_vals = np.interp(np.linspace(0, path_length, N), s_vec, y_path)
path_yaw_vals = np.interp(np.linspace(0, path_length, N), s_vec, angle)
path_k_vals = np.interp(np.linspace(0, path_length, N), s_vec, k_vec)  # use as initial guess 





# Solve problem
# Define lower and upper bounds for u_labels_k
R_max = 0.75  # Maximum curvature
lb_k = -1/R_max  # Lower bound
ub_k = +1/R_max  # Upper bound
lbx = np.full(N, lb_k)  # Lower bounds for all N elements
ubx = np.full(N, ub_k)  # Upper bounds for all N elements

solution = solver(p=np.concatenate([path_x_vals, path_y_vals, path_yaw_vals, path_k_vals,np.array([path_length])]), 
                  x0=np.zeros(N), 
                  lbx=lbx, 
                  ubx=ubx)






# Extract optimal k values
optimal_k = solution['x'].full().flatten()
#print("Optimal k values:", optimal_k)



# forwards integrate x y from the optimal k values as solver would do it
x_vec_plot = np.zeros(N)
y_vec_plot = np.zeros(N)
yaw_vec_plot = np.zeros(N)
ds = path_length / N
for i in range(1,N):
    x_vec_plot[i] = x_vec_plot[i-1] + ds * np.cos(yaw_vec_plot[i-1]) 
    y_vec_plot[i] = y_vec_plot[i-1] + ds * np.sin(yaw_vec_plot[i-1])
    yaw_vec_plot[i] = yaw_vec_plot[i-1] + optimal_k[i-1] * ds




#plot the x vals and y vals as datapoints
ax_p.scatter(path_x_vals, path_y_vals, label='path',color='red')
ax_p.plot(x_vec_plot, y_vec_plot, label='path',color='purple')


# plot the curvature solution
ax_k.plot(np.linspace(0,path_length,N), optimal_k, label='k',color='red')





plt.show()


