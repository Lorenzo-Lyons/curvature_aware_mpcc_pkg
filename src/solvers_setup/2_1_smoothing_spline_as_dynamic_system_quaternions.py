import numpy as np
import casadi as ca
import os, sys, roslib
from drone_dynamic_model import drone
# now visualise the state and input trajectories
import matplotlib.pyplot as plt

# ─────────────── ROS helper to locate extra code ──────────────
pkg_path = roslib.packages.get_pkg_dir('curvature_aware_mpcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))
from reference_path_handeling_functions import generate_path_data  # noqa

#track_choice = 'second_spline_race_track' ###
#track_choice = 'vicon_racetrack'
track_choice = 'spline_circle'



# s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
# s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
# dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec, \
# gates, gates_s_global_path, time_optimal_trajectory = generate_path_data(track_choice)

load_optimally_smoothed_path = False

s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2,k_vec, \
gates, gates_s, time_optimal_trajectory = generate_path_data(track_choice, load_optimally_smoothed_path)




# ─────────────── GLOBAL SETTINGS ──────────────
# the fianl time is free, I.e. it's the quantity to be optimised

ds_interval_target = 0.2 # ⬤ EDIT ME: discretization step (m)
N  = int(np.ceil(s_vals_global_path[-1]/ds_interval_target))           # ⬤ EDIT ME: control intervals
ds_interval_actual = s_vals_global_path[-1] / N  # actual discretization step (m)

nx = 7        #  px, py, pz, vx, vy, vz, qw, qx, qy, qz
nu = 2         # yaw_rate (about body z), pitch_rate (about body y)

# ─────────────── SYMBOLS ──────────────
x  = ca.MX.sym("x", nx)
u  = ca.MX.sym("u", nu)


# ─────────────── DYNAMICS ──────────────
def unpack_x(x):
    # px, py, pz, qw, qx, qy, qz
    return x[0], x[1], x[2], x[3], x[4], x[5], x[6]

def unpack_u(u):
    # yaw_rate (about body z), pitch_rate (about body y)
    return u[0], u[1]

def dynamics_rhs(x, u):
    px, py, pz, qw, qx, qy, qz = unpack_x(x)
    yaw_rate, pitch_rate = unpack_u(u)

    # Angular velocity in body frame
    wx, wy, wz = 0, pitch_rate, yaw_rate

    # Quaternion kinematics
    dq_w = -0.5 * ( qx*wx + qy*wy + qz*wz )
    dq_x =  0.5 * ( qw*wx + qy*wz - qz*wy )
    dq_y =  0.5 * ( qw*wy - qx*wz + qz*wx )
    dq_z =  0.5 * ( qw*wz + qx*wy - qy*wx )

    # Forward direction (x-axis of body in world frame)
    fx = 1 - 2*(qy**2 + qz**2)
    fy = 2*(qx*qy + qw*qz)
    fz = 2*(qx*qz - qw*qy)

    speed = 1.0  # fixed forward speed [m/s]

    dpx = speed * fx
    dpy = speed * fy
    dpz = speed * fz

    return ca.vertcat(dpx, dpy, dpz, dq_w, dq_x, dq_y, dq_z)





def rk4(xk, uk, h_step):
    k1 = dynamics_rhs(xk,              uk)
    k2 = dynamics_rhs(xk + 0.5*h_step*k1, uk)
    k3 = dynamics_rhs(xk + 0.5*h_step*k2, uk)
    k4 = dynamics_rhs(xk +       h_step*k3, uk)
    return xk + (h_step/6)*(k1 + 2*k2 + 2*k3 + k4)


# ─────────────── DECISION VECTOR & BOUNDS ──────────────
# resample the path according to the problem discretization
s_4_problem = np.linspace(s_vals_global_path[0], s_vals_global_path[-1], N+1)
x_4_problem = np.interp(s_4_problem, s_vals_global_path, x_vals_global_path)
y_4_problem = np.interp(s_4_problem, s_vals_global_path, y_vals_global_path)
z_4_problem = np.interp(s_4_problem, s_vals_global_path, z_vals_global_path)
# add tangent vectors
dx_ds_4_problem = np.interp(s_4_problem, s_4_local_path, dx_ds)
dy_ds_4_problem = np.interp(s_4_problem, s_4_local_path, dy_ds)
dz_ds_4_problem = np.interp(s_4_problem, s_4_local_path, dz_ds)

# re-normalize the tangent vectors to unit length
norms = np.linalg.norm(np.vstack((dx_ds_4_problem, dy_ds_4_problem, dz_ds_4_problem)), axis=0)
dx_ds_4_problem /= norms
dy_ds_4_problem /= norms
dz_ds_4_problem /= norms





# evaluate second order derivatives 4 first guess
d2x_ds2_4_problem = np.interp(s_4_problem, s_4_local_path, d2x_ds2)
d2y_ds2_4_problem = np.interp(s_4_problem, s_4_local_path, d2y_ds2)
d2z_ds2_4_problem = np.interp(s_4_problem, s_4_local_path, d2z_ds2)

centreline = np.vstack((x_4_problem, y_4_problem, z_4_problem,dx_ds_4_problem,dy_ds_4_problem,dz_ds_4_problem,d2x_ds2_4_problem,d2y_ds2_4_problem,d2z_ds2_4_problem)).T   # shape (N+1, 3)

# helper used inside the NLP construction
def generate_path_data_4_solver(k: int, N: int):
    centerline_point = centreline[k,:3]  # get the point at stage k
    tangent_vector = centreline[k,3:6]  # get the tangent vector at stage k
    acceleration_vector = centreline[k,6:9]  # get the second order devs at stage k
    return centerline_point ,tangent_vector, acceleration_vector

# define indexes of when the gates accur
gates_indexes_4_problem = []
for gate in gates:
    # evaulate the gate position on the problem discretization
    dist = np.linalg.norm(centreline[:, :3] - gate[:3], axis=1)
    index = np.argmin(dist)
    gates_indexes_4_problem.append(index)







def quat_from_x_axis(x_axis):
    """Return a quaternion [qw,qx,qy,qz] that aligns body x-axis with given x_axis."""
    x_axis = x_axis / np.linalg.norm(x_axis)   # normalize
    # Default forward axis in world frame
    ref = np.array([1.0, 0.0, 0.0])

    # Axis of rotation = cross product
    v = np.cross(ref, x_axis)
    c = np.dot(ref, x_axis)

    if np.allclose(v, 0):
        # Already aligned (or opposite)
        if c > 0:
            return np.array([1,0,0,0])   # identity
        else:
            # 180° rotation around y (or any axis perpendicular to ref)
            return np.array([0,0,1,0])   # rotate around z

    s = np.sqrt((1+c)*2)
    qw = 0.5*s
    qx, qy, qz = v / s
    return np.array([qw,qx,qy,qz])


from scipy.spatial.transform import Rotation as R
def quat_from_tangent_vec(tangent_vector):
    # tangent vector (normalized)
    tangent = tangent_vector # / np.linalg.norm(tangent_vector)

    # compute yaw and pitch from tangent
    yaw = np.arctan2(tangent[1], tangent[0])          # rotation around z
    pitch = np.arctan2(-tangent[2], np.sqrt(tangent[0]**2 + tangent[1]**2))  # rotation around y
    roll = 0.0                                        # roll = 0

    # create quaternion from yaw, pitch, roll (order: 'zyx')
    r = R.from_euler('zyx', [yaw, pitch, roll])
    q = r.as_quat()   # returns [x, y, z, w] by default in scipy

    # reorder to [w, x, y, z] for your convention
    q_wxyz_0 = np.array([q[3], q[0], q[1], q[2]])
    return q_wxyz_0








# define constraint values
lane_radius = 0.5
k_max = 0.67*(1.0 / lane_radius)  # maximum curvature (1/m)

# define smoothness weight
q_actuation = 10
q_length_path = 100  # path length weight
q_smoothness = 10  # smoothness weight (1/m)
#q_norm_penalty = 1000




w, w0, lbw, ubw = [], [], [], []
g, lbg, ubg     = [], [], []

# ─────────── 1) Initial state  X0  ─────────
# centerline_point, tangent_vector, acceleration_vector = generate_path_data_4_solver(0, N)
# # position guess = centerline point
# p0 = centerline_point
# # quaternion guess = align body-x with tangent vector
# q0 = quat_from_x_axis(tangent_vector)
# # build initial guess for state [px,py,pz,qw,qx,qy,qz]
# Xk_guess = np.concatenate([p0, q0])


# # initial state on the gate
# Xk = ca.MX.sym("X0", nx)
# w  += [Xk]
# w0 += [*Xk_guess] #[0]*nx
# lbw+= [-ca.inf]*nx
# ubw+= [ ca.inf]*nx


# First gate info
first_gate_idx = gates_indexes_4_problem[0]
gate_center = centreline[first_gate_idx, 0:3]         # position
tangent_vector = centreline[first_gate_idx, 3:6]      # tangent

# Build quaternion aligned with tangent, roll=0
# We can rotate x-axis to tangent, roll=0




#q0 = quat_from_x_axis(tangent_vector)  # returns [qw,qx,qy,qz]

# initial state guess
q_wxyz_0 = quat_from_tangent_vec(tangent_vector)  # returns [qw,qx,qy,qz]
X0_guess = np.concatenate([gate_center, q_wxyz_0])

# create NLP variable
Xk = ca.MX.sym("X0", nx)
w  += [Xk]
w0 += list(X0_guess)
lbw+= [-ca.inf]*nx
ubw+= [ ca.inf]*nx

# ----- Constraint: X0 exactly on first gate
g   += [Xk[:3] - gate_center]   
lbg += [0]*3
ubg += [0]*3

# # Orientation constraint: x-axis aligned with tangent
# Forward direction (body x-axis in world frame)
qw, qx, qy, qz = Xk[3], Xk[4], Xk[5], Xk[6]
vel_vector = ca.vertcat(
    1 - 2*(qy**2 + qz**2),
    2*(qx*qy + qw*qz),
    2*(qx*qz - qw*qy)
)
g   += [ca.dot(vel_vector, tangent_vector)]   # dot = 1 → aligned
lbg += [1]
ubg += [1]







# ─────────── 2) Loop over arc‑length intervals  ──────────────────
J = 0


for k in range(N):
    # centre‑line point & tangent vector at stage k
    centerline_point, tangent_vector, acceleration_vector = generate_path_data_4_solver(k+1, N)    

    # ---- Δt_k --------------------------------------------------
    dt_k = ca.MX.sym(f"dt_{k}")
    w  += [dt_k]
    w0 += [ds_interval_actual]          # rough initial guess (5 m/s)
    lbw+= [0]                 # lower bound > 0
    ubw+= [1]                 # or ca.inf

    # ---- control Uk ----
    Uk = ca.MX.sym(f"U_{k}", nu)
    w  += [Uk]
    w0 += [0]*nu               # guess
    lbw+= [-ca.inf]*nu
    ubw+= [ca.inf]*nu

    if k == 0:
        U0 = Uk  # store the first control for periodicity constraint later


    # ----- successor state X_{k+1} ------------------------------
    # position guess = centerline point
    p0 = centerline_point
    # quaternion guess = align body-x with tangent vector
    #q0 = quat_from_x_axis(tangent_vector)
    q0 = quat_from_tangent_vec(tangent_vector)  # returns [qw,qx,qy,qz]


    Xk_next = ca.MX.sym(f"X_{k+1}", nx)
    w  += [Xk_next]
    w0 += [*p0,*q0] # [0]*nx   #1,0,0,0
    lbw+= [-ca.inf]*nx
    ubw+= [ ca.inf]*nx

    # ----- dynamics with variable step --------------------------
    Xk_rk = rk4(Xk, Uk, dt_k)
    g   += [Xk_rk - Xk_next]
    lbg += [0]*nx
    ubg += [0]*nx

    # # ----- constraint on quaternion norm (NEW)-------------------------------
    g   += [Xk[3]**2 + Xk[4]**2 + Xk[5]**2 + Xk[6]**2 - 1]
    lbg += [0]
    ubg += [0]




    # ---- path constraints (example: gate corridor) ---- 
    pos_err = Xk_next[0:3] - centerline_point

    # forward direction (body x-axis in world frame from quaternion)
    qw, qx, qy, qz = Xk_next[3], Xk_next[4], Xk_next[5], Xk_next[6]
    vel_vector = ca.vertcat(
        1 - 2*(qy**2 + qz**2),
        2*(qx*qy + qw*qz),
        2*(qx*qz - qw*qy)
                        )


    # Constraint 1: squared distance to centerline (0 at gates)
    if k+1 in gates_indexes_4_problem:
        g   += [ca.dot(pos_err, pos_err)]
        lbg += [0]                    # minimum distance = 0 (can't be inside-out)
        ubg += [0]                     # maximum distance allowed

        # # must hit gates straight on  (since it's normalized we can remove the constraint on unitary velocity)
        # g   += [ca.dot(tangent_vector, vel_vector)]
        # lbg += [1]                    # minimum distance = 0 (can't be inside-out)
        # ubg += [1]                     # maximum distance allowed

        # add as soft constraint instead
        J += 1000 * ca.sumsqr(tangent_vector - vel_vector)

    else:
        g   += [ca.dot(pos_err, pos_err)]
        lbg += [-0.1]                   # it's positive anyway
        ubg += [lane_radius**2]       # maximum distance allowed

        # Constraint 2: enforce position lies in the normal plane (⊥ to tangent)
        g   += [ca.dot(tangent_vector, pos_err)]
        lbg += [0]
        ubg += [0]

        # # maximum curvature 
        # g   += [Uk[0], Uk[1]]  # curvature must be less than k_max
        # lbg += [-k_max,-k_max]  # no lower bound
        # ubg += [k_max, k_max ] # curvature can be anything up to k_max

    # # enforce quaternion continuity (avoid sign flip)
    # g   += [Xk[3]*Xk_next[3] + Xk[4]*Xk_next[4] + Xk[5]*Xk_next[5] + Xk[6]*Xk_next[6]]
    # lbg += [0]    # enforce >= 0 (CasADi only allows equalities/inequalities, so this is ≥0)
    # ubg += [ca.inf]



    # ----- define cost -------------------------------
    #total_time_expr += dt_k
    # ---- actuation cost ----
    #J += ca.dot(pos_err, pos_err)

    #J += q_norm_penalty * (qw**2 + qx**2 + qy**2 + qz**2 - 1)**2 # this should keep the quaternion normalized


    actuation = ca.dot(Uk, Uk)
    J += q_actuation * actuation**2  # cost of actuation
    


    if k > 0:
        J += q_smoothness * ca.sumsqr(Uk - Uk_prev) / ds_interval_actual

        # add cost on path length
        J += q_length_path * ca.norm_2(Xk_next[0:3] - Xk[0:3])


    # next loop
    Uk_prev = Uk
    Xk = Xk_next


# # ─────────── 3) Periodicity: finish == start --------------------
g   += [Xk - w[0]]
lbg += [0]*nx
ubg += [0]*nx

# also on the inputs
g   += [Uk - U0]
lbg += [0]*nu
ubg += [0]*nu



# also last input - first input add cost on the jump
# J += q_smoothness * ca.sumsqr(Uk - U0) / ds_interval_actual  # last input - first input








# ─────────────────  Build & solve the NLP  ───────────────────────
ipopt_opts = {
    "ipopt": {
        "max_iter": 100,
        "tol": 1e-4,
        "acceptable_tol": 1e-4,
        "acceptable_iter": 10,
        "max_cpu_time": 1000,
        "mu_init": 1e-3,
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based",
        "print_level": 5,
        "bound_relax_factor": 1e-3,
        "linear_solver": "mumps",
        "constr_viol_tol": 1e-3,
        "dual_inf_tol": 1e-3,
    }
}

nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)
sol    = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)











# ─────────── 5)  Extract optimal trajectories  ------------------
w_opt = sol["x"].full().squeeze()

offset = 0
state_traj = np.zeros((N+1, nx))
input_traj = np.zeros((N,   nu))
dt_traj    = np.zeros(N)

state_traj[0] = w_opt[offset:offset+nx]; offset += nx
for k in range(N):
    dt_traj[k]    = w_opt[offset];        offset += 1
    input_traj[k] = w_opt[offset:offset+nu];   offset += nu
    state_traj[k+1] = w_opt[offset:offset+nx]; offset += nx

time_vec = np.concatenate(([0], np.cumsum(dt_traj)))
#print(f"Optimal lap time: {time_vec[-1]:.3f} s")




# evaluate arc length by summing up the distances between the points
s_vals_optimal_path = np.zeros(N + 1)
for k in range(N):
    dx = state_traj[k+1, 0] - state_traj[k, 0]
    dy = state_traj[k+1, 1] - state_traj[k, 1]
    dz = state_traj[k+1, 2] - state_traj[k, 2]
    s_vals_optimal_path[k+1] = s_vals_optimal_path[k] + np.sqrt(dx**2 + dy**2 + dz**2)






# save the optimized trajectory to a .npy file
# build output path
output_dir = os.path.join(pkg_path, "src", "solvers_setup", "offline_optimal_solutions")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{track_choice}_optimally_smoothed_quaternions.npy")  # ← .npy

# assemble the array exactly as before
input_traj_2_save           = np.vstack((input_traj, input_traj[-1, :]))  # pad last input
optimal_traj   = np.hstack((state_traj, input_traj_2_save))
# add a column with the path progress values and timestep (equally spaced by construction)

#time_vec                    = np.linspace(0, T_opt, N + 1)
optimal_traj   = np.column_stack((optimal_traj, s_vals_optimal_path))


# save a single array to .npy
np.save(output_file, optimal_traj)

print()
print('---')
print("Optimal trajectory saved to:", output_file)










# # # build output path
# # output_dir = os.path.join(pkg_path, "src", "solvers_setup", "offline_optimal_solutions")
# # os.makedirs(output_dir, exist_ok=True)
# # output_file = os.path.join(output_dir, f"{track_choice}.npy")  # ← .npy

# # # assemble the array exactly as before
# # input_traj_2_save           = np.vstack((input_traj, input_traj[-1, :]))  # pad last input
# # optimal_action_state_traj   = np.hstack((input_traj_2_save, state_traj))
# # # add a column with the path progress values and timestep (equally spaced by construction)
# # s_vec = np.linspace(0, s_vals_global_path[-1], N + 1)
# # #time_vec                    = np.linspace(0, T_opt, N + 1)
# # optimal_action_state_traj   = np.column_stack((optimal_action_state_traj, s_vec, time_vec))


# # #
# # # save a single array to .npy
# # np.save(output_file, optimal_action_state_traj)




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ───────────── Prepare data ─────────────
x, y, z = state_traj[:, 0], state_traj[:, 1], state_traj[:, 2]

# Control (accelerations): assumed shape (N, 3)
wz,wy = input_traj[:, 0], input_traj[:, 1]



#curvature = np.linalg.norm(acc, axis=1) + 1e-8  # Avoid division by zero

# # Compute curvature radius = 1 / ||acc||
# curvature_radius = 1.0 / curvature

# # Pad last element to match state length
# curvature = np.append(curvature, curvature[-1])
# curvature_radius = np.append(curvature_radius, curvature_radius[-1])

# ───────────── Build line segments for color line ─────────────
points = np.column_stack((x, y, z))
segments = np.stack([points[:-1], points[1:]], axis=1)

lc = Line3DCollection(
    segments,
    cmap='plasma',
    norm=plt.Normalize(vmin=np.min(wz), vmax=np.max(wz))
)
lc.set_array(wz[:-1])  # color per segment
lc.set_linewidth(5.0)

# ───────────── Start Plot ─────────────
fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# add colored path
ax_3d.add_collection(lc)

# 1. Original path in gray
ax_3d.plot(x_4_problem, y_4_problem, z_4_problem, color='gray', lw=1.5, label='Original Path')

# 2. New (optimized) path in black
#ax_3d.plot(x, y, z, color='black', lw=2.5, label='Optimized Path')

# # 3. Tangent vectors (true magnitude)
# skip = max(1, len(x) // 50)
# ax_3d.quiver(
#     x[::skip], y[::skip], z[::skip],
#     vx[::skip], vy[::skip], vz[::skip],
#     normalize=False, color='darkgoldenrod',
#     arrow_length_ratio=0.1, linewidth=1,
#     label='Tangent vectors'
# )

# # 4. Second-order derivatives (true magnitude, scaled using curvature trick)
# scale_factor = 1 / curvature[:-1]
# scaled_ax = acc_x * scale_factor**2
# scaled_ay = acc_y * scale_factor**2
# scaled_az = acc_z * scale_factor**2

# # ax_3d.quiver(
# #     x[:-1:skip], y[:-1:skip], z[:-1:skip],
# #     scaled_ax[::skip], scaled_ay[::skip], scaled_az[::skip],
# #     normalize=False, color='tan',
# #     arrow_length_ratio=0.1, linewidth=0.8,
# #     label='Second-order derivatives'
# # )

# ax_3d.quiver(
#     x[:-1:skip], y[:-1:skip], z[:-1:skip],
#     acc_x[::skip], acc_y[::skip], acc_z[::skip],
#     normalize=False, color='orangered',
#     arrow_length_ratio=0.1, linewidth=1,
#     label='Second-order derivatives'
# )



# add initial point marked with a red x
ax_3d.scatter(x[0], y[0], z[0], color='darkgreen', s=80, label='Start Point', edgecolor='black')



# ───────────── Axes setup ─────────────
cbar = fig_3d.colorbar(lc, ax=ax_3d, pad=0.1, shrink=0.7)
cbar.set_label('Curvature [1/m]')

ax_3d.set_title("3D Path Colored by Curvature")
ax_3d.set_xlabel("X [m]")
ax_3d.set_ylabel("Y [m]")
ax_3d.set_zlabel("Z [m]")
ax_3d.grid(True)

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    plot_rad = 0.5 * max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))
    mid = np.array([np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)])
    ax.set_xlim3d(mid[0] - plot_rad, mid[0] + plot_rad)
    ax.set_ylim3d(mid[1] - plot_rad, mid[1] + plot_rad)
    ax.set_zlim3d(mid[2] - plot_rad, mid[2] + plot_rad)

set_axes_equal(ax_3d)
ax_3d.legend()
plt.tight_layout()


import numpy as np

def add_gate_circles(ax, gates, lane_radius=0.5, num_circle_points=30, color='gray'):
    for gate in gates:
        # Unpack gate
        px, py, pz, tx, ty, tz = gate
        position = np.array([px, py, pz])
        tangent = np.array([tx, ty, tz])
        tangent /= np.linalg.norm(tangent) + 1e-8  # Normalize

        # Create an orthonormal basis: pick arbitrary normal to tangent
        if abs(tangent[0]) < 1e-2 and abs(tangent[1]) < 1e-2:
            # Tangent is nearly z-axis → use x-axis
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 0, 1])

        normal1 = np.cross(tangent, arbitrary)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.cross(tangent, normal1)

        # Create circle points in plane orthogonal to tangent
        theta = np.linspace(0, 2 * np.pi, num_circle_points)
        circle_pts = (np.outer(np.cos(theta), normal1) +
                      np.outer(np.sin(theta), normal2)) * lane_radius

        # Offset circle to gate position
        circle_pts += position

        # Plot circle
        ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2],
                color=color, linewidth=2.0, alpha=1)

# Call this *after* creating your 3D plot (with ax_3d)
add_gate_circles(ax_3d, gates, lane_radius=lane_radius, color='lightsteelblue')











# Create figure
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig2.suptitle("Angular Velocities", fontsize=15)

# Plot yaw rate
axes2[0].plot(s_4_problem, input_traj_2_save[:,0], color='darkgreen', label='Yaw rate (rad/s)')
axes2[0].set_xlim([s_4_problem[0], s_4_problem[-1]])
axes2[0].set_ylabel("Yaw rate [rad/s]")
axes2[0].set_title("Yaw rate over path")
axes2[0].grid(True)
axes2[0].legend()

# Plot pitch rate
axes2[1].plot(s_4_problem, input_traj_2_save[:,1], color='orangered', label='Pitch rate (rad/s)')
axes2[1].set_ylabel("Pitch rate [rad/s]")
axes2[1].set_xlabel("s [m]")
axes2[1].set_title("Pitch rate over path")
axes2[1].grid(True)
axes2[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.show()









import matplotlib.pyplot as plt

# Time array (assumed equally spaced)
t = s_4_problem

# Extract data
x, y, z     = state_traj[:, 0], state_traj[:, 1], state_traj[:, 2]
# vx, vy, vz  = state_traj[:, 3], state_traj[:, 4], state_traj[:, 5]
# ax, ay, az  = input_traj[:, 0], input_traj[:, 1], input_traj[:, 2]

# Setup figure and axes
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
fig.suptitle("Trajectory States and Inputs", fontsize=16)

# Labels for rows and columns
components = ['X', 'Y', 'Z']
positions = [x, y, z]
# velocities = [vx, vy, vz]
# accelerations = [ax, ay, az]
# copy last acceleration to match time vector length
accelerations = [np.append(a, a[-1]) for a in accelerations]  # pad last element

# Plot loop
for i in range(3):
    # Position
    axes[0, i].plot(t, positions[i], color='dodgerblue')
    axes[0, i].set_title(f"{components[i]} Position")
    axes[0, i].grid(True)

    # Velocity
    axes[1, i].plot(t, velocities[i], color='darkgreen')
    axes[1, i].set_title(f"{components[i]} Velocity")
    axes[1, i].grid(True)

    # Acceleration
    axes[2, i].plot(t, accelerations[i], color='orangered')
    axes[2, i].set_title(f"{components[i]} Acceleration")
    axes[2, i].grid(True)
    axes[2, i].set_xlabel("Time (normalized)")

# Y-axis labels
axes[0, 0].set_ylabel("Position [m]")
axes[1, 0].set_ylabel("Velocity [m/s]")
axes[2, 0].set_ylabel("Acceleration [m/s²]")

plt.tight_layout(rect=[0, 0, 1, 0.96])







# Compute velocity norm (speed)
#speed = np.linalg.norm(state_traj[:, 3:6], axis=1)

# Compute curvature (norm of acceleration vector)
#curvature_radius = np.linalg.norm(input_traj, axis=1)  # a = d²x/ds²


# Create figure
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig2.suptitle("Velocity Norm and Curvature", fontsize=15)

# Velocity norm plot
axes2[0].plot(t, speed, color='darkgreen', label='|v|')
axes2[0].set_xlim([t[0], t[-1]])
axes2[0].set_ylim([0.9, 1.1])
axes2[0].set_ylabel("Speed [m/s]")
axes2[0].set_title("Velocity Norm")
axes2[0].grid(True)
axes2[0].legend()

# Curvature plot
axes2[1].plot(t, curvature_radius, color='orangered', label='Curvature ‖a‖')
axes2[1].set_ylabel("Curvature radius [m]")
axes2[1].set_xlabel("s")
axes2[1].set_title("Curvature radius ")
axes2[1].grid(True)
axes2[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])




plt.show()




















