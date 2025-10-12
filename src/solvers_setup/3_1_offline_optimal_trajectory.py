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

# select track 
#track_choice = 'vicon_racetrack'
#track_choice = 'spline_circle'
track_choice = 'analytic_circle'


# load optimally generated path data
load_optimally_smoothed_path = False

s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
roll_global_path, pitch_global_path, yaw_global_path, \
s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec,\
roll_4_local_path, pitch_4_local_path, yaw_4_local_path, \
wz_4_local_path, wx_4_local_path, \
gates, gates_s_global_path, time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, load_optimally_smoothed_path)



# Weights for the control action
R = 1e-4 * np.diag([10, 10, 10, 1])  # ⬅️ Tune these weights as needed
# these are relative to the lap time. So they are adding a w*u * dt at each stage, they are not really the same as the weight sin the MPC,
# beacuse they are weighed against other things




# ─────────────── GLOBAL SETTINGS ──────────────
# the fianl time is free, I.e. it's the quantity to be optimised

ds_interval_target = 0.1 # ⬤ EDIT ME: discretization step (m)
N  = int(np.ceil(s_vals_global_path[-1]/ds_interval_target))           # ⬤ EDIT ME: control intervals
ds_interval_actual = s_vals_global_path[-1] / N  # actual discretization step (m)

nx = 9              # 3 pos + 3 vel + 3 Euler
nu = 4              # roll_d, pitch_d, yaw_d, thrust
h_ref = 1.0 / N                   # τ‑domain step (unit time)

# ─────────────── SYMBOLS ──────────────
x  = ca.MX.sym("x", nx)
u  = ca.MX.sym("u", nu)
T  = ca.MX.sym("T")               # will be appended to w later

# ─────────────── DYNAMICS ──────────────
def unpack_x(x):
    # px, py, pz, vx, vy, vz, roll, pitch, yaw 
    return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]

def unpack_u(u):
    # roll_d, pitch_d, yaw_d, thrust
    return u[0], u[1], u[2], u[3]

def drone_rhs(x, u):
    px, py, pz, vx, vy, vz, roll, pitch, yaw = unpack_x(x)
    roll_d, pitch_d, yaw_d, thrust = unpack_u(u)
    return ca.vertcat(*drone(roll_d, pitch_d, yaw_d, thrust, vx, vy, vz, roll, pitch, yaw))

def rk4(xk, uk, h_step):
    k1 = drone_rhs(xk,              uk)
    k2 = drone_rhs(xk + 0.5*h_step*k1, uk)
    k3 = drone_rhs(xk + 0.5*h_step*k2, uk)
    k4 = drone_rhs(xk +       h_step*k3, uk)
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

centreline = np.vstack((x_4_problem, y_4_problem, z_4_problem,dx_ds_4_problem,dy_ds_4_problem,dz_ds_4_problem)).T   # shape (N+1, 3)

# helper used inside the NLP construction
def generate_path_data_4_problem(k: int, N: int):
    centerline_point = centreline[k,:3]  # get the point at stage k
    tangent_vector = centreline[k,3:6]  # get the tangent vector at stage k
    return centerline_point ,tangent_vector






# define constraint values
lane_radius = 0.5
input_max = np.array([np.pi / 3, np.pi / 3, np.pi, 3.6])  # roll, pitch, yaw, thrust
input_min = -input_max

v_guess = 5.0  # m/s, initial guess for velocity







w, w0, lbw, ubw = [], [], [], []
g, lbg, ubg     = [], [], []

# ─────────── 1) Initial state  X0  (free, will match XN) ─────────
Xk = ca.MX.sym("X0", nx)
w  += [Xk]
w0 += [0]*nx
lbw+= [-ca.inf]*nx
ubw+= [ ca.inf]*nx





# ─────────── 2) Loop over arc‑length intervals  ──────────────────
J = 0


for k in range(N):
    # centre‑line point & tangent vector at stage k
    centerline_point, tangent_vector = generate_path_data_4_problem(k, N)    

    # ---- Δt_k --------------------------------------------------
    dt_k = ca.MX.sym(f"dt_{k}")
    w  += [dt_k]
    w0 += [ds_interval_actual / v_guess]          # rough initial guess (5 m/s)
    lbw+= [1e-3]                 # lower bound > 0
    ubw+= [1]                 # or ca.inf

    # ---- control Uk ----
    Uk = ca.MX.sym(f"U_{k}", nu)
    w  += [Uk]
    w0 += [0, 0, 0, 0]               # guess
    lbw+= input_min.tolist()
    ubw+= input_max.tolist()

    # ----- successor state X_{k+1} ------------------------------
    Xk_next = ca.MX.sym(f"X_{k+1}", nx)
    w  += [Xk_next]
    w0 += [0]*nx
    lbw+= [-ca.inf]*nx
    ubw+= [ ca.inf]*nx

    # ----- dynamics with variable step --------------------------
    Xk_rk = rk4(Xk, Uk, dt_k)
    g   += [Xk_rk - Xk_next]
    lbg += [0]*nx
    ubg += [0]*nx


    # ---- path constraints (example: gate corridor) ---- 
    pos_err = Xk_next[0:3] - centerline_point

    # Constraint 1: squared distance to centerline point must be ≤ lane_radius²
    g   += [ca.dot(pos_err, pos_err)]
    lbg += [0]                    # minimum distance = 0 (can't be inside-out)
    ubg += [lane_radius**2]       # maximum distance allowed

    # Constraint 2: enforce position lies in the normal plane (⊥ to tangent)
    g   += [ca.dot(tangent_vector, pos_err)]
    lbg += [0]
    ubg += [0]

    # ----- accumulate total time -------------------------------
    J += dt_k
    # ---- actuation cost ----
    R_ca = ca.DM(R)
    J += ca.mtimes([Uk.T, R_ca, Uk]) * dt_k  # stage cost: uᵀRu * dt

    # next loop
    Xk = Xk_next


# ─────────── 3) Periodicity: finish == start --------------------
g   += [Xk - w[0]]
lbg += [0]*nx
ubg += [0]*nx






# ─────────────────  Build & solve the NLP  ───────────────────────
ipopt_opts = {
    "ipopt": {
        "max_iter": 4000,
        "tol": 1e-5,
        "acceptable_tol": 1e-5,
        "acceptable_iter": 10,
        "max_cpu_time": 100,
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
print(f"Optimal lap time: {time_vec[-1]:.3f} s")


















# build output path
output_dir = os.path.join(pkg_path, "src", "solvers_setup", "offline_optimal_solutions")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{track_choice}.npy")  # ← .npy

# assemble the array exactly as before
input_traj_2_save           = np.vstack((input_traj, input_traj[-1, :]))  # pad last input
optimal_action_state_traj   = np.hstack((input_traj_2_save, state_traj))
# add a column with the path progress values and timestep (equally spaced by construction)
s_vec = np.linspace(0, s_vals_global_path[-1], N + 1)
#time_vec                    = np.linspace(0, T_opt, N + 1)
optimal_action_state_traj   = np.column_stack((optimal_action_state_traj, s_vec, time_vec))


#
# save a single array to .npy
np.save(output_file, optimal_action_state_traj)











# Example labels for your states and inputs
state_labels = ["px", "py", "pz", "vx", "vy", "vz", "roll", "pitch", "yaw"]
input_labels = ["roll_desired", "pitch_desired", "yaw_desired", "thrust"]

# Create time vectors
N = state_traj.shape[0] - 1
time_states = np.arange(N + 1)       # states at nodes
time_inputs = np.arange(N)            # inputs on intervals

# --- Plot states ---
fig_states, axes_states = plt.subplots(3, 3, figsize=(15, 10))
fig_states.suptitle("States over Time", fontsize=16)

for i, ax in enumerate(axes_states.flat):
    ax.plot(time_states, state_traj[:, i])
    ax.set_title(state_labels[i])
    ax.set_xlabel("Time step")
    ax.grid(True)
    if state_labels[i] == "yaw":  # wrap yaw angle if needed
        ax.set_ylim([-np.pi, np.pi])

plt.tight_layout(rect=[0, 0, 1, 0.96])


# --- Plot inputs ---
fig_inputs, axes_inputs = plt.subplots(2, 2, figsize=(12, 8))
fig_inputs.suptitle("Inputs over Time", fontsize=16)

for i, ax in enumerate(axes_inputs.flat):
    ax.plot(time_inputs, input_traj[:, i], color='orangered')
    # add dashed line for constraints 
    ax.plot(time_inputs,  input_max[i] * np.ones(time_inputs.shape[0]), linestyle='--',color = 'gray', label='Max')
    ax.plot(time_inputs,  input_min[i] * np.ones(time_inputs.shape[0]), linestyle='--',color = 'gray', label='Min')
    ax.set_title(input_labels[i])
    ax.set_xlabel("Time step")
    ax.grid(True)
    

plt.tight_layout(rect=[0, 0, 1, 0.96])



# plot the 3D trajectory with speed encoding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 1. positions and speed ------------------------------------------------------
x, y, z  = state_traj[:, 0], state_traj[:, 1], state_traj[:, 2]
speed    = np.linalg.norm(state_traj[:, 3:6], axis=1)      # no normalisation

# 2. coloured 3‑D line --------------------------------------------------------
points   = np.column_stack((x, y, z))
segments = np.stack([points[:-1], points[1:]], axis=1)     # (N, 2, 3)

lc = Line3DCollection(
        segments,
        cmap='plasma',                                     # ← requested cmap
        norm=plt.Normalize(vmin=speed.min(), vmax=speed.max()))
lc.set_array(speed[:-1])                                   # colour per segment
lc.set_linewidth(5.0)

# 3. plot ---------------------------------------------------------------------
fig_3d = plt.figure(figsize=(10, 6))
ax_3d  = fig_3d.add_subplot(111, projection='3d')

ax_3d.add_collection(lc)
ax_3d.plot(x_4_problem, y_4_problem, z_4_problem,
           label='Reference Path', color='gray', lw=1.0)

cbar = fig_3d.colorbar(lc, ax=ax_3d, pad=0.1, shrink=0.7)
cbar.set_label('Speed [m/s]')

ax_3d.set_title("3D Position Trajectory (speed‑encoded)")
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
plt.tight_layout()
plt.show()





plt.show()






