import numpy as np
import casadi as ca
import os, sys, roslib
from drone_dynamic_model import drone
# now visualise the state and input trajectories
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from helper_functions_smoothing import add_gate_circles, set_axes_equal

# ─────────────── ROS helper to locate extra code ──────────────
pkg_path = roslib.packages.get_pkg_dir('racing_campcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))
from reference_path_handeling_functions import generate_path_data  # noqa

track_choice = 'vicon_racetrack'
#track_choice = 'spline_circle'


lane_radius = 0.5  # [m] half the distance between two gates


# s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
# s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
# dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec, \
# gates, gates_s_global_path, time_optimal_trajectory = generate_path_data(track_choice)

load_optimally_smoothed_path = True

s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
roll_global_path, pitch_global_path, yaw_global_path, \
s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec,\
roll_4_local_path, pitch_4_local_path, yaw_4_local_path, \
wz_4_local_path, wx_4_local_path, \
gates, gates_s_global_path, time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, load_optimally_smoothed_path)


# ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# pick the trajectory array to use
traj = time_optimal_trajectory_4_warmstart

# x-axis: use s (m) if it's in column 0; else use index
s = traj[:, -2] 

# roll, pitch, yaw from columns 6, 7, 8 (0-based)
roll  = traj[:, 6]
pitch = traj[:, 7]
yaw   = traj[:, 8]

# (optional) wrap angles to [-pi, pi] — uncomment if you want bounded plots
# roll  = (roll  + np.pi) % (2*np.pi) - np.pi
# pitch = (pitch + np.pi) % (2*np.pi) - np.pi
# yaw   = (yaw   + np.pi) % (2*np.pi) - np.pi

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Roll, Pitch, Yaw vs. Path Length", fontsize=16)

axes[0].plot(s, roll);   axes[0].set_ylabel("roll (rad)");  axes[0].grid(True)
axes[1].plot(s, pitch);  axes[1].set_ylabel("pitch (rad)"); axes[1].grid(True)
axes[2].plot(s, yaw);    axes[2].set_ylabel("yaw (rad)");   axes[2].set_xlabel("s (m)"); axes[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
