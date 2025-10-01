import numpy as np
import os, sys, roslib
# now visualise the state and input trajectories
import matplotlib.pyplot as plt

# ─────────────── ROS helper to locate extra code ──────────────
pkg_path = roslib.packages.get_pkg_dir('racing_campcc_pkg')
sys.path.append(os.path.join(pkg_path, 'src'))
from reference_path_handeling_functions import generate_path_data  # noqa

# select track 
track_choice = 'vicon_racetrack'
#track_choice = 'spline_circle'


# load optimally generated path data
load_optimally_smoothed_path = True

s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, \
s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path, \
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, \
gates_coordinates, gates_s, time_optimal_trajectory = generate_path_data(track_choice, load_optimally_smoothed_path)




































# plot the 3D trajectory with speed encoding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# # 1. positions and speed ------------------------------------------------------
# x, y, z  = state_traj[:, 0], state_traj[:, 1], state_traj[:, 2]
# speed    = np.linalg.norm(state_traj[:, 3:6], axis=1)      # no normalisation

# # 2. coloured 3‑D line --------------------------------------------------------
# points   = np.column_stack((x, y, z))
# segments = np.stack([points[:-1], points[1:]], axis=1)     # (N, 2, 3)

# lc = Line3DCollection(
#         segments,
#         cmap='plasma',                                     # ← requested cmap
#         norm=plt.Normalize(vmin=speed.min(), vmax=speed.max()))
# lc.set_array(speed[:-1])                                   # colour per segment
# lc.set_linewidth(5.0)

# 3. plot ---------------------------------------------------------------------
fig_3d = plt.figure(figsize=(10, 6))
ax_3d  = fig_3d.add_subplot(111, projection='3d')

#ax_3d.add_collection(lc)
ax_3d.plot(x_vals_global_path, y_vals_global_path, z_vals_global_path,
           label='Reference Path', color='gray', lw=1.0)

#cbar = fig_3d.colorbar(lc, ax=ax_3d, pad=0.1, shrink=0.7)
#cbar.set_label('Speed [m/s]')

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
