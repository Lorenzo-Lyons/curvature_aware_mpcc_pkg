import numpy as np
import casadi as ca
import os, sys, roslib
from drone_dynamic_model import drone
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
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2,k_vec, \
gates_coordinates, gates_s, time_optimal_trajectory = generate_path_data(track_choice, load_optimally_smoothed_path)



# evaluate the trosion of the curve

def compute_torsion(s, dx_ds, dy_ds, dz_ds,
                    d2x_ds2, d2y_ds2, d2z_ds2, k_vec):
    """
    Compute torsion τ(s) from arc-length parametrized curve data.

    Parameters
    ----------
    s : array_like
        Arc length samples (1D array).
    dx_ds, dy_ds, dz_ds : array_like
        First derivatives wrt s (tangent vector).
    d2x_ds2, d2y_ds2, d2z_ds2 : array_like
        Second derivatives wrt s.
    k_vec : array_like
        Curvature magnitudes (norm of 2nd derivative).

    Returns
    -------
    tau : ndarray
        Torsion values at each s (NaN at boundaries if central diff used).
    """

    # Tangent and curvature vector
    T = np.vstack((dx_ds, dy_ds, dz_ds)).T  # shape (N,3)
    d2r = np.vstack((d2x_ds2, d2y_ds2, d2z_ds2)).T
    k = np.asarray(k_vec)

    # Normal vector N = d2r/k
    N = np.zeros_like(d2r)
    mask = k > 1e-12
    N[mask] = d2r[mask] / k[mask, None]

    # Binormal B = T × N
    B = np.cross(T, N)
    B /= np.linalg.norm(B, axis=1)[:, None]

    # Differentiate N numerically wrt s
    dN_ds = np.gradient(N, s, axis=0)

    # τ = -B ⋅ dN/ds
    tau = -np.einsum('ij,ij->i', B, dN_ds)

    return tau

# evalaute it
torsion = compute_torsion(s_4_local_path, dx_ds, dy_ds, dz_ds,
                            d2x_ds2, d2y_ds2, d2z_ds2, k_vec)




def plot_curvature_torsion(s, k_vec, tau):
    """
    Plot curvature κ(s) and torsion τ(s) along arc length.
    """
    plt.figure(figsize=(8,5))
    plt.plot(s, k_vec, label="Curvature κ(s)", color="blue")
    plt.plot(s, tau, label="Torsion τ(s)", color="red")
    plt.xlabel("Arc length s")
    plt.ylabel("Value")
    plt.title("Curvature and Torsion vs Arc Length")
    plt.legend()
    plt.grid(True)


# Plot torsion-curvature results
plot_curvature_torsion(s_4_local_path, k_vec, torsion)


# plot the data
import numpy as np
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ───────────── Prepare data ─────────────
x, y, z = x_4_local_path, y_4_local_path, z_4_local_path
vx, vy, vz = dx_ds, dy_ds , dz_ds
speed = np.linalg.norm(np.array([dx_ds, dy_ds, dz_ds]), axis=1)

# Control (accelerations): assumed shape (N, 3)
acc_x, acc_y, acc_z = d2x_ds2,d2y_ds2,d2z_ds2
acc = np.vstack((acc_x, acc_y, acc_z)).T
curvature = k_vec #np.linalg.norm(acc, axis=1) + 1e-8  # Avoid division by zero

# Compute curvature radius = 1 / ||acc||
curvature_radius = 1.0 / curvature

# Pad last element to match state length
curvature = np.append(curvature, curvature[-1])
curvature_radius = np.append(curvature_radius, curvature_radius[-1])

# ───────────── Build line segments for color line ─────────────
points = np.column_stack((x, y, z))
segments = np.stack([points[:-1], points[1:]], axis=1)

lc = Line3DCollection(
    segments,
    cmap='plasma',
    norm=plt.Normalize(vmin=np.min(curvature), vmax=np.max(curvature))
)
lc.set_array(curvature[:-1])  # color per segment
lc.set_linewidth(5.0)

# ───────────── Start Plot ─────────────
fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# add colored path
ax_3d.add_collection(lc)


# 3. Tangent vectors (true magnitude)
skip = max(1, len(x) // 50)
ax_3d.quiver(
    x[::skip], y[::skip], z[::skip],
    vx[::skip], vy[::skip], vz[::skip],
    normalize=False, color='darkgoldenrod',
    arrow_length_ratio=0.1, linewidth=1,
    label='Tangent vectors'
)

# 4. Second-order derivatives (true magnitude, scaled using curvature trick)
scale_factor = 1 / curvature[:-1]
scaled_ax = acc_x * scale_factor**2
scaled_ay = acc_y * scale_factor**2
scaled_az = acc_z * scale_factor**2

# # SCALED VERSION
# ax_3d.quiver(
#     x[:-1:skip], y[:-1:skip], z[:-1:skip],
#     scaled_ax[::skip], scaled_ay[::skip], scaled_az[::skip],
#     normalize=False, color='orangered',
#     arrow_length_ratio=0.1, linewidth=0.8,
#     label='center of curvature direction (scaled)'
# )

# NON SCALED VERSION
ax_3d.quiver(
    x[:-1:skip], y[:-1:skip], z[:-1:skip],
    acc_x[::skip], acc_y[::skip], acc_z[::skip],
    normalize=False, color='orangered',
    arrow_length_ratio=0.1, linewidth=1,
    label='Second-order derivatives non scaled'
)



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





plt.show()
