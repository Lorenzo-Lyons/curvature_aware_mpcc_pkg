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

load_optimally_smoothed_path = False

s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
roll_global_path, pitch_global_path, yaw_global_path, \
s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec,\
roll_4_local_path, pitch_4_local_path, yaw_4_local_path, \
wz_4_local_path, wx_4_local_path, \
gates, gates_s_global_path, time_optimal_trajectory_4_warmstart = generate_path_data(track_choice, load_optimally_smoothed_path)


# ──────────────────────────────────────────────────────────────
path_2_stored_optimally_smoothed_track = os.path.join(pkg_path, 'src', 'solvers_setup', 'offline_optimal_solutions', track_choice + '_optimally_smoothed_euler_angles.npy')

optimally_smoothed_track = np.load(path_2_stored_optimally_smoothed_track)

# unpack the optimally smoothed track
# state: [px, py, pz, roll, pitch, yaw]
nx = 6
# controls: [wz, wx]  (body z-rate, body y-rate)
nu = 2
# last column is the s values

x = optimally_smoothed_track[:, 0]
y = optimally_smoothed_track[:, 1]
z = optimally_smoothed_track[:, 2]
roll = optimally_smoothed_track[:, 3]
pitch = optimally_smoothed_track[:, 4]
yaw = optimally_smoothed_track[:, 5]
wz = optimally_smoothed_track[:, 6]
wx = optimally_smoothed_track[:, 7]
s = optimally_smoothed_track[:, 8]

dxds = optimally_smoothed_track[:, 9]
dyds = optimally_smoothed_track[:, 10]
dzds = optimally_smoothed_track[:, 11]
d2xds2 = optimally_smoothed_track[:, 12]
d2yds2 = optimally_smoothed_track[:, 13]
d2zds2 = optimally_smoothed_track[:, 14]

# ─────────────────────────────────────────────────────────────
acc = np.column_stack((d2xds2, d2yds2, d2zds2))
tangent = np.column_stack((dxds, dyds, dzds))



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

# ───────────── Start Plot 1 ─────────────
fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# add colored path
ax_3d.add_collection(lc)

add_gate_circles(ax_3d, gates, lane_radius=lane_radius, color='lightsteelblue')

# add initial point marked with a red x
ax_3d.scatter(x[0], y[0], z[0], color='darkgreen', s=80, label='Start Point', edgecolor='black')






def euler_to_rotation_matrix(roll, pitch, yaw):
    """Return rotation matrix from roll, pitch, yaw (XYZ convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation Z (yaw) * Y (pitch) * X (roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx

# ───────────── Plot Frames Along Trajectory ─────────────
step = 5

def plot_frames(ax, x, y, z, roll, pitch, yaw, acc, tangent ,step, scale=0.5):
    """Plot local coordinate frames along trajectory."""

    # cs_x = CubicSpline(s, x)
    # cs_y = CubicSpline(s, y)
    # cs_z = CubicSpline(s, z)

    # # evaluate 2nd derivatives along the arc length
    # ddx_ds2 = cs_x(s, 2)  # ∂²x/∂s²
    # ddy_ds2 = cs_y(s, 2)  # ∂²y/∂s²
    # ddz_ds2 = cs_z(s, 2)  # ∂²z/∂s²

    # # optionally, first derivatives too
    # dx_ds = cs_x(s, 1)
    # dy_ds = cs_y(s, 1)
    # dz_ds = cs_z(s, 1)

    # # pack into array if needed
    # acc = np.column_stack((ddx_ds2, ddy_ds2, ddz_ds2))
    # first_derivs  = np.column_stack((dx_ds, dy_ds, dz_ds))


    indices = list(range(0, len(x), step))
    if indices[-1] != len(x) - 1:
        indices.append(len(x) - 1)

    for i in indices:
        R = euler_to_rotation_matrix(roll[i], pitch[i], yaw[i])
        origin = np.array([x[i], y[i], z[i]])

        # Axes directions (scaled)
        #x_axis = R[:, 0] * scale get this from stored data to make sure it's ok
        x_axis = tangent[i,:] * scale
        y_axis = R[:, 1] * scale
        z_axis = R[:, 2] * scale

        ax.quiver(*origin, *x_axis, color='r', length=scale, normalize=False)
        ax.quiver(*origin, *y_axis, color='g', length=scale, normalize=False)
        ax.quiver(*origin, *z_axis, color='b', length=scale, normalize=False)

        acc_i = acc[i,:]
        ax.quiver(*origin,*acc_i,
            normalize=False, color='tan',
            arrow_length_ratio=0.1, linewidth=0.8,
            label='Second-order derivatives'
)
    




# ------------------- second figure --------------------
# build cubic splines for each coordinate
from scipy.interpolate import CubicSpline





fig_3d = plt.figure(figsize=(12, 8))
ax_3d_2 = fig_3d.add_subplot(111, projection='3d')


# 1. Original path in gray
ax_3d_2.plot(x, y, z, color='gray', lw=1.5, label='Original Path')

# add gates
add_gate_circles(ax_3d_2, gates, lane_radius=lane_radius, color='lightsteelblue')

# add frames
plot_frames(ax_3d_2, x, y, z, roll, pitch, yaw, acc, tangent ,step, scale=0.5)

# add initial point marked with a red x
ax_3d_2.scatter(x[0], y[0], z[0], color='darkgreen', s=50, label='Start Point', edgecolor='black')



set_axes_equal(ax_3d_2)




import numpy as np
import matplotlib.pyplot as plt

# Build an arc-length parameter s (meters) along the path
dx = np.diff(x)
dy = np.diff(y)
dz = np.diff(z)
ds = np.sqrt(dx*dx + dy*dy + dz*dz)
s = np.zeros(len(x))
s[1:] = np.cumsum(ds)

# Optionally unwrap angles to avoid 2π jumps
roll_u  = np.unwrap(roll)
pitch_u = np.unwrap(pitch)
yaw_u   = np.unwrap(yaw)

# Compute curvature magnitude (example: from acceleration array 'acc')
k = np.linalg.norm(acc, axis=1)  # curvature magnitude

fig_rpy, axes_rpy = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig_rpy.suptitle("Roll, Pitch, Yaw, and Curvature vs. Path Length", fontsize=16)

# Roll
axes_rpy[0].plot(s, roll_u)
axes_rpy[0].set_ylabel("roll (rad)")
axes_rpy[0].grid(True)

# Pitch
axes_rpy[1].plot(s, pitch_u)
axes_rpy[1].set_ylabel("pitch (rad)")
axes_rpy[1].grid(True)

# Yaw
axes_rpy[2].plot(s, yaw_u)
axes_rpy[2].set_ylabel("yaw (rad)")
axes_rpy[2].grid(True)

# Curvature
axes_rpy[3].plot(s, 1/k, color="tab:red")
axes_rpy[3].set_ylabel("curvature k (1/m)")
axes_rpy[3].set_xlabel("s (m)")
axes_rpy[3].grid(True)
# add horizontal line for lane radius
axes_rpy[3].axhline(y=lane_radius, color='gray', linestyle='--', label='lane_radius')

plt.tight_layout(rect=[0, 0, 1, 0.95])















#  --------------------- plotting s-denomitaor map ---------------------
# evalaute curvature


import numpy as np
import matplotlib.pyplot as plt

def plot_denominator_grid(s, k, lane_radius, n_p=201, show_zero_contour=True):
    """
    Plot a heatmap of denominator(s,p) = 1 - k(s)*p over s (x) and p (y).
    
    Parameters
    ----------
    s : (N,) array
        Progress along path (uniformly spaced).
    k : (N,) array
        Curvature magnitude aligned with s.
    lane_radius : float
        Half-width for lateral offsets (p in [-lane_radius, +lane_radius]).
    n_p : int, optional
        Number of samples for p-axis.
    show_zero_contour : bool, optional
        If True, overlays the contour where denominator == 0.
    """
    # Lateral offsets
    p = np.linspace(-lane_radius, lane_radius, n_p)

    # Broadcast to grid
    P = p[None, :]             # shape (1, n_p)
    denom = 1.0 - k[:, None] * P  # shape (len(s), n_p)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pcm = ax.pcolormesh(s, p, denom.T, shading='auto')
    fig.colorbar(pcm, ax=ax, label='denominator = 1 - k·p')

    if show_zero_contour:
        Sgrid, Pgrid = np.meshgrid(s, p, indexing='xy')
        cs = ax.contour(Sgrid, Pgrid, denom.T, levels=[0.0], colors='k', linewidths=1.2)
        cs.collections[0].set_label('denominator = 0')
        ax.legend(loc='upper right')

    ax.set_xlabel('s (progress along path)')
    ax.set_ylabel('p (lateral offset)')
    ax.set_title('Grid of denominator(s,p) = 1 - k(s)·p')
    plt.tight_layout()


# Example usage:
plot_denominator_grid(s, k, lane_radius=lane_radius)


import numpy as np
import matplotlib.pyplot as plt

def plot_denominator_vs_s(s, k, lane_radius, n_p=201):
    """
    Plot denominator(s,p) = 1 - k(s)*p with denominator on y-axis.
    
    Parameters
    ----------
    s : (N,) array
        Progress along path.
    k : (N,) array
        Curvature magnitude aligned with s.
    lane_radius : float
        Half-width for lateral offsets.
    n_p : int
        Number of p samples.
    """
    # Lateral offsets
    p = np.linspace(-lane_radius, lane_radius, n_p)

    # Denominator grid (Ns, Np)
    denom = 1.0 - k[:, None] * p[None, :]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s[:, None], denom, color='tab:blue', alpha=0.2)  
    # Each column of denom plotted as a vertical line of points

    ax.set_xlabel("s (progress along path)")
    ax.set_ylabel("denominator d = 1 - k·p")
    ax.set_title("Denominator values across lane at each s")
    #set x-axis limits
    ax.set_xlim(s[0], s[-1])
    #set y-axis limits to 0 and maxvalue of denom
    den_max = np.max(denom)
    ax.set_ylim(0, 1.1*den_max)
    plt.tight_layout()


# Example:
plot_denominator_vs_s(s, k, lane_radius=lane_radius)











# ---- find the maximum value of a that still satisfies the condition of prefering to increase s over a local minimum ----
n_s = 2000
s = np.asarray(s)
k = np.asarray(k)

# Resample onto a uniform s-grid
s_uniform = np.linspace(s.min(), s.max(), n_s)
k_uniform = np.interp(s_uniform, s, k)

# Feasible band edges
d_min_s = 1.0 - k_uniform * lane_radius
d_max_s = 1.0 + k_uniform * lane_radius


# evaluate the gradient vector
dd_ds = np.gradient(d_min_s, s_uniform)          # slope dd/ds
a = 0 
a_increase = 0.001 # increase a to see how the slope changes
check = False
while check == False:
    # evalaute the condition on all points
    a_new = a + a_increase
    condition = 2*a_new*d_min_s*dd_ds <= 1.0
    if condition.all():
        check = False
        print('a = ', a_new, ' satisfies condition, continuing to increase a')
        a = a_new
    else:
        print('Found maximum a to satisfy condition: ', a)
        check = True

a = a * 0.9 # set a to be slightly less than the maximum value found
print('Using a = ', a, ' for the heatmap plot')



import numpy as np
import matplotlib.pyplot as plt



# def plot_J_heatmap(
#     s, k, lane_radius, a=1.0,
#     n_s=2000,        # horizontal resolution (resampled along s)
#     n_d=400,         # vertical resolution (d samples)
#     dpi=150,         # figure DPI
#     quiver_stride_s=20,   # subsample along s for arrows
#     quiver_stride_d=20,   # subsample along d for arrows
#     arrow_scale=0.01      # arrow length as fraction of s-range
# ):
#     """
#     Plot heatmap of J(s,d) = -s + a*d^2 over the feasible band
#     d ∈ [1 - k(s)R, 1 + k(s)R], with black lines for the band edges
#     and the negative-gradient field (steepest descent).
#     """

#     # Global d grid
#     d_vec = np.linspace(d_min_s.min(), d_max_s.max(), n_d)
#     S, D = np.meshgrid(s_uniform, d_vec, indexing="xy")

#     # Cost
#     J = -S + a * D**2

#     # Mask outside feasible band
#     mask = (D < d_min_s[None, :]) | (D > d_max_s[None, :])
#     J_masked = np.where(mask, np.nan, J)

#     # Plot heatmap
#     fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
#     pcm = ax.pcolormesh(s_uniform, d_vec, J_masked, shading="auto", cmap="viridis")
#     fig.colorbar(pcm, ax=ax, label=f"J(s,d) = -s + {a}·d²")

#     # Band edges
#     ax.plot(s_uniform, d_min_s, 'k-', linewidth=1.5, label="feasible band")
#     ax.plot(s_uniform, d_max_s, 'k-', linewidth=1.5)

#     # ---- Negative gradient field: -(∇J) = (1, -2*a*d) ----
#     # Build a coarser grid for arrows
#     s_idx = np.arange(0, s_uniform.size, max(1, quiver_stride_s))
#     d_idx = np.arange(0, d_vec.size,     max(1, quiver_stride_d))
#     Sq, Dq = np.meshgrid(s_uniform[s_idx], d_vec[d_idx], indexing='xy')

#     # Keep only feasible points for arrows
#     dmin_q = d_min_s[s_idx][None, :]
#     dmax_q = d_max_s[s_idx][None, :]
#     valid = (Dq >= dmin_q) & (Dq <= dmax_q)

#     # Direction field
#     U = np.ones_like(Sq)            # ds component
#     V = -2.0 * a * Dq               # dd component

#     # Normalize to unit vectors, then scale to a small visual length
#     mag = np.hypot(U, V)
#     U = np.where(mag > 0, U / mag, 0.0)
#     V = np.where(mag > 0, V / mag, 0.0)

#     # Scale arrows uniformly in data units (keeps direction intact)
#     s_range = s_uniform[-1] - s_uniform[0]
#     L = arrow_scale * s_range
#     U_plot = U * L
#     V_plot = V * L

#     # Plot only valid arrows
#     ax.quiver(
#         Sq[valid], Dq[valid], U_plot[valid], V_plot[valid],
#         angles='xy', scale_units='xy', scale=1.0,
#         width=0.002, headwidth=3, headlength=4, color='k', pivot='mid', zorder=3
#     )

#     # Axes
#     ax.set_xlabel("s (m)")
#     ax.set_ylabel("d = 1 - k·p")
#     ax.set_title("Cost heatmap J(s,d) with feasible band and -∇J field")
#     # set equal axis
#     ax.set_aspect('equal')
#     ax.set_xlim(s_uniform[0], s_uniform[-1])
#     ax.set_ylim(0, d_vec.max() * 1.1)
#     ax.legend(loc="upper right")
#     plt.tight_layout()


def plot_J_heatmap(
    s, k, lane_radius, a=1.0,
    n_s=2000,        # horizontal resolution (resampled along s)
    n_d=400,         # vertical resolution (d samples)
    dpi=150,         # figure DPI
    quiver_stride_s=1,   # subsample along s for boundary arrows
    arrow_scale=0.01      # arrow length as fraction of s-range
):
    """
    Plot heatmap of J(s,d) = -s + a*d^2 over the feasible band
    d ∈ [1 - k(s)R, 1 + k(s)R], with black band edges, and ONLY the
    negative-gradient vectors evaluated on the lower bound d_min(s).

    NOTE: Assumes s_uniform, d_min_s, d_max_s are precomputed as in your snippet.
    """
    # ---- Heatmap on (s_uniform, d) ----
    d_vec = np.linspace(d_min_s.min(), d_max_s.max(), n_d)
    S, D = np.meshgrid(s_uniform, d_vec, indexing="xy")

    J = -S + a * D**2
    mask = (D < d_min_s[None, :]) | (D > d_max_s[None, :])
    J_masked = np.where(mask, np.nan, J)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    pcm = ax.pcolormesh(s_uniform, d_vec, J_masked, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label=f"J(s,d) = -s + {a}·d²")

    # Band edges
    ax.plot(s_uniform, d_min_s, 'k-', lw=1.5, label="feasible band")
    ax.plot(s_uniform, d_max_s, 'k-', lw=1.5)

    # ---- -∇J on the LOWER BOUND only ----
    # Sample s along the boundary
    s_idx = np.arange(0, s_uniform.size, max(1, quiver_stride_s))
    S0 = s_uniform[s_idx]
    D0 = d_min_s[s_idx]

    # Slope of the lower bound d_min(s)
    dmin_slope = np.gradient(d_min_s, s_uniform)[s_idx]   # dd_min/ds at S0
    
    # going along the line dmin(s), the tangent vector is t = (1, dd_min/ds)
    # Raw negative gradient at boundary points: g = (1, -2*a*d)
    dJds = np.ones_like(S0)
    dJdd = -2.0 * a * D0

    # tangent vector of the lower bound
    Tx = 1.0
    Ty = dmin_slope
    # normalize 
    t_norm = np.sqrt(Tx**2 + Ty**2)
    Tx /= t_norm
    Ty /= t_norm

    # evalaute projection of the gradient along the tangent
    dot_t = dJds * Tx + dJdd * Ty

    # evalaute quiver plot as the projection along the tangent
    U = dot_t * Tx
    V = dot_t * Ty



    ax.quiver(
        S0, D0, U, V,
        angles='xy', scale_units='xy', scale=1.0,
        width=0.002, headwidth=3, headlength=4,
        color='k', pivot='tail', zorder=3
    )

    # Axes
    ax.set_xlabel("s (m)")
    ax.set_ylabel("d = 1 - k·p")
    ax.set_title("Cost heatmap with -∇J shown only on the lower bound d_min(s)")
    ax.set_xlim(s_uniform[0], s_uniform[-1])
    ax.set_ylim(0, d_vec.max() * 1.1)
    ax.set_aspect('equal')  # if you want equal data scaling
    ax.legend(loc="upper right")
    plt.tight_layout()

    # also plot a simple plot of the dot_t
    fig_dot, ax_dot = plt.subplots(figsize=(10, 4))
    ax_dot.plot(S0, dot_t, 'b-', lw=2)
    ax_dot.axhline(0, color='k', ls='--')
    ax_dot.set_xlabel('s (m)')
    ax_dot.set_ylabel('Projection of -∇J along tangent')
    ax_dot.set_title('Projection of -∇J along the tangent of d_min(s)')
    ax_dot.grid(True)



plot_J_heatmap(s, k, lane_radius=lane_radius, a=a)





plt.show()


