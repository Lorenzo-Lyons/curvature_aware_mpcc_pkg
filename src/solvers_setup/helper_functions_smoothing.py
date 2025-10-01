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
        

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    plot_rad = 0.5 * max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))
    mid = np.array([np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)])
    ax.set_xlim3d(mid[0] - plot_rad, mid[0] + plot_rad)
    ax.set_ylim3d(mid[1] - plot_rad, mid[1] + plot_rad)
    ax.set_zlim3d(mid[2] - plot_rad, mid[2] + plot_rad)

