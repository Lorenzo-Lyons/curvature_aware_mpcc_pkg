import numpy as np
from scipy.interpolate import CubicSpline





def generate_track(Checkpoints_x, Checkpoints_y, Checkpoints_s, Ds_back, Ds_forward, Checkpoints_k):
    # Checkpoints_Rx,Checkpoints_Ry,Checkpoints_Rz,
    # n_points = 2000
    # spline_sections = 1000

    # determine where to set the origin of s
    # find index of s = Ds_back
    #index_start_lap = np.argmin(np.abs(Checkpoints_s - 0))

    index_start_lap = np.argmin(np.abs(Checkpoints_s - Ds_back))

    # now determine how long is a lap by finding the closest point to the starting point
    distance_vector = np.sqrt((Checkpoints_x - Checkpoints_x[index_start_lap]) ** 2 + (Checkpoints_y - Checkpoints_y[index_start_lap]) ** 2)
    index_forward = 10 # jump forwards by a little bit otherwise you will find the starting point again
    idx_finish_lap = np.argmin(np.abs(distance_vector[index_start_lap+index_forward:] - distance_vector[index_start_lap])) + index_start_lap + index_forward

    # determine the length of the track
    track_length = Checkpoints_s[idx_finish_lap] - Checkpoints_s[index_start_lap]
    print('track_length =', track_length)

    # find inxes of s = Ds_forward + track_length
    idx_finish = np.argmin(np.abs(Checkpoints_s - (Ds_forward + track_length)))

    # now produce global path
    s_vals_global_path = Checkpoints_s[index_start_lap:idx_finish_lap] - Checkpoints_s[index_start_lap]
    x_vals_global_path = np.array(Checkpoints_x[index_start_lap:idx_finish_lap])
    y_vals_global_path = np.array(Checkpoints_y[index_start_lap:idx_finish_lap])
    # R_x_vals_global_path = np.array(Checkpoints_Rx[index_start_lap:idx_finish_lap])
    # R_y_vals_global_path = np.array(Checkpoints_Ry[index_start_lap:idx_finish_lap])
    # R_z_vals_global_path = np.array(Checkpoints_Rz[index_start_lap:idx_finish_lap])
    k_vals_global_path = np.array(Checkpoints_k[index_start_lap:idx_finish_lap])

    # now produce path for local path generation
    s_4_local_path = Checkpoints_s[:idx_finish] - Checkpoints_s[index_start_lap]
    x_4_local_path = np.array(Checkpoints_x[:idx_finish])
    y_4_local_path = np.array(Checkpoints_y[:idx_finish])
    # R_x_4_local_path = np.array(Checkpoints_Rx[:idx_finish])
    # R_y_4_local_path = np.array(Checkpoints_Ry[:idx_finish])
    # R_z_4_local_path = np.array(Checkpoints_Rz[:idx_finish])
    k_4_local_path = np.array(Checkpoints_k[:idx_finish])


    # # just use Checkpoints_x and Checkpoints_y as the global path in the future
    # x_vals_global_path = Checkpoints_x
    # y_vals_global_path = Checkpoints_y

    # # define looped path for local path generation (in order to not run out of path)
    # x_4_local_path = np.concatenate((Checkpoints_x[:-1],Checkpoints_x, Checkpoints_x[1:]))
    # y_4_local_path = np.concatenate((Checkpoints_y[:-1],Checkpoints_y, Checkpoints_y[1:]))
    # s_4_local_path = np.concatenate((s_vals_global_path[:-1] -s_vals_global_path[-1],s_vals_global_path, s_vals_global_path[1:] + s_vals_global_path[-1]))
    
    # # generate cubic splines to resample in the required interval and with the required number of brakes
    # s_for_spline_generation = np.linspace(s_4_local_path[0], s_4_local_path[-1], spline_sections)
    # x_for_spline_generation = np.interp(s_for_spline_generation, s_4_local_path, x_4_local_path)
    # y_for_spline_generation = np.interp(s_for_spline_generation, s_4_local_path, y_4_local_path)

    # cs_scipy_x = CubicSpline(s_for_spline_generation, x_for_spline_generation)
    # cs_scipy_y = CubicSpline(s_for_spline_generation, y_for_spline_generation)



    # s_4_local_path = np.linspace(Ds_back, s_vals_global_path[-1] + Ds_forward, n_points)
    # x_4_local_path = cs_scipy_x(s_4_local_path)
    # y_4_local_path = cs_scipy_y(s_4_local_path)
    
    # R_x_vals_global_path, R_y_vals_global_path, R_z_vals_global_path, R_x_4_local_path, R_y_4_local_path, R_z_4_local_path,\
    return s_vals_global_path, x_vals_global_path, y_vals_global_path, s_4_local_path, x_4_local_path, y_4_local_path,\
    k_vals_global_path, k_4_local_path




def raw_track(choice):
    n_checkpoints = 100
    x_shift_vicon_lab = -3
    y_shift_vicon_lab = -2.2 #-2.7

    if choice == 'savoiardo':

        R = 0.8  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.82
        theta_init2 = np.pi * -0.5
        theta_end2 = np.pi * 0.5
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
        theta_init4 = np.pi * 0.5
        theta_end4 = np.pi * 1.5
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)
        Checkpoints_x1 = np.linspace(- 1.5 * R, 1.5 * R, n_checkpoints)
        Checkpoints_y1 = np.zeros(n_checkpoints) - R
        Checkpoints_x2 = 1.5 * R + R * np.cos(theta_vec2)
        Checkpoints_y2 = R * np.sin(theta_vec2)
        Checkpoints_x3 = np.linspace(1.5 * R, -1.5*R, n_checkpoints)
        Checkpoints_y3 = R * np.ones(n_checkpoints)
        Checkpoints_x4 = -1.5* R + R * np.cos(theta_vec4)
        Checkpoints_y4 = R * np.sin(theta_vec4)

        Checkpoints_x = [*Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1], *Checkpoints_x4[0:n_checkpoints - 1], *Checkpoints_x1[0:n_checkpoints]]
        Checkpoints_y = [*Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints - 1], *Checkpoints_y4[0:n_checkpoints -1], *Checkpoints_y1[0:n_checkpoints]]

    elif choice == 'double_donut':

        R = 1  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48
        theta_init1 = np.pi * -0.5
        theta_end1 = np.pi * 0.0
        theta_vec1 = np.linspace(theta_init1, theta_end1, n_checkpoints)
        theta_init2 = np.pi * 1
        theta_end2 = np.pi * -1
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
        theta_init3 = np.pi * 0
        theta_end3 = np.pi * 1.5
        theta_vec3 = np.linspace(theta_init3, theta_end3, n_checkpoints)

        Checkpoints_x1 = - R + R * np.cos(theta_vec1)
        Checkpoints_y1 =  + R * np.sin(theta_vec1)
        Checkpoints_x2 = + R + R * np.cos(theta_vec2)
        Checkpoints_y2 =  + R * np.sin(theta_vec2)
        Checkpoints_x3 = - R + R * np.cos(theta_vec3)
        Checkpoints_y3 = + R * np.sin(theta_vec3)

        Checkpoints_x = [*Checkpoints_x1[0:n_checkpoints - 1], *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints]]
        Checkpoints_y = [*Checkpoints_y1[0:n_checkpoints - 1], *Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints]]

    elif choice == 'straight_line':
        Checkpoints_x = np.linspace(0, 100, n_checkpoints)
        Checkpoints_y = np.zeros(n_checkpoints)

    elif choice == 'savoiardo_saturate_steering':
        R = 0.3  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48
        theta_init2 = np.pi * -0.5
        theta_end2 = np.pi * 0.5
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
        theta_init4 = np.pi * 0.5
        theta_end4 = np.pi * 1.5
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)
        Checkpoints_x1 = np.linspace(- 4.5 * R, 4.5 * R, n_checkpoints)
        Checkpoints_y1 = np.zeros(n_checkpoints) - R
        Checkpoints_x2 = 4.5 * R + R * np.cos(theta_vec2)
        Checkpoints_y2 = R * np.sin(theta_vec2)
        Checkpoints_x3 = np.linspace(4.5 * R, -4.5*R, n_checkpoints)
        Checkpoints_y3 = R * np.ones(n_checkpoints)
        Checkpoints_x4 = -4.5* R + R * np.cos(theta_vec4)
        Checkpoints_y4 = R * np.sin(theta_vec4)

        Checkpoints_x = [*Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1], *Checkpoints_x4[0:n_checkpoints - 1], *Checkpoints_x1[0:n_checkpoints ]]
        Checkpoints_y = [*Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints - 1], *Checkpoints_y4[0:n_checkpoints], *Checkpoints_y1[0:n_checkpoints - 1]]

    elif choice == 'racetrack_saturate_steering':

        R = 0.8  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48
        theta_init2 = np.pi * -0.5
        theta_end2 = np.pi * 0.5
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)

        theta_init3 = np.pi * 1.5
        theta_end3 = np.pi * 0.5
        theta_vec3 = np.linspace(theta_init3, theta_end3, n_checkpoints)

        theta_init6 = np.pi * 0.5
        theta_end6 = np.pi * 1.0
        theta_vec6 = np.linspace(theta_init6, theta_end6, n_checkpoints)

        theta_init8 = np.pi * -1.0
        theta_end8 = np.pi * 0.0
        theta_vec8 = np.linspace(theta_init8, theta_end8, n_checkpoints)

        theta_init10 = np.pi * 1.0
        theta_end10 = np.pi * 0.0
        theta_vec10 = np.linspace(theta_init10, theta_end10, n_checkpoints)

        theta_init12 = np.pi * -1.0
        theta_end12 = np.pi * -0.5
        theta_vec12 = np.linspace(theta_init12, theta_end12, n_checkpoints)

        # DEFINED STARTING FROM  START POINT AND THEN SHIFT IT LATER IF NEEDED
        Checkpoints_x1 = np.linspace(0, 3*R, n_checkpoints)
        Checkpoints_y1 = np.zeros(n_checkpoints) - R

        Checkpoints_x2 = + 3*R + R * np.cos(theta_vec2)
        Checkpoints_y2 =  R * np.sin(theta_vec2)

        Checkpoints_x3 = 3*R + R * np.cos(theta_vec3)
        Checkpoints_y3 = 2*R +R * np.sin(theta_vec3)

        Checkpoints_x4 = + 3*R + R * np.cos(theta_vec2)
        Checkpoints_y4 = + 4*R + R * np.sin(theta_vec2)

        Checkpoints_x5 = np.linspace(3*R, -3*R, n_checkpoints)
        Checkpoints_y5 = np.zeros(n_checkpoints) + 5*R

        Checkpoints_x6 = - 3*R + 2*R * np.cos(theta_vec6)
        Checkpoints_y6 = + 3*R + 2*R * np.sin(theta_vec6)

        Checkpoints_x7 = np.zeros(n_checkpoints) - 5*R
        Checkpoints_y7 = np.linspace(3 * R, 0, n_checkpoints)

        Checkpoints_x8 = - 4.5 * R + 0.5 * R * np.cos(theta_vec8)
        Checkpoints_y8 = + 0.5 * R * np.sin(theta_vec8)

        Checkpoints_x9 = np.zeros(n_checkpoints) - 4*R
        Checkpoints_y9 = np.linspace(0, 2*R, n_checkpoints)

        Checkpoints_x10 = - 3.5 * R + 0.5 * R * np.cos(theta_vec10)
        Checkpoints_y10 = + 2 * R + 0.5 * R * np.sin(theta_vec10)

        Checkpoints_x11 = np.zeros(n_checkpoints) - 3*R
        Checkpoints_y11 = np.linspace(2 * R, 0, n_checkpoints)

        Checkpoints_x12 = - 2.5 * R + 0.5 * R * np.cos(theta_vec8)
        Checkpoints_y12 = + 0.5 * R * np.sin(theta_vec8)

        Checkpoints_x13 = np.zeros(n_checkpoints) - 2*R
        Checkpoints_y13 = np.linspace(0, 2*R, n_checkpoints)

        Checkpoints_x14 = - 1.5 * R + 0.5 * R * np.cos(theta_vec10)
        Checkpoints_y14 = + 2 * R + 0.5 * R * np.sin(theta_vec10)

        Checkpoints_x15 = np.zeros(n_checkpoints) - 1 * R
        Checkpoints_y15 = np.linspace(2 * R, 0, n_checkpoints)

        Checkpoints_x16 = + R * np.cos(theta_vec12)
        Checkpoints_y16 = + R * np.sin(theta_vec12)

        Checkpoints_x = [*Checkpoints_x1[0:n_checkpoints - 1],
                         *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1],
                         *Checkpoints_x4[0:n_checkpoints - 1],
                         *Checkpoints_x5[0:n_checkpoints - 1],
                         *Checkpoints_x6[0:n_checkpoints - 1],
                         *Checkpoints_x7[0:n_checkpoints - 1],
                         *Checkpoints_x8[0:n_checkpoints - 1],
                         *Checkpoints_x9[0:n_checkpoints - 1],
                         *Checkpoints_x10[0:n_checkpoints - 1],
                         *Checkpoints_x11[0:n_checkpoints - 1],
                         *Checkpoints_x12[0:n_checkpoints - 1],
                         *Checkpoints_x13[0:n_checkpoints - 1],
                         *Checkpoints_x14[0:n_checkpoints - 1],
                         *Checkpoints_x15[0:n_checkpoints - 1],
                         *Checkpoints_x16[0:n_checkpoints]]

        Checkpoints_y = [*Checkpoints_y1[0:n_checkpoints - 1],
                         *Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints - 1],
                         *Checkpoints_y4[0:n_checkpoints - 1],
                         *Checkpoints_y5[0:n_checkpoints - 1],
                         *Checkpoints_y6[0:n_checkpoints - 1],
                         *Checkpoints_y7[0:n_checkpoints - 1],
                         *Checkpoints_y8[0:n_checkpoints - 1],
                         *Checkpoints_y9[0:n_checkpoints - 1],
                         *Checkpoints_y10[0:n_checkpoints - 1],
                         *Checkpoints_y11[0:n_checkpoints - 1],
                         *Checkpoints_y12[0:n_checkpoints - 1],
                         *Checkpoints_y13[0:n_checkpoints - 1],
                         *Checkpoints_y14[0:n_checkpoints - 1],
                         *Checkpoints_y15[0:n_checkpoints - 1],
                         *Checkpoints_y16[0:n_checkpoints]]


    elif choice == 'circle':
        n_checkpoints = 4 * n_checkpoints
        R = 0.3  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.82
        theta_init = np.pi * -0.5
        theta_end = np.pi * 1.5
        theta_vec = np.linspace(theta_init, theta_end, n_checkpoints)
        Checkpoints_x = R * np.cos(theta_vec)
        Checkpoints_y = R * np.sin(theta_vec)

    elif choice == 'gain_sweep_track':
        R = 0.4  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48
        straight_bit_half_length = 2.3

        theta_init2 = np.pi * -0.5
        theta_end2 = np.pi * 0.5
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
        theta_init4 = np.pi * 0.5
        theta_end4 = np.pi * 1.5
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)
        Checkpoints_x1 = np.linspace(- straight_bit_half_length, straight_bit_half_length, n_checkpoints)
        Checkpoints_y1 = np.zeros(n_checkpoints) - R
        Checkpoints_x2 = straight_bit_half_length + R * np.cos(theta_vec2)
        Checkpoints_y2 = R * np.sin(theta_vec2)
        Checkpoints_x3 = np.linspace(straight_bit_half_length, -straight_bit_half_length, n_checkpoints)
        Checkpoints_y3 = R * np.ones(n_checkpoints)
        Checkpoints_x4 = -straight_bit_half_length + R * np.cos(theta_vec4)
        Checkpoints_y4 = R * np.sin(theta_vec4)

        Checkpoints_x = [*Checkpoints_x1[0:n_checkpoints - 1], *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1], *Checkpoints_x4[0:n_checkpoints]]
        Checkpoints_y = [*Checkpoints_y1[0:n_checkpoints - 1], *Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints - 1], *Checkpoints_y4[0:n_checkpoints]]

    elif choice == 'gain_sweep_track_2':
        R = 0.49  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48
        straight_bit = 0.5

        Checkpoints_x1, Checkpoints_y1 = straight([0, straight_bit], [0, 0], n_checkpoints)
        Checkpoints_x2, Checkpoints_y2 = curve([straight_bit ,1 * R], R, [-0.5, 0.5], n_checkpoints)
        Checkpoints_x3, Checkpoints_y3 = straight([straight_bit, 0], [2*R, 2*R], n_checkpoints)
        Checkpoints_x4, Checkpoints_y4 = curve([0,3 * R], R, [1.5, 0.5], n_checkpoints)
        Checkpoints_x5, Checkpoints_y5 = straight([0, 10*straight_bit], [8*R, 8*R], 10*n_checkpoints)



        Checkpoints_y = [*Checkpoints_x1[0:n_checkpoints - 1],
                         *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1],
                         *Checkpoints_x4[0:n_checkpoints - 1],
                         *Checkpoints_x1[0:n_checkpoints - 1],
                         *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1],
                         *Checkpoints_x4[0:n_checkpoints - 1],
                         *Checkpoints_x5[0:]]

        Checkpoints_x = [*Checkpoints_y1[0:n_checkpoints - 1],
                         *Checkpoints_y2[0:n_checkpoints - 1],
                         *Checkpoints_y3[0:n_checkpoints - 1],
                         *Checkpoints_y4[0:n_checkpoints - 1],
                         *Checkpoints_y1[0:n_checkpoints - 1]+4*R,
                         *Checkpoints_y2[0:n_checkpoints - 1]+4*R,
                         *Checkpoints_y3[0:n_checkpoints - 1]+4*R,
                         *Checkpoints_y4[0:n_checkpoints - 1]+4*R,
                         *Checkpoints_y5[0:]]

    elif choice == 'racetrack_Lab':
        R = 0.5  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48

        # DEFINED STARTING FROM  START POINT AND THEN SHIFT IT LATER IF NEEDED


        Checkpoints_x1, Checkpoints_y1 = straight([-1+3*R, 1], [0, 0], n_checkpoints)

        Checkpoints_x2, Checkpoints_y2 = curve([1, 2*R], 2 * R, [-0.5, 0.5], n_checkpoints)

        Checkpoints_x3, Checkpoints_y3 = straight([1, -1], [4*R, 4*R], n_checkpoints)

        Checkpoints_x4, Checkpoints_y4 = curve([-1, 2.5*R], 1.5 * R, [0.5, 1.5], n_checkpoints)

        Checkpoints_x5, Checkpoints_y5 = straight([-1, 1], [R, R], n_checkpoints)

        Checkpoints_x6, Checkpoints_y6 = curve([1, 2 * R], 1 * R, [-0.5, 0.5], n_checkpoints)

        Checkpoints_x7, Checkpoints_y7 = straight([1, -1], [3 * R, 2 * R], n_checkpoints)

        Checkpoints_x8, Checkpoints_y8 = curve([-1, 3 * R], 1 * R, [1.5, 0], n_checkpoints)

        Checkpoints_x9, Checkpoints_y9 = straight([-1 + R, -1 + R], [3 * R, 2 * R], n_checkpoints)

        Checkpoints_x10, Checkpoints_y10 = curve([-1 + 3*R, 2 * R], 2 * R, [-1.0, -0.5], n_checkpoints)

        Checkpoints_x = [*Checkpoints_x1[0:n_checkpoints - 1],
                         *Checkpoints_x2[0:n_checkpoints - 1],
                         *Checkpoints_x3[0:n_checkpoints - 1],
                         *Checkpoints_x4[0:n_checkpoints - 1],
                         *Checkpoints_x5[0:n_checkpoints - 1],
                         *Checkpoints_x6[0:n_checkpoints - 1],
                         *Checkpoints_x7[0:n_checkpoints - 1],
                         *Checkpoints_x8[0:n_checkpoints - 1],
                         *Checkpoints_x9[0:n_checkpoints - 1],
                         *Checkpoints_x10[0:n_checkpoints]]
        y_shift = 2*R  #towards the bottom
        Checkpoints_y = [*Checkpoints_y1[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y2[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y3[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y4[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y5[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y6[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y7[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y8[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y9[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y10[0:n_checkpoints] - y_shift]

    elif choice == 'racetrack_Lab_safety_GP':
        R = 1  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.48

        # DEFINED STARTING FROM  START POINT AND THEN SHIFT IT LATER IF NEEDED
        


        Checkpoints_x1, Checkpoints_y1 = straight([-2, 2], [-1.5, -1.5], n_checkpoints)

        Checkpoints_x2, Checkpoints_y2 = curve([2, -0.5], R, [-0.5, 0], n_checkpoints)

        Checkpoints_x3, Checkpoints_y3 = straight([3, 3], [-0.5, 0.5], n_checkpoints)

        Checkpoints_x4, Checkpoints_y4 = curve([2, 0.5], R, [0, 1], n_checkpoints)

        Checkpoints_x5, Checkpoints_y5 = curve([0, 0.5], R, [0, -1], n_checkpoints)

        Checkpoints_x6, Checkpoints_y6 = curve([-2, 0.5], R, [0, 1], n_checkpoints)

        Checkpoints_x7, Checkpoints_y7 = straight([-3, -3], [0.5, -0.5], n_checkpoints)

        Checkpoints_x8, Checkpoints_y8 = curve([-2, -0.5], R, [-1, -0.5], n_checkpoints)

        #Checkpoints_x9, Checkpoints_y9 = curve([-1.5, -1], 0.5*R, [1, 0], n_checkpoints)

        #Checkpoints_x10, Checkpoints_y10 = curve([-0.5, -1], 0.5*R, [-1, -0.5], n_checkpoints)
        x_shift = 0.5
        Checkpoints_x = [*Checkpoints_x1[0:n_checkpoints - 1] + x_shift,
                         *Checkpoints_x2[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x3[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x4[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x5[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x6[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x7[0:n_checkpoints - 1]+ x_shift,
                         *Checkpoints_x8[0:n_checkpoints]+ x_shift]
        y_shift = 0  #towards the bottom
        Checkpoints_y = [*Checkpoints_y1[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y2[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y3[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y4[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y5[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y6[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y7[0:n_checkpoints - 1] - y_shift,
                         *Checkpoints_y8[0:n_checkpoints] - y_shift]


    elif choice == 'racetrack_vicon':

        #l_large = 8.7
        l_large = 7
        l_small = 5.5

        l_start = l_large / 2 # initial point (necessary for pure pursuit smoothing)

        R1 = 0.5
        R2 = 1.0

        theta_init1 = np.pi * (-0.5)
        theta_end1 = theta_init1 + np.pi * (0.5)  
        theta_vec1 = np.linspace(theta_init1, theta_end1, n_checkpoints)

        theta_init2 = theta_end1
        theta_end2 = theta_init2 + np.pi * (1.0)  
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)

        theta_init3 = 0. * np.pi
        theta_end3 = - np.pi * (1.0)  
        theta_vec3 = np.linspace(theta_init3, theta_end3, n_checkpoints)

        theta_init4 = 0
        theta_end4 = np.pi
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)

        theta_init5 = - 1 * np.pi
        theta_end5 = - 0.5 * np.pi
        theta_vec5 = np.linspace(theta_init5, theta_end5, n_checkpoints)

        # adding R location in 3D to resolve ambiguity
        theta_R1_init = 0
        theta_R1_end = np.pi
        theta_R1_vec = np.linspace(theta_R1_init, theta_R1_end, n_checkpoints)


        # 1
        Checkpoints_x1 = np.linspace(l_start, l_large-R2 , n_checkpoints)
        Checkpoints_y1 = np.linspace(0,0 , n_checkpoints)
        Checkpoints_k1 = 0 * np.ones(n_checkpoints)

        # 2
        Checkpoints_x2 = Checkpoints_x1[-1] + R2 * np.cos(theta_vec1)
        Checkpoints_y2 = R2 + R2 * np.sin(theta_vec1)
        Checkpoints_k2 = + 1 / R2 * np.ones(n_checkpoints)

        # 3
        Checkpoints_x3 = np.linspace(l_large, l_large, n_checkpoints)
        Checkpoints_y3 = np.linspace(R2, l_small-R1, n_checkpoints)
        Checkpoints_k3 = 0 * np.ones(n_checkpoints)

        # 4
        Checkpoints_x4 = l_large - R1 + R1 * np.cos(theta_vec2)
        Checkpoints_y4 = l_small - R1 + R1 * np.sin(theta_vec2)
        Checkpoints_k4 = + 1 / R1 * np.ones(n_checkpoints)

        # 5
        Checkpoints_x5 = np.linspace(Checkpoints_x4[-1], Checkpoints_x4[-1], n_checkpoints)
        Checkpoints_y5 = np.linspace(Checkpoints_y4[-1], R1+R2, n_checkpoints)
        Checkpoints_k5 = 0 * np.ones(n_checkpoints)

        # 6
        Checkpoints_x6 = l_large - 2*R1 - R2 + R2 * np.cos(theta_vec3)
        Checkpoints_y6 = R1 + R2 + R2 * np.sin(theta_vec3)
        Checkpoints_k6 = - 1 / R2 * np.ones(n_checkpoints)

        # 7
        Checkpoints_x7 = np.linspace(Checkpoints_x6[-1], Checkpoints_x6[-1], n_checkpoints)
        Checkpoints_y7 = np.linspace(Checkpoints_y6[-1], l_small-R2, n_checkpoints)
        Checkpoints_k7 = 0 * np.ones(n_checkpoints)

        # 8
        Checkpoints_x8 = 4 * R1 + R2 + R2 * np.cos(theta_vec4)
        Checkpoints_y8 = l_small - R2 + R2 * np.sin(theta_vec4)
        Checkpoints_k8 = + 1 / R2 * np.ones(n_checkpoints)

        # 9
        Checkpoints_x9 = np.linspace(Checkpoints_x8[-1], Checkpoints_x8[-1], n_checkpoints)
        Checkpoints_y9 = np.linspace(Checkpoints_y8[-1], 0.5 + R1, n_checkpoints)
        Checkpoints_k9 = 0 * np.ones(n_checkpoints)

        # 10
        Checkpoints_x10 =  R1 + R2 + R1 * np.cos(theta_vec3)
        Checkpoints_y10 = 0.5 + R1 + R1 * np.sin(theta_vec3)
        Checkpoints_k10 = - 1 / R1 * np.ones(n_checkpoints)

        # 11
        Checkpoints_x11 = np.linspace(Checkpoints_x10[-1], Checkpoints_x10[-1], n_checkpoints)
        Checkpoints_y11 = np.linspace(Checkpoints_y10[-1], l_small - R1, n_checkpoints)
        Checkpoints_k11 = 0 * np.ones(n_checkpoints)

        # 12
        Checkpoints_x12 = R1 + R1 * np.cos(theta_vec2)
        Checkpoints_y12 = l_small - R1 + R1 * np.sin(theta_vec2)
        Checkpoints_k12 = + 1 / R1 * np.ones(n_checkpoints)

        # 13
        Checkpoints_x13 = np.linspace(Checkpoints_x12[-1], Checkpoints_x12[-1], n_checkpoints)
        Checkpoints_y13 = np.linspace(Checkpoints_y12[-1], R2, n_checkpoints)
        Checkpoints_k13 = 0 * np.ones(n_checkpoints)

        # 14
        Checkpoints_x14 = R2 + R2 * np.cos(theta_vec5)
        Checkpoints_y14 = R2 + R2 * np.sin(theta_vec5)
        Checkpoints_k14 = + 1 / R2 * np.ones(n_checkpoints)

        # 15
        Checkpoints_x15 = np.linspace(R2, l_start , n_checkpoints)
        Checkpoints_y15 = np.linspace(0,0 , n_checkpoints)
        Checkpoints_k15 = 0 * np.ones(n_checkpoints)


        # Concatenate all checkpoints
        Checkpoints_x = np.concatenate((
                                        Checkpoints_x1[0:n_checkpoints - 1],
                                        Checkpoints_x2[0:n_checkpoints - 1],
                                        Checkpoints_x3[0:n_checkpoints - 1],
                                        Checkpoints_x4[0:n_checkpoints - 1],
                                        Checkpoints_x5[0:n_checkpoints - 1],
                                        Checkpoints_x6[0:n_checkpoints - 1],
                                        Checkpoints_x7[0:n_checkpoints - 1],
                                        Checkpoints_x8[0:n_checkpoints - 1],
                                        Checkpoints_x9[0:n_checkpoints - 1],
                                        Checkpoints_x10[0:n_checkpoints - 1],
                                        Checkpoints_x11[0:n_checkpoints - 1],
                                        Checkpoints_x12[0:n_checkpoints - 1],
                                        Checkpoints_x13[0:n_checkpoints - 1],
                                        Checkpoints_x14[0:n_checkpoints -1],
                                        Checkpoints_x15[0:n_checkpoints-1],
                                        ),
                                        axis=0)

        Checkpoints_y = np.concatenate((
                                        Checkpoints_y1[0:n_checkpoints - 1],
                                        Checkpoints_y2[0:n_checkpoints - 1],
                                        Checkpoints_y3[0:n_checkpoints - 1],
                                        Checkpoints_y4[0:n_checkpoints - 1],
                                        Checkpoints_y5[0:n_checkpoints - 1],
                                        Checkpoints_y6[0:n_checkpoints - 1],
                                        Checkpoints_y7[0:n_checkpoints - 1],
                                        Checkpoints_y8[0:n_checkpoints - 1],
                                        Checkpoints_y9[0:n_checkpoints - 1],
                                        Checkpoints_y10[0:n_checkpoints - 1],
                                        Checkpoints_y11[0:n_checkpoints - 1],
                                        Checkpoints_y12[0:n_checkpoints - 1],
                                        Checkpoints_y13[0:n_checkpoints - 1],
                                        Checkpoints_y14[0:n_checkpoints -1 ],
                                        Checkpoints_y15[0:n_checkpoints-1] ,
                                        ),
                                        axis=0)
        
        Checkpoints_k = np.concatenate((
                                        Checkpoints_k1[0:n_checkpoints - 1],
                                        Checkpoints_k2[0:n_checkpoints - 1],
                                        Checkpoints_k3[0:n_checkpoints - 1],
                                        Checkpoints_k4[0:n_checkpoints - 1],
                                        Checkpoints_k5[0:n_checkpoints - 1],
                                        Checkpoints_k6[0:n_checkpoints - 1],
                                        Checkpoints_k7[0:n_checkpoints - 1],
                                        Checkpoints_k8[0:n_checkpoints - 1],
                                        Checkpoints_k9[0:n_checkpoints - 1],
                                        Checkpoints_k10[0:n_checkpoints - 1],
                                        Checkpoints_k11[0:n_checkpoints - 1],
                                        Checkpoints_k12[0:n_checkpoints - 1],
                                        Checkpoints_k13[0:n_checkpoints - 1],
                                        Checkpoints_k14[0:n_checkpoints -1],
                                        Checkpoints_k15[0:n_checkpoints- 1],
                                        ),
                                        axis=0)



    elif choice == 'racetrack_vicon_2':

        #l_large = 8.7
        l_large = 4.5
        l_small = 4.5

        l_start = l_large / 2 # initial point (necessary for pure pursuit smoothing)

        R = 0.75

        theta_init1 = np.pi * (-0.5)
        theta_end1 = theta_init1 + np.pi * (0.5)  
        theta_vec1 = np.linspace(theta_init1, theta_end1, n_checkpoints)

        theta_init2 = theta_end1
        theta_end2 = theta_init2 + np.pi * (1.0)  
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)

        theta_init3 = 0. * np.pi
        theta_end3 = - np.pi * (1.0)  
        theta_vec3 = np.linspace(theta_init3, theta_end3, n_checkpoints)

        theta_init4 = 0
        theta_end4 = np.pi
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)

        theta_init5 = - 1 * np.pi
        theta_end5 = - 0.5 * np.pi
        theta_vec5 = np.linspace(theta_init5, theta_end5, n_checkpoints)

        # adding R location in 3D to resolve ambiguity
        theta_R1_init = 0
        theta_R1_end = np.pi
        theta_R1_vec = np.linspace(theta_R1_init, theta_R1_end, n_checkpoints)

        # 1
        Checkpoints_x1 = np.linspace(l_start, l_large-R, n_checkpoints)
        Checkpoints_y1 = np.linspace(0,0 , n_checkpoints)
        Checkpoints_k1 = 0 * np.ones(n_checkpoints)

        # 2
        Checkpoints_x2 = Checkpoints_x1[-1] + R * np.cos(theta_vec1)
        Checkpoints_y2 = R + R * np.sin(theta_vec1)
        Checkpoints_k2 = + 1 / R * np.ones(n_checkpoints)

        # 3
        Checkpoints_x3 = np.linspace(l_large, l_large, n_checkpoints)
        Checkpoints_y3 = np.linspace(R, l_small-R, n_checkpoints)
        Checkpoints_k3 = 0 * np.ones(n_checkpoints)

        # 4
        Checkpoints_x4 = l_large - R + R * np.cos(theta_vec2)
        Checkpoints_y4 = l_small - R + R * np.sin(theta_vec2)
        Checkpoints_k4 = + 1 / R * np.ones(n_checkpoints)

        # 5
        Checkpoints_x5 = np.linspace(Checkpoints_x4[-1], Checkpoints_x4[-1], n_checkpoints)
        Checkpoints_y5 = np.linspace(Checkpoints_y4[-1], l_small-R-0.5, n_checkpoints) #R+R
        Checkpoints_k5 = 0 * np.ones(n_checkpoints)

        # 6
        Checkpoints_x6 = l_large - 2*R - R + R * np.cos(theta_vec3)
        Checkpoints_y6 = Checkpoints_y5[-1] + R * np.sin(theta_vec3) # R + R
        Checkpoints_k6 = - 1 / R * np.ones(n_checkpoints)

        # 7
        Checkpoints_x7 = np.linspace(Checkpoints_x6[-1], Checkpoints_x6[-1], n_checkpoints)
        Checkpoints_y7 = np.linspace(Checkpoints_y6[-1], l_small-R, n_checkpoints)
        Checkpoints_k7 = 0 * np.ones(n_checkpoints)

        # 8
        Checkpoints_x8 = Checkpoints_x7[-1] - R + R * np.cos(theta_vec4)
        Checkpoints_y8 = l_small - R + R * np.sin(theta_vec4)
        Checkpoints_k8 = + 1 / R * np.ones(n_checkpoints)

        # 9
        Checkpoints_x9 = np.linspace(Checkpoints_x8[-1], Checkpoints_x8[-1], n_checkpoints)
        Checkpoints_y9 = np.linspace(Checkpoints_y8[-1], R, n_checkpoints)
        Checkpoints_k9 = 0 * np.ones(n_checkpoints)

        # 10
        Checkpoints_x10 = R + R * np.cos(theta_vec5)
        Checkpoints_y10 = R + R * np.sin(theta_vec5)
        Checkpoints_k10 = + 1 / R * np.ones(n_checkpoints)

        # 11
        Checkpoints_x11= np.linspace(R, l_start , n_checkpoints)
        Checkpoints_y11 = np.linspace(0,0 , n_checkpoints)
        Checkpoints_k11 = 0 * np.ones(n_checkpoints)

        # Concatenate all checkpoints
        Checkpoints_x = np.concatenate((
                                        Checkpoints_x1[0:n_checkpoints - 1],
                                        Checkpoints_x2[0:n_checkpoints - 1],
                                        Checkpoints_x3[0:n_checkpoints - 1],
                                        Checkpoints_x4[0:n_checkpoints - 1],
                                        Checkpoints_x5[0:n_checkpoints - 1],
                                        Checkpoints_x6[0:n_checkpoints - 1],
                                        Checkpoints_x7[0:n_checkpoints - 1],
                                        Checkpoints_x8[0:n_checkpoints - 1],
                                        Checkpoints_x9[0:n_checkpoints - 1],
                                        Checkpoints_x10[0:n_checkpoints - 1],
                                        Checkpoints_x11[0:n_checkpoints - 1],
                                        ),
                                        axis=0)

        Checkpoints_y = np.concatenate((
                                        Checkpoints_y1[0:n_checkpoints - 1],
                                        Checkpoints_y2[0:n_checkpoints - 1],
                                        Checkpoints_y3[0:n_checkpoints - 1],
                                        Checkpoints_y4[0:n_checkpoints - 1],
                                        Checkpoints_y5[0:n_checkpoints - 1],
                                        Checkpoints_y6[0:n_checkpoints - 1],
                                        Checkpoints_y7[0:n_checkpoints - 1],
                                        Checkpoints_y8[0:n_checkpoints - 1],
                                        Checkpoints_y9[0:n_checkpoints - 1],
                                        Checkpoints_y10[0:n_checkpoints - 1],
                                        Checkpoints_y11[0:n_checkpoints - 1],
                                        ),
                                        axis=0)

        
        Checkpoints_k = np.concatenate((
                                        Checkpoints_k1[0:n_checkpoints - 1],
                                        Checkpoints_k2[0:n_checkpoints - 1],
                                        Checkpoints_k3[0:n_checkpoints - 1],
                                        Checkpoints_k4[0:n_checkpoints - 1],
                                        Checkpoints_k5[0:n_checkpoints - 1],
                                        Checkpoints_k6[0:n_checkpoints - 1],
                                        Checkpoints_k7[0:n_checkpoints - 1],
                                        Checkpoints_k8[0:n_checkpoints - 1],
                                        Checkpoints_k9[0:n_checkpoints - 1],
                                        Checkpoints_k10[0:n_checkpoints - 1],
                                        Checkpoints_k11[0:n_checkpoints - 1],
                                        ),
                                        axis=0)



    else:
        print('Invalid choice of track:')
        print('You selected: ', choice)

    # fit a cubic spline to the checkpoints
    #tck, u = interpolate.splprep([Checkpoints_x, Checkpoints_y], s=0)
    # # from scipy.interpolate import splprep, splev
    # # tck, u = splprep([Checkpoints_x, Checkpoints_y], s=0)  # `s=0` for an exact fit (no smoothing)

    # # # Generate smooth points along the B-spline
    # # s_fine = np.linspace(0, 1, 1000)  # Parameter values for evaluation
    # # Checkpoints_x, Checkpoints_y = splev(s_fine, tck)



    Checkpoints_s = np.zeros(len(Checkpoints_x))
    
    #compute arc length
    for ii in range(1, len(Checkpoints_x)):
        Checkpoints_s[ii] = Checkpoints_s[ii - 1] + np.sqrt((Checkpoints_x[ii] - Checkpoints_x[ii - 1]) ** 2 + (Checkpoints_y[ii] - Checkpoints_y[ii - 1]) ** 2)
    

    # Checkpoints_Rx + x_shift_vicon_lab, Checkpoints_Ry + y_shift_vicon_lab, Checkpoints_Rz,
    return Checkpoints_x + x_shift_vicon_lab, Checkpoints_y + y_shift_vicon_lab, Checkpoints_s , Checkpoints_k



















import casadi

def K_RBF_kernel(x1, x2, lengthscale,n,m):
    #check if x1 is SX casadi type
    if type(x1) == casadi.SX:
        K = casadi.SX.zeros((n, m))
    elif type(x1) == casadi.MX:
        K = casadi.MX.zeros((n, m))
    else:
        K = np.zeros((n, m))

    for ii in range(n):
        for jj in range(m):
            K[ii, jj] = np.exp((-(x1[ii]-x2[jj])**2/(2*lengthscale**2)))
    return K

def K_matern2_kernel(x1, x2, lengthscale,n,m):
    sqrt5 = np.sqrt(5)
    #check if x1 is SX casadi type
    if type(x1) == casadi.SX:
        K = casadi.SX.zeros((n, m))
        abs = lambda x: casadi.fabs(x)

    elif type(x1) == casadi.MX:
        K = casadi.MX.zeros((n, m))
        abs = lambda x: casadi.fabs(x)

    else:
        K = np.zeros((n, m))
        abs = lambda x: np.abs(x)

    for ii in range(n):
        for jj in range(m):
            r = abs((x1[ii]-x2[jj])) / lengthscale
            K[ii, jj] = (1 + sqrt5 * r + (5.0 / 3.0) * r**2) * np.exp(-sqrt5 * r)
    return K








# do the same for the derivatives
def K_RBF_kernel_derivative(x1, x2, lengthscale):
    K = np.zeros((len(x1), len(x2)))
    for ii in range(len(x1)):
        for jj in range(len(x2)):
            K[ii, jj] = (x1[ii]-x2[jj]) * np.exp((-(x1[ii]-x2[jj])**2/(2*lengthscale**2)))/(lengthscale**2)
    return K


def kernelized_path(x_star, x , y,lengthscale,lambda_val):
    #K_xx = K_RBF_kernel(x, x, lengthscale)
    fixed_matrix, s_normalized = generate_fixed_path_quantities(lengthscale, lambda_val,len(x))
    right_side_vector_block = fixed_matrix @ y
    
    # evalaute the kernelized function in the query point (here we assume that x_star is a single value)
    #K_x_star = K_RBF_kernel(x_star, x, lengthscale,len(x_star),len(x))
    K_x_star = K_matern2_kernel(x_star, x, lengthscale,len(x_star),len(x))


    # evalaute products
    y_star = K_x_star @ right_side_vector_block

    return y_star

def generate_fixed_path_quantities(lengthscale, lambda_val,n_points):
    normalized_s = np.linspace(0.0, 1.0, n_points)
    # K_xx = K_RBF_kernel(normalized_s, normalized_s, lengthscale,n_points,n_points)
    K_xx = K_matern2_kernel(normalized_s, normalized_s, lengthscale,n_points,n_points)
    fixed_matrix = np.linalg.inv(K_xx + lambda_val * np.eye(len(normalized_s))) 
    return fixed_matrix, normalized_s





def generate_path_data(track_choice):

    # # --- set up track ---
    Checkpoints_x, Checkpoints_y, Checkpoints_s,\
    Checkpoints_k = raw_track(track_choice) # Checkpoints_Rx , Checkpoints_Ry , Checkpoints_Rz,



    # add a loop before and a loop afterwards to generate extra reference for the MPC
    x_4_local_path = np.concatenate((Checkpoints_x[:-1],Checkpoints_x, Checkpoints_x[1:]))
    y_4_local_path = np.concatenate((Checkpoints_y[:-1],Checkpoints_y, Checkpoints_y[1:]))
    s_4_local_path = np.concatenate((Checkpoints_s[:-1] -Checkpoints_s[-1],Checkpoints_s, Checkpoints_s[1:] + Checkpoints_s[-1]))

    # R_x_4_local_path = np.concatenate((Checkpoints_Rx[:-1],Checkpoints_Rx, Checkpoints_Rx[1:]))
    # R_y_4_local_path = np.concatenate((Checkpoints_Ry[:-1],Checkpoints_Ry, Checkpoints_Ry[1:]))
    # R_z_4_local_path = np.concatenate((Checkpoints_Rz[:-1],Checkpoints_Rz, Checkpoints_Rz[1:]))

    k_4_local_path = np.concatenate((Checkpoints_k[:-1],Checkpoints_k, Checkpoints_k[1:]))
    


    # smooth out the R localtion
    # window = 21
    # R_x_4_local_path = sliding_window_smooth(R_x_4_local_path, window)
    # R_y_4_local_path = sliding_window_smooth(R_y_4_local_path, window)
    # R_z_4_local_path = sliding_window_smooth(R_z_4_local_path, window)
    # # re smooth the R location
    # R_x_4_local_path = sliding_window_smooth(R_x_4_local_path, window)
    # R_y_4_local_path = sliding_window_smooth(R_y_4_local_path, window)
    # R_z_4_local_path = sliding_window_smooth(R_z_4_local_path, window)

    window = 5
    k_vals_global_path =  sliding_window_smooth(k_4_local_path, window)
    k_vals_global_path =  sliding_window_smooth(k_vals_global_path, window)


    # now use the trajectory to generate the smoothed track
    Ds_back = 1
    Ds_forward = 20    # these distances are before and after 1 lap 

    # #     R_x_vals_global_path,\
    # # R_y_vals_global_path,\
    # # R_z_vals_global_path,\
    # # R_x_4_local_path,\
    # # R_y_4_local_path,\
    # # R_z_4_local_path,\

    s_vals_global_path,\
    x_vals_global_path,\
    y_vals_global_path,\
    s_4_local_path,\
    x_4_local_path,\
    y_4_local_path,\
    k_vals_global_path,\
    k_4_local_path = generate_track(x_4_local_path, y_4_local_path, s_4_local_path, Ds_back, Ds_forward,\
                                      k_vals_global_path) # remove last point to avoid overlap     ## R_x_4_local_path,R_y_4_local_path,R_z_4_local_path,


    # genrate derivatives of the path
    dx_ds, dy_ds, d2x_ds2, d2y_ds2 = produce_ylabels_devs(s_4_local_path,x_4_local_path,y_4_local_path)

    # R_x_vals_global_path, R_y_vals_global_path, R_z_vals_global_path,\
    # R_x_4_local_path, R_y_4_local_path, R_z_4_local_path,\
    return s_vals_global_path, x_vals_global_path, y_vals_global_path, s_4_local_path, x_4_local_path, y_4_local_path,\
           dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
            k_vals_global_path, k_4_local_path






def sliding_window_smooth(data, window_size):
    """
    Smooth data using a sliding window (moving average).
    Args:
        data (np.ndarray): Input data to smooth.
        window_size (int): Size of the smoothing window (must be odd).
    Returns:
        np.ndarray: Smoothed data.
    """
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')  # Pad edges
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data





def produce_ylabels_devs(s_4_local_path,x_4_local_path,y_4_local_path):
            # --- evaluate first derivatives ---
    # we now also need to specify the curve derivatives
    ds = np.diff(s_4_local_path)
    dx = np.diff(x_4_local_path)
    dy = np.diff(y_4_local_path)

    dev_x = dx / ds
    dev_y = dy / ds

    # make sure the norm of the dev vector is 1
    norm_dev = np.sqrt(dev_x ** 2 + dev_y ** 2)

    #renormalize to make shure the norm is 1
    dev_x_renormalized = dev_x / norm_dev
    dev_y_renormalized = dev_y / norm_dev

    # --- evaluate cuvature radius and second derivatives ---
    # ddev_x = np.diff(dev_x_renormalized)
    # ddev_y = np.diff(dev_y_renormalized)

    # dev2_x = ddev_x / ds[:-1]
    # dev2_y = ddev_y / ds[:-1]

    cs_scipy_x = CubicSpline(s_4_local_path, x_4_local_path)
    cs_scipy_y = CubicSpline(s_4_local_path, y_4_local_path)

    # First derivatives
    dx_ds = cs_scipy_x(s_4_local_path, 1)  # First derivative of x with respect to s
    dy_ds = cs_scipy_y(s_4_local_path, 1)  # First derivative of y with respect to s

    # re normalize the first derivatives
    norm_dev = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
    dx_ds = dx_ds / norm_dev
    dy_ds = dy_ds / norm_dev

    # Second derivatives
    d2x_ds2 = cs_scipy_x(s_4_local_path, 2)  # Second derivative of x with respect to s
    d2y_ds2 = cs_scipy_y(s_4_local_path, 2)  # Second derivative of y with respect to s

    return dx_ds, dy_ds, d2x_ds2, d2y_ds2



if __name__ == '__main__':

    # testing by plotting 
    import matplotlib.pyplot as plt

    track_choice = 'racetrack_vicon_2'

    # # --- set up track ---
    Checkpoints_x, Checkpoints_y, Checkpoints_s,\
    Checkpoints_k = raw_track(track_choice)

    s_vals_global_path, x_vals_global_path, y_vals_global_path, s_4_local_path, x_4_local_path, y_4_local_path,\
    dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
    k_vals_global_path, k_4_local_path = generate_path_data(track_choice)


    # plot raw track
    plt.figure()
    plt.plot(x_4_local_path, y_4_local_path, 'gray',label='raw track 4 local path',linewidth=5)
    plt.plot(x_vals_global_path, y_vals_global_path, 'k',label='raw track 4 local path',linewidth=1)
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend()


    # plot curvature and heading angle
    # evaluate heading angle
    track_heading_angle_global = np.arctan2(dy_ds, dx_ds)


    figure, ax = plt.subplots(2)
    #plot curvature
    ax[0].plot(s_4_local_path, k_4_local_path, 'k',label='curvature 4 local path',linewidth=2)
    # plot heading angle
    ax[1].plot(s_4_local_path, track_heading_angle_global, 'purple',label='heading angle global path',linewidth=2)

    # set legend title axis labels
    ax[0].legend()
    ax[1].legend()
    ax[0].set(xlabel='s [m]', ylabel='curvature')
    ax[1].set(xlabel='s [m]', ylabel='heading angle [rad]')
    ax[0].set_title('Curvature')
    ax[1].set_title('Heading angle')

    plt.show()






    # chose sub interval to fit
    # difficult bend is at 8 m
    start_point = 6 # m
    length_path = 4 # m (full rack length is 41.11m)

    # set kernel parameters
    alpha = 0.0001**2
    n_points_kernelized = 41 # for full track
    length_scale = 1.3/n_points_kernelized #1/n_points_kernelized #4 * 1/n_points_kernelized # length scale is n times the distance between two points
    # was 0.1


    # extract path interval
    mask = (s_4_local_path >= start_point) & (s_4_local_path <= start_point + length_path)
    # Extract the indexes where the condition is true
    indexes = np.where(mask)[0]

    s_data_points_fit = s_4_local_path[indexes]
    k_points_fit = k_4_local_path[indexes]

    #x_data_points_fit = x_4_local_path[indexes]
    #y_data_points_fit = y_4_local_path[indexes]


    # Fit the model
    # resample with the specified number of points for kenerlized regression

    s_4_kernelized = np.linspace(s_data_points_fit[0], s_data_points_fit[-1], n_points_kernelized)
    #x_4_kernelized = np.interp(s_4_kernelized, s_data_points_fit, x_data_points_fit)
    #y_4_kernelized = np.interp(s_4_kernelized, s_data_points_fit, y_data_points_fit)
    k_4_kernelized = np.interp(s_4_kernelized, s_data_points_fit, k_4_local_path[indexes])



    # --- now evalaute the path using the custom made kernelized linear regression ---
    s_4_kernelized_eval_normalized = np.linspace(0, 1, n_points_kernelized)

    krr_x_output_custom  = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , x_4_kernelized ,length_scale, alpha)
    krr_y_output_custom  = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , y_4_kernelized ,length_scale, alpha)
    
    # evaluate first derivatives
    krr_x_output_custom_d = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , dev_x_kernelized ,length_scale, alpha)
    krr_y_output_custom_d = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , dev_y_kernelized ,length_scale, alpha)
    
    # evaluate second derivatives
    krr_x_output_custom_dd = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , dev2_x_kernelized ,length_scale, alpha)
    krr_y_output_custom_dd = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , dev2_y_kernelized ,length_scale, alpha)

    # evaluate radius of curvature
    R_vec_data_krr_custom = 1 / np.sqrt(krr_x_output_custom_dd ** 2 + krr_y_output_custom_dd ** 2 + (1/max_R)**2)

    # now fit the x y z R locations
    krr_Rx_output_custom  = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , R_x_4_kernelized ,length_scale, alpha)
    krr_Ry_output_custom  = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , R_y_4_kernelized ,length_scale, alpha)
    krr_Rz_output_custom  = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , R_z_4_kernelized ,length_scale, alpha)

    # # add kernelized linear regression to the plot
    # axs_R[0].plot(s_data_points_fit, krr_Rx_output_custom, 'violet',linestyle = '--',label='kernelized linear regression Rx')
    # axs_R[1].plot(s_data_points_fit, krr_Ry_output_custom, 'violet',linestyle = '--',label='kernelized linear regression Ry')
    # axs_R[2].plot(s_data_points_fit, krr_Rz_output_custom, 'violet',linestyle = '--',label='kernelized linear regression Rz')

    # axs_R[0].legend()
    # axs_R[1].legend()
    # axs_R[2].legend()

    # # add plot ocentre of curvature
    # ax_3d_path.plot(krr_Rx_output_custom, krr_Ry_output_custom, krr_Rz_output_custom, 'violet',linestyle = '--',label='centre of curvature kernelized linear regression')





    # curvature from the kernelized linear regression
    k_vec_from_kernelized = kernelized_path(s_4_kernelized_eval_normalized, s_4_kernelized_normalized , k_4_kernelized ,length_scale, alpha)





    
    # --- recreate path by integrating the curvature data ---
    discretization = len(k_vec_from_kernelized)
    #ds_for_k_reconstruction = np.linspace(0, 1, len(k_vec_from_kernelized))
    track_heading_vec = np.zeros(len(k_vec_from_kernelized))
    track_x_from_heading = np.zeros(len(k_vec_from_kernelized))
    track_y_from_heading = np.zeros(len(k_vec_from_kernelized))

    # assign initial values
    track_heading_vec[0] = np.arctan2(dy_ds[indexes][0], dx_ds[indexes][-1])
    track_x_from_heading[0] = x_data_points_fit[0]
    track_y_from_heading[0] = y_data_points_fit[0]

    for ii in range(1,discretization):
        ds = s_data_points_fit[ii] - s_data_points_fit[ii-1]
        track_heading_vec[ii] = track_heading_vec[ii-1] + k_vec_from_kernelized[ii] * ds 
        track_x_from_heading[ii] = track_x_from_heading[ii-1] + np.cos(track_heading_vec[ii-1]) * ds 
        track_y_from_heading[ii] = track_y_from_heading[ii-1] + np.sin(track_heading_vec[ii-1]) * ds

    # ------------------------------
    # plot the curvature info
    figure, ax = plt.subplots(1)
    ax.plot(s_4_local_path, k_4_local_path, 'k',label='curvature from checkpoints')
    ax.plot(s_data_points_fit, k_vec_from_kernelized, 'violet',label='curvature from kernelized linear regression')
    ax.legend()









    # plot the cheby path
    plt.figure()
    # # plot the original path
    # plt.plot(x_vals_global_path, y_vals_global_path, 'cadetblue',linestyle='--',label='global path raw track',linewidth=3)


    # plot built in curve
    plt.plot(x_data_points_fit, y_data_points_fit, 'k',linestyle='--',linewidth=3,label='data points',zorder = 20,alpha = 0.4)

    # plot custom curve
    plt.plot(x_custom, y_custom, 'purple',linestyle=':',linewidth=3,label='custom cheby path')

    #plot overall tack
    plt.plot(x_4_local_path, y_4_local_path, 'gray',label='global path',linewidth=5,zorder=0)

    # plot kernelized linear regression
    plt.plot(krr_x_output,krr_y_output , 'green',linestyle = '--',label='kernelized linear regression x')

    # plot kernelized linear regression custom
    plt.plot(krr_x_output_custom,krr_y_output_custom , 'violet',linestyle = '--',label='kernelized linear regression custom x',zorder = 21)

    # plot trajectory of pure pursuit controller
    plt.plot(x_4_local_path, y_4_local_path, 'yellowgreen',label='pure pursuit controller trajectory')

    # plot location of R
    plt.plot(krr_Rx_output_custom, krr_Ry_output_custom, 'or',label='centre of curvature')

    # plot reconstruction from curvature info
    plt.plot(track_x_from_heading, track_y_from_heading, 'blue',label='reconstructed path from curvature info')


    plt.legend()
    plt.show()



    # plot first derivatives
    figure, ax = plt.subplots(3)

    #evaluate norm of first derivatives
    norm_dev_data = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
    norm_dev__custom_k = np.sqrt(krr_x_output_custom_d ** 2 + krr_y_output_custom_d ** 2)

    #plot norm of first derivatives
    ax[0].plot(s_4_local_path, norm_dev_data, 'k',label='norm of first derivatives from data')
    ax[0].plot(s_data_points_fit, norm_dev__custom_k, 'violet',label='norm of first derivatives from custom kernelized linear regression')

    #plot first derivatives
    ax[1].plot(s_4_local_path, dx_ds, 'k',label='dev_x from data')
    ax[2].plot(s_4_local_path, dy_ds, 'k',label='dev_y from data')

    #plot first derivatives custom
    ax[1].plot(s_data_points_fit, krr_x_output_custom_d, 'violet',label='dev_x from custom kernelized linear regression')
    ax[2].plot(s_data_points_fit, krr_y_output_custom_d, 'violet',label='dev_y from custom kernelized linear regression')














    # plot radius of curvature
    figure, ax = plt.subplots(3)

    # # plot data R
    # ax[0].plot(s_4_local_path[:-2], R_vec_data, 'k',label='radius of curvature from data')

    # plot custom R
    ax[0].plot(s_cheby, R_vec, 'r',label='radius of curvature from custom cheby 2n dev')

    # #plot spline R
    ax[0].plot(s_4_local_path, R_vec_data_splines, 'k',label='radius of curvature from data (splines)')

    # plot cheby R
    ax[0].plot(s_cheby, R_cheby, 'b',label='radius of curvature from cheby separate polynomials for 2nddevs')

    # # plot data used to fit the cheby polynomials
    # ax[0].plot(s_data_points_fit, R_vec_data[indexes], 'orange',linestyle = ':',label='data points for cheby fit')

    # plot kernelized linear regression R
    ax[0].plot(s_data_points_fit, R_vec_data_krr, 'darkgreen',linestyle = '--',label='kernelized linear regression R')

    # plot kernelized linear regression custom R
    ax[0].plot(s_data_points_fit, R_vec_data_krr_custom, 'violet',linestyle = '--',label='kernelized linear regression custom R')


    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('R [m]')
    ax[0].legend()

    # plot in two subplots the second derivatives data
    ax[1].plot(s_4_local_path, d2x_ds2, 'k',label='dev2_x from data')
    ax[2].plot(s_4_local_path, d2y_ds2, 'k',label='dev2_y from data')
    # plot the re-evalauted data from cheby
    ax[1].plot(s_cheby, x_Cdev2_devs, 'r',label='dev2_x from cheby')
    ax[2].plot(s_cheby, y_Cdev2_devs, 'r',label='dev2_y from cheby')

    # plot the re-evalauted data from cheby coeff2
    ax[1].plot(s_cheby, x_Cdev2_from_coeff2, 'b',label='dev2_x from cheby coeff2')
    ax[2].plot(s_cheby, y_Cdev2_from_coeff2, 'b',label='dev2_y from cheby coeff2')

    # plot the kernelized linear regression
    ax[1].plot(s_data_points_fit, krr_x_output_2dev, 'g',label='kernelized linear regression dev2_x')
    ax[2].plot(s_data_points_fit, krr_y_output_2dev, 'g',label='kernelized linear regression dev2_y')

    # plot the custom kernelized linear regression
    ax[1].plot(s_data_points_fit, krr_x_output_custom_dd, 'violet',linestyle = '--',label='kernelized linear regression custom dev2_x')
    ax[2].plot(s_data_points_fit, krr_y_output_custom_dd, 'violet', linestyle = '--',label='kernelized linear regression custom dev2_y')

    # # plot smoothed data
    # ax[1].plot(s_4_local_path, d2x_ds2_smoothed, 'orange',label='dev2_x from smoothed data')
    # ax[2].plot(s_4_local_path, d2y_ds2_smoothed, 'orange',label='dev2_y from smoothed data')

    # plot the s interval of the global path as two vertical lines
    ax[0].axvline(x=s_vals_global_path[0],color='gray',linestyle='--')
    ax[0].axvline(x=s_vals_global_path[-1],color='gray',linestyle='--')
    # also on ax 1 and 2
    ax[1].axvline(x=s_vals_global_path[0],color='gray',linestyle='--')
    ax[1].axvline(x=s_vals_global_path[-1],color='gray',linestyle='--')
    ax[2].axvline(x=s_vals_global_path[0],color='gray',linestyle='--')
    ax[2].axvline(x=s_vals_global_path[-1],color='gray',linestyle='--')




    ax[1].set_xlabel('s [m]')
    ax[1].set_ylabel('dev2 x')
    ax[1].legend()
    ax[2].set_xlabel('s [m]')
    ax[2].set_ylabel('dev2 y')
    ax[2].legend()



    plt.show()






