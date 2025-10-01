import numpy as np
import sys
import math
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
import roslib
import os
pkg_path = roslib.packages.get_pkg_dir('racing_campcc_pkg')




# Realize the track structure for each available choice
def produce_track(choice,n_checkpoints):
    # initialize the gate as empty
    gates = []

    # tridimentional spline tracks, each tangent vector is defined at the extremity of the spline and represent the gate through which the drone must pass
    if choice == 'spline_circle':
        R = 3                                     # radius of the circle
        tg_val = 5                                # module of the tangent vector at the extremity of the spline

        ## first spline ##
        # define the points needed for the interpolation
        t1 = [0, 1]                               # t parameter of the spline
        xp1 = R                                   # 1D data for X dimension
        yp1 = 0                                # 1D data for Y dimension
        zp1 = 1                                   # 1D data for Z dimension
        # define the extremity values (start and stop of the spline) of the tangent values for x,y,z (for the end we're gonna use the start of the following spline)
        tg1_init = [0,tg_val,0]
        # assemble the spline information
        spline1 = [t1, xp1, yp1, zp1, tg1_init]

        ## second spline ##
        t2 = [0, 1]                 
        xp2 = 0
        yp2 = R
        zp2 = 1
        theta = np.pi*1
        tg2_init = [tg_val * np.cos(theta), tg_val * np.sin(theta), 0]
        spline2 = [t2, xp2, yp2, zp2, tg2_init]

        ## third spline ##
        t3 = [0, 1]                    
        xp3 = -R
        yp3 = 0
        zp3 = 1
        tg3_init = [0, -tg_val, 0]
        spline3 = [t3, xp3, yp3, zp3, tg3_init]

        ## fourth spline ##
        t4 = [0, 1]
        xp4 = 0
        yp4 = -R
        zp4 = 1
        tg4_init = [tg_val, 0, 0]
        spline4 = [t4, xp4, yp4, zp4, tg4_init]

        # generate the spline vector
        splines = [spline1, spline2, spline3, spline4]

        # generate the checkpoints vector
        Checkpoints_x, Checkpoints_y, Checkpoints_z,\
        Checkpoints_dx, Checkpoints_dy, Checkpoints_dz,\
        Checkpoints_ddx, Checkpoints_ddy, Checkpoints_ddz,\
        Checkpoints_s  = spline_assembler(n_checkpoints, splines) 

        ## produce the points to plot the gates ##
        # initialize the gate vector
        gates = np.zeros((len(splines), 6))

        # generate the gates vector
        for i in range(0, len(splines)):
            gates[i,:3] = splines[i][1:4]
            gates[i,3:6] = splines[i][4]

    elif choice == 'vicon_racetrack':
            
            tg_vals = 3 * np.ones(8)  # set to 1 nominally now (will be set to 5 later as first guess)  # module of the tangent vector at the extremity of the spline
            tg_val = tg_vals[0]  # module of the tangent vector at the extremity of the spline
            ## first spline ##
            t1 = [0, 1]                    
            xp1 = 0     
            yp1 = -3
            zp1 = 1
            a1 = np.pi*0.25
            tg1_init = [tg_val*np.cos(a1), tg_val*np.sin(a1), 0]            
            spline1 = [t1, xp1, yp1, zp1, tg1_init]

            ## second spline ##
            tg_val = tg_vals[1]
            t2 = [0, 1]                       
            xp2 = 1
            yp2 = 0
            zp2 = 2.5 
            a2 = np.pi*0.15
            tg2_init = [tg_val*np.cos(a2), tg_val*np.sin(a2), 0]
            spline2 = [t2, xp2, yp2, zp2, tg2_init]

            ## third spline ##
            tg_val = tg_vals[2]
            t3 = [0, 1]
            xp3 = 2
            yp3 = 2.5
            zp3 = 1
            a2 = np.pi*0.65
            tg3_init = [tg_val*np.cos(a2), tg_val*np.sin(a2), 0]
            spline3 = [t3, xp3, yp3, zp3, tg3_init]

            ## fourth spline ##
            tg_val = tg_vals[3]
            t4 = [0, 1]
            xp4 = -3
            yp4 = 3
            zp4 = 1
            a3 = np.pi*1.5
            tg4_init = [tg_val*np.cos(a3), tg_val*np.sin(a3), 0]
            spline4 = [t4, xp4, yp4, zp4, tg4_init]

            ## fifth spline ##
            tg_val = tg_vals[4]
            t5 = [0, 1]
            xp5 = -2
            yp5 = 1.5
            zp5 = 2
            tg5_init = [tg_val, 0, 0]
            spline5 = [t5, xp5, yp5, zp5, tg5_init]

            ## sixth spline ##
            tg_val = tg_vals[5]
            t6 = [0, 1]
            xp6 = -2
            yp6 = 0.5
            zp6 = 3
            tg6_init = [-tg_val, 0, 0]
            spline6 = [t6, xp6, yp6, zp6, tg6_init]

            ## seventh spline ##
            tg_val = tg_vals[6]
            t7 = [0, 1]
            xp7 = -2
            yp7 = -2
            zp7 = 2
            tg7_init = [tg_val, 0, 0]
            spline7 = [t7, xp7, yp7, zp7, tg7_init]

            ## eight spline ##
            # define the points needed for the interpolation
            tg_val = tg_vals[7]
            a_7 = np.pi*1.25
            t8 = [0, 1]                               # t parameter of the spline
            xp8 = -2                                  # 1D data for X dimension
            yp8 = 0                                   # 1D data for Y dimension
            zp8 = 1                                   # 1D data for Z dimension
            tg8_init = [tg_val*np.cos(a_7),tg_val* np.sin(a_7), 0]
            spline8 = [t8, xp8, yp8, zp8, tg8_init]   # assembled points

            # generate the spline vector
            splines = [spline1, spline2, spline3, spline4, spline5, spline6, spline7, spline8]

            # generate the checkpoints vector
            Checkpoints_x, Checkpoints_y, Checkpoints_z,\
            Checkpoints_dx, Checkpoints_dy, Checkpoints_dz,\
            Checkpoints_ddx, Checkpoints_ddy, Checkpoints_ddz,\
            Checkpoints_s  = spline_assembler(n_checkpoints, splines)   

            ## produce the points to plot the gates ##
            # initialize the gate vector
            gates = np.zeros((len(splines), 6))

            # generate the gates vector
            for i in range(0, len(splines)):
                gates[i,:3] = splines[i][1:4]
                gates[i,3:6] = splines[i][4]
    
    else:
        print('Invalid choice of track:')
        print('You selected: ', choice)

    return Checkpoints_x, Checkpoints_y, Checkpoints_z,\
            Checkpoints_dx, Checkpoints_dy, Checkpoints_dz,\
            Checkpoints_ddx, Checkpoints_ddy, Checkpoints_ddz, gates, Checkpoints_s


def resample_trajectory_by_s(optimally_smoothed_track, ds=0.01):
    """
    Resample the track so that s (col 8) is uniformly spaced by 'ds'.
    Columns (expected order):
    0:px 1:py 2:pz 3:roll 4:pitch 5:yaw
    6:wz 7:wx 8:s 9:dxds 10:dyds 11:dzds 12:d2xds2 13:d2yds2 14:d2zds2
    Returns:
    resampled_track: array with same columns, resampled at uniform s.
    """
    # Extract and sort by s to ensure monotonic input for splines
    s = optimally_smoothed_track[:, 8].astype(float)
    order = np.argsort(s)
    s = s[order]
    track_sorted = optimally_smoothed_track[order, :].astype(float)

    # New uniform s grid
    s_new = np.arange(s[0], s[-1] + 1e-12, ds)

    # Helper: cubic spline per column
    def cs_interp(y):
        return CubicSpline(s, y, bc_type='natural')(s_new)

    # Interpolate each column
    px_new   = cs_interp(track_sorted[:, 0])
    py_new   = cs_interp(track_sorted[:, 1])
    pz_new   = cs_interp(track_sorted[:, 2])
    roll_new = cs_interp(track_sorted[:, 3])
    pitch_new= cs_interp(track_sorted[:, 4])
    yaw_new  = cs_interp(track_sorted[:, 5])
    wz_new   = cs_interp(track_sorted[:, 6])
    wx_new   = cs_interp(track_sorted[:, 7])
    # s itself is the new grid
    dxds_new   = cs_interp(track_sorted[:, 9])
    dyds_new   = cs_interp(track_sorted[:, 10])
    dzds_new   = cs_interp(track_sorted[:, 11])
    d2xds2_new = cs_interp(track_sorted[:, 12])
    d2yds2_new = cs_interp(track_sorted[:, 13])
    d2zds2_new = cs_interp(track_sorted[:, 14])

    # Stack back in original order
    resampled_track = np.column_stack([
        px_new, py_new, pz_new,
        roll_new, pitch_new, yaw_new,
        wz_new, wx_new,
        s_new,
        dxds_new, dyds_new, dzds_new,
        d2xds2_new, d2yds2_new, d2zds2_new
    ])
    return resampled_track


def generate_path_data(track_choice, optimally_smoothed = False):
    # for local path generation
    Ds_back = 1
    Ds_forward = 20    # these distances are before and after 1 lap 

    if optimally_smoothed:
        # # load optimally smoothed track data
        # path_2_stored_optimally_smoothed_track = os.path.join(pkg_path, 'src', 'solvers_setup', 'offline_optimal_solutions', track_choice + '_optimally_smoothed.npy')
        # try:
        #     optimally_smoothed_track = np.load(path_2_stored_optimally_smoothed_track)
        #     Checkpoints_s_raw = optimally_smoothed_track[:, 9]
        #     Checkpoints_x_raw = optimally_smoothed_track[:, 0]
        #     Checkpoints_y_raw = optimally_smoothed_track[:, 1]
        #     Checkpoints_z_raw = optimally_smoothed_track[:, 2]

        #     Checkpoints_dx_ds_raw = optimally_smoothed_track[:, 3]
        #     Checkpoints_dy_ds_raw = optimally_smoothed_track[:, 4]
        #     Checkpoints_dz_ds_raw = optimally_smoothed_track[:, 5]

        #     Checkpoints_d2x_ds2_raw = optimally_smoothed_track[:, 6]
        #     Checkpoints_d2y_ds2_raw = optimally_smoothed_track[:, 7]
        #     Checkpoints_d2z_ds2_raw = optimally_smoothed_track[:, 8]

        #     # now produce local path and global path data
        #     s_4_local_path, x_4_local_path, s_vals_global_path, x_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_x_raw, Ds_back, Ds_forward)
        #     s_4_local_path, y_4_local_path, s_vals_global_path, y_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_y_raw, Ds_back, Ds_forward)
        #     s_4_local_path, z_4_local_path, s_vals_global_path, z_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_z_raw, Ds_back, Ds_forward)

        #     # same for the derivatives
        #     s_4_local_path, dx_ds, s_vals_global_path, dx_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dx_ds_raw, Ds_back, Ds_forward)
        #     s_4_local_path, dy_ds, s_vals_global_path, dy_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dy_ds_raw, Ds_back, Ds_forward)
        #     s_4_local_path, dz_ds, s_vals_global_path, dz_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dz_ds_raw, Ds_back, Ds_forward)

        #     # same for dev 2
        #     s_4_local_path, d2x_ds2, s_vals_global_path, d2x_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2x_ds2_raw, Ds_back, Ds_forward)
        #     s_4_local_path, d2y_ds2, s_vals_global_path, d2y_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2y_ds2_raw, Ds_back, Ds_forward)
        #     s_4_local_path, d2z_ds2, s_vals_global_path, d2z_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2z_ds2_raw, Ds_back, Ds_forward)

        #     # just for the gates position
        #     dummy1, dummy2, dummy3,\
        #     dummy4, dummy5, dummy6,\
        #     dummy7, dummy8, dummy8,\
        #     dummy9, gates  = raw_track(track_choice, 300) 


        # load optimally smoothed track data
        path_2_stored_optimally_smoothed_track = os.path.join(pkg_path, 'src', 'solvers_setup', 'offline_optimal_solutions', track_choice + '_optimally_smoothed_euler_angles.npy')
        try:
            optimally_smoothed_track = np.load(path_2_stored_optimally_smoothed_track)
            # px py pz roll pitch yaw wx wy s dxds dyds dzds d2xds2 d2yds2 d2zds2
            optimally_smoothed_track = resample_trajectory_by_s(optimally_smoothed_track, ds=0.01)
        
            Checkpoints_x_raw = optimally_smoothed_track[:, 0]
            Checkpoints_y_raw = optimally_smoothed_track[:, 1]
            Checkpoints_z_raw = optimally_smoothed_track[:, 2]

            Checkpoints_roll_raw = optimally_smoothed_track[:, 3]
            Checkpoints_pitch_raw = optimally_smoothed_track[:, 4]
            Checkpoints_yaw_raw = optimally_smoothed_track[:, 5]

            Checkpoints_wz_raw = optimally_smoothed_track[:, 6]
            Checkpoints_wx_raw = optimally_smoothed_track[:, 7]

            Checkpoints_s_raw = optimally_smoothed_track[:, 8]

            Checkpoints_dx_ds_raw = optimally_smoothed_track[:, 9]
            Checkpoints_dy_ds_raw = optimally_smoothed_track[:, 10]
            Checkpoints_dz_ds_raw = optimally_smoothed_track[:, 11]

            Checkpoints_d2x_ds2_raw = optimally_smoothed_track[:, 12]
            Checkpoints_d2y_ds2_raw = optimally_smoothed_track[:, 13]
            Checkpoints_d2z_ds2_raw = optimally_smoothed_track[:, 14]



            # now produce local path and global path data
            s_4_local_path, x_4_local_path, s_vals_global_path, x_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_x_raw, Ds_back, Ds_forward)
            s_4_local_path, y_4_local_path, s_vals_global_path, y_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_y_raw, Ds_back, Ds_forward)
            s_4_local_path, z_4_local_path, s_vals_global_path, z_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_z_raw, Ds_back, Ds_forward)

            # same for the derivatives
            _, dx_ds, s_vals_global_path, dx_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dx_ds_raw, Ds_back, Ds_forward)
            _, dy_ds, s_vals_global_path, dy_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dy_ds_raw, Ds_back, Ds_forward)
            _, dz_ds, s_vals_global_path, dz_ds_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dz_ds_raw, Ds_back, Ds_forward)

            # same for dev 2
            _, d2x_ds2, s_vals_global_path, d2x_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2x_ds2_raw, Ds_back, Ds_forward)
            _, d2y_ds2, s_vals_global_path, d2y_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2y_ds2_raw, Ds_back, Ds_forward)
            _, d2z_ds2, s_vals_global_path, d2z_ds2_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_d2z_ds2_raw, Ds_back, Ds_forward)

            # reference path frame
            _,roll_4_local_path, _, roll_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_roll_raw, Ds_back, Ds_forward)
            _,pitch_4_local_path, _, pitch_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_pitch_raw, Ds_back, Ds_forward)
            _,yaw_4_local_path, _, yaw_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_yaw_raw, Ds_back, Ds_forward)

            # reference path angular rates
            _,wz_4_local_path, _, _ = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_wz_raw, Ds_back, Ds_forward)
            _,wx_4_local_path, _, _ = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_wx_raw, Ds_back, Ds_forward)

            # just for the gates position
            dummy1, dummy2, dummy3,\
            dummy4, dummy5, dummy6,\
            dummy7, dummy8, dummy8,\
            dummy9, gates  = raw_track(track_choice, 100) 

            print('')
            print('✔ loaded optimally smoothed track data from file:')
            print(f'{path_2_stored_optimally_smoothed_track}')


        
        except FileNotFoundError:
            print('')
            print('Looking for smoothed track file at:')
            print(f'{path_2_stored_optimally_smoothed_track}')
            print(f"Optimally smoothed track file for {track_choice} not found. Using raw track data.")
            optimally_smoothed = False


    if optimally_smoothed == False:
        # # --- generate raw ---
        n_checkpoints = 100  # number of checkpoints to be used to define each spline of the track (just an initial guess, it will be overwritten by the track definition)
        Checkpoints_x_raw, Checkpoints_y_raw, Checkpoints_z_raw,\
        Checkpoints_dx_raw, Checkpoints_dy_raw, Checkpoints_dz_raw,\
        Checkpoints_ddx_raw, Checkpoints_ddy_raw, Checkpoints_ddz_raw,\
        Checkpoints_s_raw, gates  = raw_track(track_choice, n_checkpoints) 
        
        # now generate track data for the mpc
        s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
        dx_vals_global_path, dy_vals_global_path, dz_vals_global_path,\
        d2x_vals_global_path, d2y_vals_global_path, d2z_vals_global_path,\
        s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
        dx_4_local_path, dy_4_local_path, dz_4_local_path,\
        d2x_4_local_path, d2y_4_local_path, d2z_4_local_path = generate_track_3d(Checkpoints_x_raw,  Checkpoints_y_raw,  Checkpoints_z_raw, Checkpoints_s_raw,
                                                                                Checkpoints_dx_raw, Checkpoints_dy_raw, Checkpoints_dz_raw,
                                                                                Checkpoints_ddx_raw, Checkpoints_ddy_raw, Checkpoints_ddz_raw,
                                                                                Ds_back, Ds_forward)

        # LEGACY:  now differentiate to have the derivatives of the path
        #dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2 = produce_3d_labels_devs(s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path)
        
        # # now just take the analytically computed derivatives
        dx_ds = dx_4_local_path
        dy_ds = dy_4_local_path
        dz_ds = dz_4_local_path
        d2x_ds2 = d2x_4_local_path
        d2y_ds2 = d2y_4_local_path
        d2z_ds2 = d2z_4_local_path

        # dummy values for the reference path frame
        roll_4_local_path = np.zeros(len(s_4_local_path))
        pitch_4_local_path = np.zeros(len(s_4_local_path))
        yaw_4_local_path = np.zeros(len(s_4_local_path))
        roll_global_path = np.zeros(len(s_vals_global_path))
        pitch_global_path = np.zeros(len(s_vals_global_path))
        yaw_global_path = np.zeros(len(s_vals_global_path))
        wz_4_local_path = np.zeros(len(s_4_local_path))
        wx_4_local_path = np.zeros(len(s_4_local_path))


    # find the s value of the gates on the path starting from the x-y-z coordinates of the gates
    gates_s_global_path = find_s_gate_position(gates, s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path)

    # load offline-computed time optimal trajecotry to use as a warmstart for the mpc.
    # the time optimal trajectory has the following columns:
    # [roll_desired, pitch_desired, yaw_desired, thrust, x, y, z, vx, vy, vz, s, time]
    try:
        path_2_stored_time_optimal_trajectories = os.path.join(pkg_path, 'src', 'solvers_setup', 'offline_optimal_solutions', track_choice + '.npy')
        time_optimal_trajectory = np.load(path_2_stored_time_optimal_trajectories)
        # now append the additional Ds_forwards to make sure we don't run out of reference warmstart trajectory
        # find index where s = Ds_forwards
        index_Ds_forward = np.argmin(np.abs(time_optimal_trajectory[:, 13] - Ds_forward))
        # produce extra loop part by adding the last time and s value to the end of the trajectory
        extra_loop_part = time_optimal_trajectory[1:index_Ds_forward, :].copy()
        # add the last time and s value to the end of the trajectory
        extra_loop_part[:, 13] += time_optimal_trajectory[-1, 13]  # add the last s value
        extra_loop_part[:, 14] += time_optimal_trajectory[-1, 14]  # add the last time value
        optimal_lap_time = time_optimal_trajectory[-1, 14]  # store the optimal lap time
        # concatenate the extra loop part to the time optimal trajectory
        time_optimal_trajectory_4_warmstart = np.concatenate((time_optimal_trajectory, extra_loop_part), axis=0)


    except FileNotFoundError:
        print('')
        print('WARNING: ')
        print('Looking for time optimal trajectory file at:')
        print(f'{path_2_stored_time_optimal_trajectories}')
        print(f"Time optimal trajectory file for {track_choice} not found. Using empty array as placeholder.")
        time_optimal_trajectory_4_warmstart = []

    
    # --- evaluate minimum curvature radius ---
    # evaluate curvature as the norm of the second order devs
    k_vec = np.sqrt(d2x_ds2**2 + d2y_ds2**2 + d2z_ds2**2)
    R_vec = 1/k_vec

    # switching to normalized second devs and separate curvature value
    d2x_ds2 = d2x_ds2 / k_vec
    d2y_ds2 = d2y_ds2 / k_vec
    d2z_ds2 = d2z_ds2 / k_vec


    # print trakc data
    print('-------------------------------------------------')
    print('TRACK DATA')
    print('Track choice: ', track_choice)
    print('Minimum curvature radius: ', np.min(R_vec))
    print('Track length: ', np.max(s_vals_global_path))
    if len(time_optimal_trajectory_4_warmstart)>0:
        print('Optimal lap time: ', optimal_lap_time)
    print('-------------------------------------------------')


    return  s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
            roll_global_path, pitch_global_path, yaw_global_path, \
            s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
            dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2, k_vec,\
            roll_4_local_path, pitch_4_local_path, yaw_4_local_path, \
            wz_4_local_path, wx_4_local_path, \
            gates, gates_s_global_path, time_optimal_trajectory_4_warmstart


def raw_track(track_choice, n_checkpoints):
    
    # number of checkpoints to be used to define each spline of the track

    # cartesian coordinates (x-y-z) of the points defining the track and the gates
    Checkpoints_x_raw, Checkpoints_y_raw, Checkpoints_z_raw,\
    Checkpoints_dx_raw, Checkpoints_dy_raw, Checkpoints_dz_raw,\
    Checkpoints_ddx_raw, Checkpoints_ddy_raw, Checkpoints_ddz_raw,\
    gates, Checkpoints_s_raw = produce_track(track_choice, n_checkpoints)

    # # from the x-y-z points obtain the sum of the arc lenghts between each point, i.e. the path seen as a function of s
    # spline_discretization = len(Checkpoints_x_raw)
    # Checkpoints_s_raw = np.zeros(spline_discretization)
    # for i in range(1,spline_discretization):
    #     Checkpoints_s_raw[i] = Checkpoints_s_raw[i-1] + np.sqrt((Checkpoints_x_raw[i]-Checkpoints_x_raw[i-1])**2+
    #                                                     (Checkpoints_y_raw[i]-Checkpoints_y_raw[i-1])**2+
    #                                                     (Checkpoints_z_raw[i]-Checkpoints_z_raw[i-1])**2)
        
    return  Checkpoints_x_raw, Checkpoints_y_raw, Checkpoints_z_raw,\
            Checkpoints_dx_raw, Checkpoints_dy_raw, Checkpoints_dz_raw,\
            Checkpoints_ddx_raw, Checkpoints_ddy_raw, Checkpoints_ddz_raw,\
            Checkpoints_s_raw, gates




def generate_track_3d(   Checkpoints_x_raw,  Checkpoints_y_raw,  Checkpoints_z_raw, Checkpoints_s_raw,
                        Checkpoints_dx_raw, Checkpoints_dy_raw, Checkpoints_dz_raw,
                        Checkpoints_ddx_raw, Checkpoints_ddy_raw, Checkpoints_ddz_raw,
                        Ds_back, Ds_forward):

    # this function takes the raw checkpoints and generates the global path and local path for the MPC controller
    # the values for local path generation basically always ensure that the prediction horizon is covered, by adding
    # a little bit of the path backwards and a little bit forwards

    s_4_local_path, x_4_local_path, s_vals_global_path, x_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_x_raw, Ds_back, Ds_forward)
    _, y_4_local_path, _, y_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_y_raw, Ds_back, Ds_forward)
    _, z_4_local_path, _, z_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_z_raw, Ds_back, Ds_forward)

    _, dx_4_local_path, _, dx_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dx_raw, Ds_back, Ds_forward)
    _, dy_4_local_path, _, dy_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dy_raw, Ds_back, Ds_forward)
    _, dz_4_local_path, _, dz_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_dz_raw, Ds_back, Ds_forward)

    _, d2x_4_local_path, _, d2x_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_ddx_raw, Ds_back, Ds_forward)
    _, d2y_4_local_path, _, d2y_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_ddy_raw, Ds_back, Ds_forward)
    _, d2z_4_local_path, _, d2z_vals_global_path = from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_ddz_raw, Ds_back, Ds_forward)

    return  s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path,\
            dx_vals_global_path, dy_vals_global_path, dz_vals_global_path,\
            d2x_vals_global_path, d2y_vals_global_path, d2z_vals_global_path,\
            s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path,\
            dx_4_local_path, dy_4_local_path, dz_4_local_path,\
            d2x_4_local_path, d2y_4_local_path, d2z_4_local_path




def from_global_to_4_local_path(Checkpoints_s_raw, Checkpoints_x_raw, Ds_back, Ds_forward):
    Checkpoints_x = np.concatenate((Checkpoints_x_raw[:-1],Checkpoints_x_raw, Checkpoints_x_raw[1:]))
    Checkpoints_s = np.concatenate((Checkpoints_s_raw[:-1] -Checkpoints_s_raw[-1],Checkpoints_s_raw, Checkpoints_s_raw[1:] + Checkpoints_s_raw[-1]))

    index_local_path_start = np.argmin(np.abs(Checkpoints_s + Ds_back))
    index_s0 = np.argmin(np.abs(Checkpoints_s))
    index_finish_lap = np.argmin(np.abs(Checkpoints_s - Checkpoints_s_raw[-1]))
    index_finish_local_path = np.argmin(np.abs(Checkpoints_s - (Ds_forward + Checkpoints_s_raw[-1])))

    s_4_local_path = np.array(Checkpoints_s[index_local_path_start:index_finish_local_path])
    x_4_local_path = np.array(Checkpoints_x[index_local_path_start:index_finish_local_path])

    s_vals_global_path = np.array(Checkpoints_s[index_s0:index_finish_lap])
    x_vals_global_path = np.array(Checkpoints_x[index_s0:index_finish_lap])

    return s_4_local_path, x_4_local_path, s_vals_global_path, x_vals_global_path




def produce_3d_labels_devs(s_4_local_path, x_4_local_path, y_4_local_path, z_4_local_path):
    """
    Compute first and second derivatives of x, y, z with respect to s,
    and normalize the first derivatives to unit tangent vectors in 3D.
    """
    # Create cubic splines for x(s), y(s), z(s)
    cs_scipy_x = CubicSpline(s_4_local_path, x_4_local_path)
    cs_scipy_y = CubicSpline(s_4_local_path, y_4_local_path)
    cs_scipy_z = CubicSpline(s_4_local_path, z_4_local_path)

    # First derivatives
    dx_ds = cs_scipy_x(s_4_local_path, 1)
    dy_ds = cs_scipy_y(s_4_local_path, 1)
    dz_ds = cs_scipy_z(s_4_local_path, 1)

    # Normalize first derivatives to get unit tangent vectors
    norm_dev = np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)
    dx_ds = dx_ds / norm_dev
    dy_ds = dy_ds / norm_dev
    dz_ds = dz_ds / norm_dev

    # Second derivatives
    d2x_ds2 = cs_scipy_x(s_4_local_path, 2)
    d2y_ds2 = cs_scipy_y(s_4_local_path, 2)
    d2z_ds2 = cs_scipy_z(s_4_local_path, 2)

    return dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2


def find_s_gate_position(gates, s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path):
    # Initialize the gates_s vector
    gates_s = np.zeros(len(gates)) 
    for i in range(gates.shape[0]):
        gate_i = gates[i,:3]
        # evaluate distances to the track
        distances = np.sqrt((x_vals_global_path - gate_i[0])**2 + (y_vals_global_path - gate_i[1])**2 + (z_vals_global_path - gate_i[2])**2)
        gate_index_i = np.argmin(np.abs(distances))  # find the index of the closest point on the path to the gate
        gates_s[i] = s_vals_global_path[gate_index_i]

    # # evauate the x-y-z coordinates according to every s value of the global path and check if they match with the x-y-z coordinates of the gates
    # for ii in range(len(s_vals_global_path)):
    #     # for every point of the path check every gate
    #     for jj in range(len(gates_xyz)):
    #         # if the coordinates match, store the s value of the path in the proper position of gates_s
    #         if (x_vals_global_path[ii] == gates_xyz[jj,0] and y_vals_global_path[ii] == gates_xyz[jj,1] and z_vals_global_path[ii] == gates_xyz[jj,2]):
    #             # store the s value of the path in the proper position of gates_s
    #             gates_s[jj] = s_vals_global_path[ii]

    return gates_s










import numpy as np
from scipy import interpolate




import numpy as np

def hermite_spline(t, p0, p1, m0, m1):
    """
    Cubic Hermite spline formula.
    t: scalar or array in [0, 1]
    p0, p1: endpoints (np.array of shape (3,))
    m0, m1: tangent vectors at endpoints (np.array of shape (3,))
    Returns: array of shape (len(t), 3)
    """
    t = np.asarray(t)  # allow array operations
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    # Broadcast to shape (len(t), 1) for correct broadcasting with (3,)
    h00 = h00[:, np.newaxis]
    h10 = h10[:, np.newaxis]
    h01 = h01[:, np.newaxis]
    h11 = h11[:, np.newaxis]

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def hermite_spline_first_derivative(t, p0, p1, m0, m1):
    t = np.asarray(t)
    h00p = 6*t**2 - 6*t
    h10p = 3*t**2 - 4*t + 1
    h01p = -6*t**2 + 6*t
    h11p = 3*t**2 - 2*t

    h00p = h00p[:, np.newaxis]
    h10p = h10p[:, np.newaxis]
    h01p = h01p[:, np.newaxis]
    h11p = h11p[:, np.newaxis]

    return h00p * p0 + h10p * m0 + h01p * p1 + h11p * m1


def hermite_spline_second_derivative(t, p0, p1, m0, m1):
    t = np.asarray(t)
    h00pp = 12*t - 6
    h10pp = 6*t - 4
    h01pp = -12*t + 6
    h11pp = 6*t - 2

    h00pp = h00pp[:, np.newaxis]
    h10pp = h10pp[:, np.newaxis]
    h01pp = h01pp[:, np.newaxis]
    h11pp = h11pp[:, np.newaxis]

    return h00pp * p0 + h10pp * m0 + h01pp * p1 + h11pp * m1

def evaluate_s_checkpoints(segment_points):
    # Compute cumulative distances between consecutive points
    deltas = np.diff(segment_points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    s_values_single_spline = np.concatenate(([0], np.cumsum(segment_lengths)))
    return s_values_single_spline


def spline_assembler(n_points, splines):
    """
    Assemble multiple Hermite spline segments into a continuous 3D spline path.

    Parameters:
    - n_points: int
        Number of sample points to generate per spline segment.
    - splines: list of lists
        Each element is [t_range, x_start, y_start, z_start, tangent_start].
        The spline segments are connected end-to-start in a closed loop.

    Returns:
    - Checkpoints_x, Checkpoints_y, Checkpoints_z: lists of float
        Concatenated x, y, z coordinates of the assembled spline path.
    """

    t_samples = np.linspace(0, 1, n_points)
    Checkpoints_x, Checkpoints_y, Checkpoints_z = [], [], []
    Checkpoints_dx, Checkpoints_dy, Checkpoints_dz = [], [], []
    Checkpoints_ddx, Checkpoints_ddy, Checkpoints_ddz = [], [], []
    Checkpoints_s = []

    for i, spline in enumerate(splines):
        _, x_start, y_start, z_start, tangent_start = spline

        # Get next spline's start point and tangent (looped)
        next_spline = splines[(i + 1) % len(splines)]
        x_end, y_end, z_end, tangent_end = next_spline[1], next_spline[2], next_spline[3], next_spline[4]

        p0 = np.array([x_start, y_start, z_start])
        p1 = np.array([x_end, y_end, z_end])
        m0 = np.array(tangent_start)
        m1 = np.array(tangent_end)

        segment_points = hermite_spline(t_samples, p0, p1, m0, m1)
        #first_dev = hermite_spline_first_derivative(t_samples, p0, p1, m0, m1)
        #second_dev = hermite_spline_second_derivative(t_samples, p0, p1, m0, m1)
        s_values_single_spline = evaluate_s_checkpoints(segment_points)
        # build derivative using cubic spline interpolation
        dx_ds_single_spline, dy_ds_single_spline, dz_ds_single_spline,\
        d2x_ds2_single_spline, d2y_ds2_single_spline, d2z_ds2_single_spline = produce_3d_labels_devs(s_values_single_spline,\
                                                                                segment_points[:, 0],\
                                                                                segment_points[:, 1],\
                                                                                segment_points[:, 2])

        if i == 0:
            Checkpoints_x.extend(segment_points[:, 0])
            Checkpoints_y.extend(segment_points[:, 1])
            Checkpoints_z.extend(segment_points[:, 2])

            Checkpoints_dx.extend(dx_ds_single_spline)
            Checkpoints_dy.extend(dy_ds_single_spline)
            Checkpoints_dz.extend(dz_ds_single_spline)

            Checkpoints_ddx.extend(d2x_ds2_single_spline)
            Checkpoints_ddy.extend(d2y_ds2_single_spline)
            Checkpoints_ddz.extend(d2z_ds2_single_spline)

            Checkpoints_s.extend(s_values_single_spline)
        else:
            Checkpoints_x.extend(segment_points[1:, 0])
            Checkpoints_y.extend(segment_points[1:, 1])
            Checkpoints_z.extend(segment_points[1:, 2])

            Checkpoints_dx.extend(dx_ds_single_spline[1:])
            Checkpoints_dy.extend(dy_ds_single_spline[1:])
            Checkpoints_dz.extend(dz_ds_single_spline[1:])

            Checkpoints_ddx.extend(d2x_ds2_single_spline[1:])
            Checkpoints_ddy.extend(d2y_ds2_single_spline[1:])
            Checkpoints_ddz.extend(d2z_ds2_single_spline[1:])

            Checkpoints_s.extend(Checkpoints_s[-1] + s_values_single_spline[1:])

    return  Checkpoints_x, Checkpoints_y, Checkpoints_z,\
            Checkpoints_dx, Checkpoints_dy, Checkpoints_dz,\
            Checkpoints_ddx, Checkpoints_ddy, Checkpoints_ddz,\
            Checkpoints_s






# This function finds the parameter s of the point on the curve r(s) which provides minimum distance between the path and the vehicle
def find_s_of_closest_point_on_global_path(x_y_z_state, s_vals_global_path, x_vals_global_path, y_vals_global_path, z_vals_global_path, previous_index, estimated_ds):
    
    # find the minimum distance among the arc lenghts of the path
    min_ds = np.min(np.diff(s_vals_global_path))

    # find a likely number of idx steps of s_vals_global_path that the vehicle might have travelled
    estimated_index_jumps = math.ceil(estimated_ds / min_ds)

    # define the minimum number of steps that the vehicle could have traveled, if it was moving
    minimum_index_jumps = math.ceil(0.1 / min_ds)

    # in case the vehicle is still, ensure a minimum search space to account for localization error
    if estimated_index_jumps < minimum_index_jumps:
        estimated_index_jumps = minimum_index_jumps

    # define the search space span
    Delta_indexes = estimated_index_jumps * 3

    # takes the previous index (where the vehicle was) and defines a search space around it
    start_i = previous_index - Delta_indexes
    finish_i = previous_index + Delta_indexes

    # check if start_i is negative and finish_i is bigger than the path, in which case the search space is wrapped around
    if start_i < 0:
        s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i]), axis=0)
        x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[: finish_i]), axis=0)
        y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[: finish_i]), axis=0)
        z_search_vector = np.concatenate((z_vals_global_path[start_i:], z_vals_global_path[: finish_i]), axis=0)

    elif finish_i > s_vals_global_path.size:
        s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
        x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
        y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
        z_search_vector = np.concatenate((z_vals_global_path[start_i:], z_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
    else:
        s_search_vector = s_vals_global_path[start_i: finish_i]
        x_search_vector = x_vals_global_path[start_i: finish_i]
        y_search_vector = y_vals_global_path[start_i: finish_i]
        z_search_vector = z_vals_global_path[start_i: finish_i]

    # initialize the vector of distances
    distances = np.zeros(s_search_vector.size)

    # evaluate the distance between the vehicle (x_y_z_state[0:3]) and each point of the search vector (i.e. of the path)
    for ii in range(0, s_search_vector.size):
        distances[ii] = math.dist([x_search_vector[ii], y_search_vector[ii], z_search_vector[ii]], x_y_z_state[0:3])

    # retrieve the index of the minimum distance element
    local_index = np.argmin(distances)

    # check if the found minimum is on the boarder (indicating that the real minimum is outside of the search vector)
    # this offers some protection against failing the local search but it doesn't fix all of the possible problems
    # for example if path loops back (like a bean shape)
    # then you can still get an error (If you have lane boundary information then you colud put a check on the actual value of the min)
    if local_index == 0 or local_index == s_search_vector.size-1:
        print('search vector was not long enough, doing search on full path')
        distances_2 = np.zeros(s_vals_global_path.size)
        for ii in range(0, s_vals_global_path.size):
            distances_2[ii] = math.dist([x_vals_global_path[ii], y_vals_global_path[ii], z_vals_global_path[ii]], x_y_z_state[0:3])
        index = np.argmin(distances_2)
    else:
        index = np.where(s_vals_global_path == s_search_vector[local_index])
        # extract an int from the "where" operand (needed)
        index = index[0]
        index = index[0]

    s = float(s_vals_global_path[index])

    return s, index



# # This function takes the actual s parameter of the path and provide the cheby coeff of the local path (i.e. the path of lenght [a,b])
# def evaluate_local_path_Chebyshev_coefficients_high_order_cheby(s, x_of_s, y_of_s, z_of_s, Ds_forecast, s_vals_global_path, loop_path, Cheby_data_points):
#     # obtain the s values related to the local path
#     s_subpath_cheby_for_xyz_data_generation, s_subpath_for_fitting_operation = produce_s_local_path(s, Ds_forecast, Cheby_data_points, s_vals_global_path, loop_path)

#     # from the x-y points of the global path take just the subset that is needed for the local path
#     x_data_points_fitting_Cheby = x_of_s(s_subpath_cheby_for_xyz_data_generation)
#     y_data_points_fitting_Cheby = y_of_s(s_subpath_cheby_for_xyz_data_generation)
#     z_data_points_fitting_Cheby = z_of_s(s_subpath_cheby_for_xyz_data_generation)


#     # smooth the local path
#     smoothing_factor = 0.2
#     x_of_s = interpolate.UnivariateSpline(s_subpath_for_fitting_operation, x_data_points_fitting_Cheby, s=smoothing_factor)
#     y_of_s = interpolate.UnivariateSpline(s_subpath_for_fitting_operation, y_data_points_fitting_Cheby, s=smoothing_factor)
#     z_of_s = interpolate.UnivariateSpline(s_subpath_for_fitting_operation, z_data_points_fitting_Cheby, s=smoothing_factor)

#     # take out the smoothed data points not as a fcn of s
#     x_smooth_local = x_of_s(s_subpath_for_fitting_operation)
#     y_smooth_local = y_of_s(s_subpath_for_fitting_operation)
#     z_smooth_local = z_of_s(s_subpath_for_fitting_operation)

#     # fit the data points with a chebychev polynomial (forces pro needs to recive a continuous path, no a piece wise, that's why we choose cheby polynomials)
#     # coeffx, coeffy, coeffz are the coefficients of the cheby polynomials dependent on the s of fitting, so here we associate the x,y,z, coordinates to the s_subpath_for_fitting_operation
#     coeffx = np.polynomial.chebyshev.chebfit(s_subpath_for_fitting_operation, x_smooth_local, 19)
#     coeffy = np.polynomial.chebyshev.chebfit(s_subpath_for_fitting_operation, y_smooth_local, 19)
#     coeffz = np.polynomial.chebyshev.chebfit(s_subpath_for_fitting_operation, z_smooth_local, 19)

#     # find the min s and max s of the local path
#     a = s_subpath_for_fitting_operation.min()
#     b = s_subpath_for_fitting_operation.max()
    
#     return coeffx, coeffy, coeffz, a, b



# select the s elements relative to the local path
# s_subpath_cheby_for_xyz_data_generation = succession of s values relative to the path (closed or capped)
# s_subpath_for_fitting_operation = forces pro needs increasing s values, so we use this vector to fit the cheby polynomials,
# the s values that exceed the max s are associated to the same x-y-z values of the first elements of the s vector (just for forcespro in that case)
def produce_s_local_path(s,Ds_forecast,Cheby_data_points,s_vals_global_path,loop_path):
    # allow for some track behind the vehicle
    Ds_backwards = 0.05 * Ds_forecast
                                                         #start value      stop value       #number of equally spaced points
    s_subpath_cheby_for_xyz_data_generation = np.linspace(s - Ds_backwards, s + Ds_forecast, Cheby_data_points)
    s_subpath_for_fitting_operation = np.linspace(s - Ds_backwards, s + Ds_forecast, Cheby_data_points)

    # if there is a loop path enabled, wrap around in case the subpath exceeds the path limits
    if loop_path:
        s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation > s_vals_global_path.max()] = s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation > s_vals_global_path.max()] - s_vals_global_path.max()
        s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation < s_vals_global_path.min()] = s_vals_global_path.max() + s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation < s_vals_global_path.min()] 
    
    # if there is no loop path enabled, cap the subpath to the path limits
    else:
        # cap upper value
        s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation > s_vals_global_path[-1]] = s_vals_global_path[-1]

        # cap lower value
        s_subpath_cheby_for_xyz_data_generation[s_subpath_cheby_for_xyz_data_generation < s_vals_global_path[0]] = s_vals_global_path[0]

        # set both s_vectors to be equal
        s_subpath_for_fitting_operation = s_subpath_cheby_for_xyz_data_generation
        
    return s_subpath_cheby_for_xyz_data_generation, s_subpath_for_fitting_operation