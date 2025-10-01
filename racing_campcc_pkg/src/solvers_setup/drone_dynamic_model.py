import numpy as np

def drone(roll_d, pitch_d, yaw_d, thrust, vx,vy,vz, roll, pitch, yaw):
    # # z = des_roll des_pitch des_yaw thrust_controller x y z vx vy vz roll pitch yaw
    # roll_d = z[0]       # takes desired roll
    # pitch_d = z[1]      # takes desired pitch
    # yaw_d = z[2]        # takes desired yaw  NOW SET TO BE RELATIVE TO THE CURRENT YAW.
    # thrust = z[3]       # takes thrust


    # # x y z positions do not affect the dynamics, so no need to store them now
    # #x = z[4]
    # #y = z[5]
    # #z = z[6]
    # vx = z[7]           # takes velocity in x
    # vy = z[8]           # takes velocity in y
    # vz = z[9]           # takes velocity in z
    # roll = z[10]        # takes roll state
    # pitch = z[11]       # takes pitch state
    # yaw = z[12]         # takes yaw state

    # parameters
    m = 1.90            # mass of the drone [kg]
    g = 9.81            # gravity constant [m/s^2]
    k_drag = 0.3        # drag coefficient [kg/s]
    # moved to bottom
    # k_roll = 1.04       # roll coefficient 
    # k_pitch = 1.04      # pitch coefficient
    # k_yaw = 1           # yaw coefficient
    # tau_roll = 0.17     # roll time constant [s]
    # tau_pitch = 0.17    # pitch time constant [s]
    # tau_yaw = 0.564     # yaw time constant [s]

    T = m*g + thrust    # total thrust

    # Forces acting along the body axes
    Fx = T * ( np.sin(yaw) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) - k_drag * vx 
    Fy = T * (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(yaw) * np.sin(pitch)) - k_drag * vy
    Fz = T * (np.cos(roll) * np.cos(pitch)) - m*g - k_drag * vz 


    # overall accellerations
    acc_x = Fx / m
    acc_y = Fy / m
    acc_z = Fz / m

    # evaluate angular velocities
    wx, wy, wz = evaluate_w(roll_d, pitch_d, yaw_d, roll, pitch, yaw)

    # simplified drone nominal model
    xdot1 = vx
    xdot2 = vy
    xdot3 = vz
    xdot4 = acc_x
    xdot5 = acc_y
    xdot6 = acc_z
    xdot7 = wx #((k_roll * roll_d) - roll) / tau_roll
    xdot8 = wy #((k_pitch * pitch_d) - pitch) / tau_pitch
    xdot9 = wz # ((k_yaw * yaw_d) - yaw) / tau_yaw # track a given yaw angle
    #xdot9 = ((k_yaw * yaw_d) - 0) / tau_yaw # the yaw_d input is now defined as the difference between the desired yaw and the current yaw, so we can set the current yaw to 0 for simplicity
    
    # assemble derivatives of [x y z vx vy vz roll pitch yaw]
    xdot = [xdot1, xdot2, xdot3, xdot4 ,xdot5, xdot6, xdot7, xdot8, xdot9] # np.array()

    return xdot

def evaluate_w(roll_d,pitch_d,yaw_d,roll,pitch,yaw):
    k_roll = 1.04       # roll coefficient 
    k_pitch = 1.04      # pitch coefficient
    k_yaw = 1           # yaw coefficient
    tau_roll = 0.17     # roll time constant [s]
    tau_pitch = 0.17    # pitch time constant [s]
    tau_yaw = 0.564     # yaw time constant [s]

    wx = ((k_roll * roll_d) - roll) / tau_roll
    wy = ((k_pitch * pitch_d) - pitch) / tau_pitch
    wz = ((k_yaw * yaw_d) - yaw) / tau_yaw # track a given yaw angle
    return wx,wy,wz