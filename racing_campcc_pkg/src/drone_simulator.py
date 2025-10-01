#!/usr/bin/env python3

### this script simulates the output from the optitrack, so you can use it for tests ###

from solvers_setup.drone_dynamic_model import drone
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy import integrate
from tf.transformations import quaternion_from_euler
from dynamic_reconfigure.server import Server
from dynamic_reconfigure_pkg.cfg import Drone_Forwards_int_opti_message_dynamic_reconfigureConfig
from visualization_msgs.msg import MarkerArray, Marker



class Forward_intergrate_drone:
    def __init__(self, dt_int):
        print("Setting up drone simulator")

        # set up variables
        self.safety_value = 0
        self.state = np.zeros(9)          # the proper form of the initial state is specified in the main
        self.dt_int = dt_int                # timestep
        self.reset_state = False

        # set the drone initial input values
        self.des_roll = 0
        self.des_pitch = 0
        self.des_yaw = 0
        self.thrust = 0   


        # set up ros nodes and topics
        print("setting ros topics and node")

	    # node that recieves the safety value signal
        rospy.Subscriber('safety_value', Float32, self.callback_safety)


        # Drone Subscribers (inputs: des_roll, des_pitch, des_yaw, thrust)
        rospy.Subscriber('des_roll', Float32, self.callback_des_roll)
        rospy.Subscriber('des_pitch', Float32, self.callback_des_pitch)
        rospy.Subscriber('des_yaw', Float32, self.callback_des_yaw)
        rospy.Subscriber('thrust', Float32, self.callback_thrust)
        
        # this node has to publish info about the state of the system
        self.drone_state_publisher = rospy.Publisher('Drone_state', Float32MultiArray, queue_size=10)

        # publish for rviz the norm of the velocity
        self.v_norm_publisher = rospy.Publisher('v_norm', Float32, queue_size=10)

        # for rviz visualization of the vehicle as a moving frame
        self.rviz_state_publisher = rospy.Publisher('rviz_data', PoseStamped, queue_size=10)

        # for rviz visualization of the vehicle as a drone
        self.rviz_drone_visualization_publisher = rospy.Publisher('rviz_drone_visual' , MarkerArray, queue_size=10)

        # for dynamic parameter reconfiguration by means of rqt_reconfigure GUI
        srv = Server(Drone_Forwards_int_opti_message_dynamic_reconfigureConfig, self.reconfig_callback_simulator_GUI)



    ## GENERAL CALLBACKS ##
    # Safety callback function



    # Parameter reconfiguration 
    def reconfig_callback_simulator_GUI(self, config, level):
        print('reconfiguring parameters from dynamic_reconfig')
        #self.dt_int = config['dt_int'] # set the inegrating step
        #self.rate = rospy.Rate(1 / self.dt_int) # accordingly reset the output rate of a new measurement

        # define the reset configurations
        # common to both point mass and drone
        reset_state_x = config['reset_state_x']
        reset_state_y = config['reset_state_y']
        reset_state_z = config['reset_state_z']

        # drone specific
        reset_state_roll = config['reset_state_roll']
        reset_state_pitch = config['reset_state_pitch']
        reset_state_yaw = config['reset_state_yaw']


        reset_state = config['reset_state']

        if reset_state:
            print('Resetting state')
            self.state = [reset_state_x, reset_state_y, reset_state_z, 0.0, 0.0, 0.0, reset_state_roll, reset_state_pitch, reset_state_yaw]

        return config



    def callback_safety(self, safety_val_incoming):
        self.safety_value = safety_val_incoming.data
        if self.safety_value == 0:
            self.des_roll = 0.0
            self.des_pitch = 0.0
            self.des_yaw = 0.0
            self.thrust = 0.0

    ## DRONE CALLBACKS ##
    # des_roll callback function
    def callback_des_roll(self, des_roll):
        if self.safety_value == 1:
            self.des_roll = des_roll.data

    # des_pitch callback function
    def callback_des_pitch(self, des_pitch):
        if self.safety_value == 1:
            self.des_pitch = des_pitch.data
    
    # des_yaw callback function
    def callback_des_yaw(self, des_yaw):
        if self.safety_value == 1:
            self.des_yaw = des_yaw.data
    
    # thrust callback function
    def callback_thrust(self, thrust):
        if self.safety_value == 1:
            self.thrust = thrust.data


    ### FUNCTIONS ###
    def forwards_integrate_1_step(self):
        # perform forwards integration
        t0 = 0
        t_bound = self.dt_int

        # drone inputs added to the states
        import numpy as np
        import copy

        y0 = np.array([
            float(self.des_roll),
            float(self.des_pitch),
            float(self.des_yaw),
            float(self.thrust),
            *copy.deepcopy(self.state)
        ])



        # forwards integrate using RK4
        RK45_output = integrate.RK45(self.integrating_function, t0, y0, t_bound)
        while RK45_output.status == 'running':
            RK45_output.step()

        z_next = RK45_output.y

        # update the state (first four elements are the inputs)
        self.state = z_next[4:].tolist() 

                         
        # Create the Float32MultiArray message
        state_msg = Float32MultiArray()

        # Assuming self.state is a list or array of length 9: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        state_msg.data = list(self.state)

        # publish results of the integration (necessary for the MPC optimization)
        self.drone_state_publisher.publish(state_msg)
        




        # publish the norm of the velocity for rviz display
        v_norm = np.sqrt(self.state[3]**2 + self.state[4]**2 + self.state[5]**2 + 0.001)
        self.v_norm_publisher.publish(v_norm)

        # VISUALIZE THE VEHICLE AS A MOVING FRAME IN RVIZ
        rviz_message = PoseStamped()

        # position
        rviz_message.pose.position.x = self.state[0]
        rviz_message.pose.position.y = self.state[1]
        rviz_message.pose.position.z = self.state[2]

        # orientation
        quat = quaternion_from_euler(self.state[6], self.state[7], self.state[8])
        rviz_message.pose.orientation.x = quat[0]
        rviz_message.pose.orientation.y = quat[1]
        rviz_message.pose.orientation.z = quat[2]
        rviz_message.pose.orientation.w = quat[3]

        # frame data is necessary for rviz
        rviz_message.header.frame_id = 'map'

        # publish rviz message of the moving frame
        self.rviz_state_publisher.publish(rviz_message)
        

        # VISUALIZE THE DRONE IN RVIZ
        marker_array = MarkerArray()              # definition of an array of markers
        marker = Marker()                         # single marker within the marker_array

        marker.header.frame_id = "map"            # map frame, used when markers or data should be positioned in relation to a global map.
        marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

        marker.type = marker.MESH_RESOURCE
        marker.mesh_resource = "package://racing_campcc_pkg/src/Dae_models/quadrotor_base.dae"
        marker.id = 1

        # Set the scale of the marker
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        # Set the color
        rgba = [100.0, 100.0, 100.0, 1.0]   
        marker.color.r = rgba[0] / 256
        marker.color.g = rgba[1] / 256
        marker.color.b = rgba[2] / 256
        marker.color.a = rgba[3]

        # Set the position of the marker
        marker.pose.position.x = self.state[0]
        marker.pose.position.y = self.state[1]
        marker.pose.position.z = self.state[2]

        # Set the oritentation of the marker  
        quat = quaternion_from_euler(self.state[6], self.state[7], self.state[8])
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker_array.markers.append(marker)

        # publish rviz message of the drone
        self.rviz_drone_visualization_publisher.publish(marker_array)



    # definition of the function to be integrated by RK45 method
    def integrating_function(self, t, z):                   # RK4 wants a function that takes as input time and state
        # unpack the state vector
        roll_d = z[0]       # takes desired roll
        pitch_d = z[1]      # takes desired pitch
        yaw_d = z[2]        # takes desired yaw  NOW SET TO BE RELATIVE TO THE CURRENT YAW.
        thrust = z[3]       # takes thrust
        # x y z positions do not affect the dynamics, so no need to store them now
        # x = z[4]
        # y = z[5]
        # z = z[6]
        vx = z[7]           # takes velocity in x
        vy = z[8]           # takes velocity in y
        vz = z[9]           # takes velocity in z
        roll = z[10]         # takes roll state
        pitch = z[11]        # takes pitch state
        yaw = z[12]          # takes yaw state

        x_dot = drone(roll_d, pitch_d, yaw_d, thrust, vx,vy,vz, roll, pitch, yaw)
        # adding zeros to the derivatives of u since they are constant (in the timestep)
        zdot = np.array([0.0, 0.0, 0.0, 0.0, *x_dot])   #  we need to give the derivative of the full state (including the control inputs)

        return zdot



import time

if __name__ == '__main__':
    try:
        rospy.init_node('Forwards_Integrator_node', anonymous=True)
        dt_int = 0.01  # timestep in seconds

        drone_simulator = Forward_intergrate_drone(dt_int)
        rate = rospy.Rate(1 / dt_int)

        expected_duration = dt_int

        while not rospy.is_shutdown():
            start_time = time.time()

            drone_simulator.forwards_integrate_1_step()

            elapsed = time.time() - start_time
            if elapsed > expected_duration:
                rospy.logwarn(f"Simulator loop took {elapsed:.4f}s (expected {expected_duration:.4f}s) — falling behind!")

            # #TEMPORARY self.state[6], self.state[7], self.state[8]
            # import numpy as np
            # from tf.transformations import euler_matrix

            # # Extract roll, pitch, yaw from the state
            # roll = drone_simulator.state[6]
            # pitch = drone_simulator.state[7]
            # yaw = drone_simulator.state[8]

            # # Compute rotation matrix from body to world frame
            # rot_matrix = euler_matrix(roll, pitch, yaw, axes='sxyz')

            # # Body Z-axis in world frame (the 3rd column of the rotation matrix)
            # body_z_world = rot_matrix[:3, 2]  # shape (3,), normalized

            # # World vertical axis (assumed to be Z-up)
            # world_up = np.array([0, 0, 1])

            # # Compute angle between them
            # cos_theta = np.dot(body_z_world, world_up)
            # cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical issues
            # tilt_angle_rad = np.arccos(cos_theta)
            # tilt_angle_deg = np.degrees(tilt_angle_rad)

            #rospy.loginfo(f"Tilt angle from vertical: {tilt_angle_deg:.2f} degrees")

            #rospy.loginfo(np.round(np.array([drone_simulator.state[6],drone_simulator.state[7],drone_simulator.state[8]]),2))

            rate.sleep()

    except rospy.ROSInterruptException:
        pass


