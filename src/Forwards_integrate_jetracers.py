#!/usr/bin/env python3

# this script simulates the output from the optitrack, so you can use it for tests

import rospy
from std_msgs.msg import String, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



#set up global variables and relative callbacks
safety_value = 0
def safety_value_subscriber_callback(msg):
    # need global keyword to overwrite global variable
    global safety_value
    safety_value = msg.data

throttle_3 = 0
steering_3 = 0
def throttle_3_subscriber_callback(msg):
    # need global keyword to overwrite global variable
    global throttle_3
    throttle_3 = msg.data
def steering_3_subscriber_callback(msg):
    # need global keyword to overwrite global variable
    global steering_3
    steering_3 = msg.data




def continuous_dynamics_Jetracer(t, z):
    u = z[0:2]
    x = z[2:6]
    L  =0.175  # length of the jetracer [m]
    #numerical values taken from BEP paper, so linearized around th = 0.129 --> V=1 m/s
    c = 1.54/1.63  # longitudinal damping coefficient divided by the mass
    a_th = 60/1.63  # motor curve coefficient divided by the mass
    b_th = 6/1.63
    xdot1 = x[3] * np.cos(x[2])  # x_dot = v * cos(eta)
    xdot2 = x[3] * np.sin(x[2])  # y_dot = v * sin(eta)
    xdot3 = x[3] * np.tan(u[1]) / L  # v * tan(delta) / L
    if u[0] > 0.1:
        xdot4 = -c * x[3] + u[0] * a_th - b_th # vdot = -C*v+a_th*throttle+b_throttle  (linear motor torque eaten by linear viscosity)
    else:
        xdot4 = -c * x[3]

    #adding zeros to the derivatives of u since they are constant (in the timestep)
    zdot = np.array([0, 0, xdot1, xdot2, xdot3, xdot4])
    return zdot


def forwards_integrate():
    #set up subscriber nodes
    safety_value_subscriber = rospy.Subscriber('safety_value', Float32, safety_value_subscriber_callback)

    throttle_subscriber_3 = rospy.Subscriber('throttle_3', Float32, throttle_3_subscriber_callback)
    steering_subscriber_3 = rospy.Subscriber('steering_3', Float32, steering_3_subscriber_callback)

    #set up publisher handles
    test_publisher_1 = rospy.Publisher('Optitrack_data_topic_1', PoseStamped, queue_size=10)
    test_publisher_2 = rospy.Publisher('Optitrack_data_topic_2', PoseStamped, queue_size=10)
    test_publisher_3 = rospy.Publisher('Optitrack_data_topic_3', PoseStamped, queue_size=10)
    test_publisher_4 = rospy.Publisher('Optitrack_data_topic_4', PoseStamped, queue_size=10)

    test_velocity_publisher_1 = rospy.Publisher('velocity_1', Float32, queue_size=10)
    test_velocity_publisher_2 = rospy.Publisher('velocity_2', Float32, queue_size=10)
    test_velocity_publisher_3 = rospy.Publisher('velocity_3', Float32, queue_size=10)
    test_velocity_publisher_4 = rospy.Publisher('velocity_4', Float32, queue_size=10)


    # initialize current node and set publishing rate
    rospy.init_node('Optitrack_tests', anonymous=False)
    dt_int = 0.01
    rate = rospy.Rate(1/dt_int)

    # create empty messages
    message_1 = PoseStamped()
    message_2 = PoseStamped()
    message_3 = PoseStamped()
    message_4 = PoseStamped()

    # velocity_message_1 = Float32()
    # velocity_message_2 = Float32()
    # velocity_message_3 = Float32()
    # velocity_message_4 = Float32()
    plot_figures = True

    # initiate states
    state_3 = [0, 0, 0, 0]
    #this is needed to plot the figure in real time
    plt.ion()


    while not rospy.is_shutdown():
        #perform forwards integration
        t0 = 0
        t_bound = dt_int

        #only activates if safety is off

        delta = - steering_3 * (20.0 / 180.0 * np.pi) # must convert back to emulate what jetracer does
        y0 = np.array([throttle_3, delta] + state_3)


        #forwards integrate using RK4
        RK45_output = integrate.RK45(continuous_dynamics_Jetracer, t0, y0, t_bound)
        RK45_output.step()

        z_next = RK45_output.y
        state_3 = z_next[2:]
        state_3 = state_3.tolist()

        if plot_figures:
            L = 0.175
            rect1 = Rectangle((z_next[2]+(L/1.5)/2 * np.sin(z_next[4]), z_next[3]-(L/1.5)/2 * np.cos(z_next[4])), L, (L/1.5), angle=z_next[4]/np.pi*180, color='green')
            plt.gca().add_patch(rect1)
            plt.ylabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.legend()
            plt.title('Vehicle')
            plt.xlim(z_next[2]-2, z_next[2]+2)
            plt.ylim(z_next[3]-2, z_next[3]+2)
            plt.show()
            plt.pause(0.001)
            plt.clf()
            #####





        #fill in the messages and publish them
        message_1.header.stamp = rospy.Time.now()
        message_1.pose.position.x = 0.0
        message_1.pose.position.y = 0.8
        message_1.pose.orientation.y = 3.14 * (-1/8)

        message_2.header.stamp = rospy.Time.now()
        message_2.pose.position.x = -1.5
        message_2.pose.position.y = 0.8
        message_2.pose.orientation.y = 3.14 * 1

        message_3.header.stamp = rospy.Time.now()
        message_3.pose.position.x = state_3[0]
        message_3.pose.position.y = state_3[1]
        message_3.pose.orientation.y = state_3[2]

        message_4.header.stamp = rospy.Time.now()
        message_4.pose.position.x= 0.0
        message_4.pose.position.y= 0.8
        message_4.pose.orientation.y = 3.14 * (-1/8)

        #velocity
        velocity_message_1 = 1
        velocity_message_2 = 1
        velocity_message_3 = state_3[3]
        velocity_message_4 = 1


        #publish the messages
        test_publisher_1.publish(message_1)
        test_publisher_2.publish(message_2)
        test_publisher_3.publish(message_3)
        test_publisher_4.publish(message_4)

        test_velocity_publisher_1.publish(velocity_message_1)
        test_velocity_publisher_2.publish(velocity_message_2)
        test_velocity_publisher_3.publish(velocity_message_3)
        test_velocity_publisher_4.publish(velocity_message_4)

        rate.sleep()

if __name__ == '__main__':
    try:
        forwards_integrate()
    except rospy.ROSInterruptException:
        pass
