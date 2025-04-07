#!/usr/bin/env python3
import numpy as np
import rospy


from curvature_aware_mpcc_pkg.msg import ThreeTimeStampsFloat32 
from std_msgs.msg import Float32

class republish_inputs:
    def __init__(self, car_number):
        rospy.init_node('3timestamp_republisher_' + str(car_number), anonymous=True)
        self.car_number = car_number
        rospy.Subscriber('throttle_car_' + str(car_number), ThreeTimeStampsFloat32, self.callback_throttle)
        rospy.Subscriber('steering_car_' + str(car_number), ThreeTimeStampsFloat32, self.callback_steering)

        # publishers for re-publishing the inputs with the car time stamp
        self.pub_throttle = rospy.Publisher('throttle_complete_stamp_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)
        self.pub_steering = rospy.Publisher('steering_complete_stamp_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)

        self.comm_delay = 0
        past_delays = 1
        self.past_delays_th = np.zeros(past_delays)
        self.past_delays_st = np.zeros(past_delays)
        self.publish_comm_delay = rospy.Publisher('commdelay_laptop_2_car_' + str(car_number), Float32, queue_size=1)

        # publish the communication delay at 20 Hz
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # calculate the average communication delay
            avg_delay = np.mean([*self.past_delays_th, *self.past_delays_st])
            # create a message to publish the delays
            msg = Float32()
            msg.data = float(avg_delay)
            self.publish_comm_delay.publish(msg)
            rate.sleep()


    def callback_throttle(self, msg):
        # add the last time stamp to the message
        msg.header3.stamp = rospy.Time.now()
        #republish
        self.pub_throttle.publish(msg)

        # extrat time 1 from header 1
        time_1 = msg.header1.stamp.to_sec()
        time_3 = msg.header3.stamp.to_sec()
        delay_th = (time_3 - time_1) / 2
        #update past delays
        self.past_delays_th = [delay_th, *self.past_delays_th[:-1]]
        #print('throttle delay: ', self.past_delays_th)

    
    def callback_steering(self, msg):
        # add the last time stamp to the message
        msg.header3.stamp = rospy.Time.now()
        #republish
        self.pub_steering.publish(msg)
        # extrat time 1 from header 1
        time_1 = msg.header1.stamp.to_sec()
        time_3 = msg.header3.stamp.to_sec()
        delay_st = (time_3 - time_1) / 2
        #update past delays
        self.past_delays_st = [delay_st, *self.past_delays_st[:-1]]



if __name__ == '__main__':
    car_number = 1
    republish_inputs_obj = republish_inputs(car_number)
    
    #rospy.spin()