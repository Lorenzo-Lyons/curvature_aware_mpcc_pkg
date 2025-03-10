#!/usr/bin/env python3
import numpy as np
import rospy


from curvature_aware_mpcc_pkg.msg import ThreeTimeStampsFloat32 

class republish_inputs:
    def __init__(self, car_number):
        rospy.init_node('3timestamp_republisher_' + str(car_number), anonymous=True)
        self.car_number = car_number
        rospy.Subscriber('throttle_car_' + str(car_number), ThreeTimeStampsFloat32, self.callback_throttle)
        rospy.Subscriber('steering_car_' + str(car_number), ThreeTimeStampsFloat32, self.callback_steering)

        # publishers for re-publishing the inputs with the car time stamp
        self.pub_throttle = rospy.Publisher('throttle_complete_stamp_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)
        self.pub_steering = rospy.Publisher('steering_complete_stamp_' + str(car_number), ThreeTimeStampsFloat32, queue_size=1)

    def callback_throttle(self, msg):
        # add the last time stamp to the message
        msg.header3.stamp = rospy.rospy.Time.now()
        #republish
        self.pub_throttle.publish(msg)
    
    def callback_steering(self, msg):
        # add the last time stamp to the message
        msg.header3.stamp = rospy.rospy.Time.now()
        #republish
        self.pub_steering.publish(msg)


if __name__ == '__main__':
    car_number = 1
    republish_inputs_obj = republish_inputs(car_number)
    
    rospy.spin()