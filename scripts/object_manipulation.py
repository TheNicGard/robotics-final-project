#!/usr/bin/env python3

import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Point, Quaternion

class ObjectManipulation(object):
    def __init__(self):
        rospy.init_node('object_manipulation')
        
    # cite: https://answers.gazebosim.org//question/22125/how-to-set-a-models-position-using-gazeboset_model_state-service-in-python/
    def move_object(self, name, position):
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position = position
        state_msg.pose.orientation = Quaternion(0, 0, 0, 0)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__=="__main__":
    obj_man = ObjectManipulation()
    obj_man.move_object("A2", Point(0, 0, 3))
