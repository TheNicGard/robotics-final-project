#!/usr/bin/env python3

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from pathing import find_path_a_star


class DuckExpress(object):
    def __init__(self):
        # Will turn true after initialization
        self.initialized = False

        # Initialize duck express node
        rospy.init_node('duck_express')

        # Set the topic names and frame names
        self.base_frame = "base_footprint"
        self.map_topic = "map"
        self.odom_frame = "odom"
        self.scan_topic = "scan"

        # Initialize our map
        self.map = OccupancyGrid()

        # Subscribe to the map server
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.get_map)

        # Subscribe to the lidar scan from the robot
        # rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)

        rospy.sleep(1)
        self.initialized = True

    def get_map(self, data):
        self.map = data

        for i in range(self.map.info.width, 0, -1):
            counter = 0
            avg = 0
            for j in range(self.map.info.height):
                avg += self.map.data[i - j*self.map.info.width]
                counter += 1
                if counter == 1:
                    avg = int(avg / counter)
                    if avg < 0:
                        print("++", end="")
                    elif avg < 10:
                        print("  ", end="")
                    else:
                        print("\u2588\u2588", end="")
                    counter = 0
                    avg = 0

            print("")

    def translate_map(self):
        if not self.initialized:
            return

        


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckExpress()

    node.run()

