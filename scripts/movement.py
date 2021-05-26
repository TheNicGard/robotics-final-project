#!/usr/bin/env python3

import os
import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from pathing import find_path_a_star

PATH_PREFIX = os.path.dirname(__file__) + "/robot_maps/"
MAP_NAME = "neighborhood_simple"

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

        # Align OccupancyGrid to map
        self.road_map = np.loadtxt(PATH_PREFIX + MAP_NAME + ".txt")
        rospy.sleep(1)
        self.origin_index = None
        self.origin_coords = None
        self.align_occupancy_grid()
        
        self.initialized = True

    def get_map(self, data):
        self.map = data

    def align_occupancy_grid(self):
        for i in range(self.map.info.height):
            for j in range(self.map.info.width):
                if (self.map.data[i - (j+1)*self.map.info.width] > 10 and 
                    self.map.data[i - (j+2)*self.map.info.width] > 10 and 
                    self.map.data[i - (j+3)*self.map.info.width] > 10 and 
                    self.map.data[(i+1) - j*self.map.info.width] > 10 and
                    self.map.data[(i+2) - j*self.map.info.width] > 10 and
                    self.map.data[(i+3) - j*self.map.info.width] > 10):
                    self.origin_index = ((i-17) - (j-17)*self.map.info.width)
                    self.origin_coords = (i-17, j-17)
                    break
            if self.origin_index != None:
                break

        print("INDEX:", self.origin_index)
        print("COORDS:", self.origin_coords)
        print("RANGE:", self.origin_coords[1], "-", self.origin_coords[1] + 25*len(self.road_map[0]))
        print(self.road_map)
        iter_index = self.origin_index
        iter_row = self.origin_coords[0]
        num_rows = 0
        for i in range(self.map.info.height):
            for j in range(self.map.info.width):
                value = self.map.data[i - j*self.map.info.width]
                
                if ((i == iter_row) and
                    (j >= self.origin_coords[1]) and
                    (j < (self.origin_coords[1] + 25*len(self.road_map[0]))) and
                    ((((i - j*self.map.info.width) - iter_index) % (25*self.map.info.width)) == 0)):
                    print("X", end="")
                elif value < 0:
                    print("+", end="")
                elif value < 10:
                    print(" ", end="")
                else:
                    print("\u2588", end="")

            print("")
            if i == iter_row:
                num_rows += 1
                if num_rows == len(self.road_map):
                    iter_row = 0
                    iter_index = 0
                else:
                    iter_row += 25
                    iter_index += 25



    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckExpress()

    node.run()

