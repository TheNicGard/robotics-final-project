#!/usr/bin/env python3

import os
import rospy
import numpy as np
import math

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from std_srvs.srv import Empty, EmptyRequest

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from pathing import find_path_a_star

PATH_PREFIX = os.path.dirname(__file__) + "/robot_maps/"
MAP_NAME = "neighborhood_simple"

def manhattan_distance(x_node, y_node):
    return abs(x_node[0] - y_node[0]) + abs(x_node[1] - y_node[1])

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

class Node(object):
    def __init__(self, name, index, real_coords, map_coords):
        self.name = name
        self.occupancy_index = index
        self.real_coords = real_coords
        self.map_coords = map_coords

        # Neighbors to be initialized later
        self.n = None
        self.s = None
        self.e = None
        self.w = None

    def __repr__(self):
        return self.name + "| map pos: " + "(" + str(self.map_coords[0]) + ", " + str(self.map_coords[1]) + ") --> index: [" + str(self.occupancy_index) + "], (" + str(self.real_coords[0]) + ", " + str(self.real_coords[1]) + ")"

    def __str__(self):
        return '(' + self.name + ')'

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
        self.amcl_topic = "/amcl_pose"
        self.local_topic = "/global_localization"
        self.set_map_topic = "/set_map"

        # Node publisher for debugging
        rospy.Subscriber("/particlecloud", PoseArray, self.get_particles)

        # Particle filter
        rospy.Subscriber(self.amcl_topic, PoseWithCovarianceStamped, self.get_location)

        # Initialize location
        self.current_location = PoseWithCovarianceStamped()
        # rospy.wait_for_service(self.local_topic)
        # self.global_localization = rospy.ServiceProxy(self.local_topic, Empty)
        # localization_empty_msg = EmptyRequest()
        # self.global_localization(localization_empty_msg)

        # self.set_map = rospy.ServiceProxy(self.set_map_topic, Empty)
        # set_map_empty_msg = EmptyRequest()
        # self.set_map(set_map_empty_msg)


        # Initialize our map
        self.map = OccupancyGrid()

        # Subscribe to the map server
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.get_map)

        # Subscribe to the lidar scan from the robot
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)

        # enable listening for and broadcasting corodinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        # set threshold values for linear and angular movement before we preform an update
        self.lin_mvmt_threshold = 0.2        
        self.ang_mvmt_threshold = (np.pi / 6)

        self.odom_pose_last_motion_update = None

        # Align OccupancyGrid to map
        self.road_map = np.loadtxt(PATH_PREFIX + MAP_NAME + ".txt")
        rospy.sleep(1)
        self.node_map = {}
        self.current_node = None
        self.align_occupancy_grid()
        
        self.initialized = True

    def get_map(self, data):
        self.map = data

    def get_location(self, data):
        if not self.initialized:
            return

        self.current_location = data

        self.current_pos[0] = self.current_location.pose.pose.position.x
        self.current_pos[1] = self.current_location.pose.pose.position.y

    def get_particles(self, data):
        return

    def align_occupancy_grid(self):
        """ The purpose of align_occupancy_grid is to map the road map onto the occupancy 
            grid. Basically, we want to get the real points of our allowable nodes so that
            we can run A_Star on it """
        print(self.map.info)
        
        timer = 5
        decrement = 0
        candidates = []

        # First, we loop through the OccupancyGrid and stop once we've hit the "top left" corner
        # We use a timer and candidates, because the houses are not perfectly aligned in the map
        for i in range(0, self.map.info.width):
            if timer == 0:
                break
            else:
                timer -= decrement

            for j in range(0, self.map.info.height):
                value = self.map.data[i + j*self.map.info.width] 

                if (value > 10 and
                    self.map.data[i + (j-1)*self.map.info.width] > 10 and 
                    self.map.data[i + (j-2)*self.map.info.width] > 10 and 
                    self.map.data[i + (j-3)*self.map.info.width] > 10 and 
                    self.map.data[(i+1) + j*self.map.info.width] > 10 and
                    self.map.data[(i+2) + j*self.map.info.width] > 10 and
                    self.map.data[(i+3) + j*self.map.info.width] > 10):
                    decrement = 1 # start timer
                    candidates.append([i, j, i + j*self.map.info.width])

        # Get the corner point - largest i, smallest j
        # First get min_j
        max_i = 0
        max_j = 0
        for item in candidates:
            if item[1] > max_j:
                max_j = item[1]

        # Filter
        candidates = [item for item in candidates if item[1] == max_j]

        # Now get max_i
        for item in candidates:
            if item[0] > max_i:
                max_i = item[0]

        # Filter again
        candidates= [item for item in candidates if item[0] == max_i]

        # Should only be one item left - our origin/bottom left house
        if len(candidates) != 1:
            print("ERROR: align_occupancy grid found", len(candidates), "candidates")

        # Use the road_size to estimate where the road is (since the robot starts on the road and not on the house)
        candidate = candidates[0]
        road_size = 1
        origin_coords = (round((((candidate[0] * self.map.info.resolution) + self.map.info.origin.position.x) - road_size)),
                         round((((candidate[1] * self.map.info.resolution) + self.map.info.origin.position.y) + road_size)))

        node_size = 1.5
        for i in range(len(self.road_map)):
            for j in range(len(self.road_map[0])):
                elmt = self.road_map[i][j]

                if elmt == 0:
                    new_node_name = str((i, j))
                    real_coords = ((origin_coords[0] + (j * node_size), origin_coords[1] - (i * node_size)))
                    new_node = Node(new_node_name, 0, real_coords, (i, j))
                    self.node_map[new_node_name] = new_node

        print(self.road_map)
        print("")

        # Initialize neighbors
        for key, node in self.node_map.items():
            # print(self.node_map[key])
            coords = node.map_coords
            coords = (int(coords[0]), int(coords[1]))
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            positions = ['n', 's', 'e', 'w']
            for n, p in zip(neighbors, positions):
                neighbor = (coords[0] + n[0], coords[1] + n[1])
                
                # Check if in bounds
                if neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] >= len(self.road_map) or neighbor[1] >= len(self.road_map[0]):
                    continue

                # Set neighbor
                neighbor_name = str(neighbor)
                if neighbor_name in self.node_map:
                    if p == 'n':
                        node.n = self.node_map[neighbor_name]
                    if p == 's':
                        node.s = self.node_map[neighbor_name]
                    if p == 'e':
                        node.e = self.node_map[neighbor_name]
                    if p == 'w':
                        node.w = self.node_map[neighbor_name]

        # Set robot's location
        self.current_node = self.node_map['(0, 0)']
        self.current_pos = list(self.current_node.real_coords)

        print("Current node is:", repr(self.current_node))

    def robot_scan_received(self, data):
        """ Mostly taken from particle filter - just updates the robot's current position
            and estimates which node it is closest to. """
        # wait until initialization is complete
        if not(self.initialized):
            print("not init")
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
            print("not transform 1")
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated) 
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, data.header.stamp, rospy.Duration(0.5))
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
            print("not transform 2")
            return

        # calculate the pose of the laser distance sensor 
        p = PoseStamped(
            header=Header(stamp=rospy.Time(0),
                          frame_id=data.header.frame_id))

        self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # determine where the robot thinks it is based on its odometry
        p = PoseStamped(
            header=Header(stamp=data.header.stamp,
                          frame_id=self.base_frame),
            pose=Pose())

        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)

        # we need to be able to compare the current odom pose to the prior odom pose
        # if there isn't a prior odom pose, set the odom_pose variable to the current pose
        if not self.odom_pose_last_motion_update:
            self.odom_pose_last_motion_update = self.odom_pose
            return

        # check to see if we've moved far enough to perform an update

        curr_x = self.odom_pose.pose.position.x
        old_x = self.odom_pose_last_motion_update.pose.position.x
        curr_y = self.odom_pose.pose.position.y
        old_y = self.odom_pose_last_motion_update.pose.position.y
        curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
        old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)

        if (np.abs(curr_x - old_x) > self.lin_mvmt_threshold or 
            np.abs(curr_y - old_y) > self.lin_mvmt_threshold or
            np.abs(curr_yaw - old_yaw) > self.ang_mvmt_threshold):
            print("Current node is:", str(self.current_node))
            print("Location:", self.current_location)
            print("")

            # Now we must adjust the robot's position - distance is multipled by resolution
            distance_moved = math.sqrt(((curr_x - old_x) ** 2) + ((curr_y - old_y) ** 2))

            curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
            old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)

            #The following code is used to determine whether the robot is moving in reverse since distance is always positive
            x_pos = False
            if curr_x - old_x > 0:
                x_pos = True

            y_pos = False
            if curr_y - old_y > 0:
                y_pos = True


            moving_backwards = False
            working_yaw = curr_yaw % (2 * np.pi)

            #use the unit circle to determine if based on the sign of the x and y if we are moving forward or backwards
            #ex. if robot is facing between pi/2 and pi, it is moving backwards if x is positive or y is negative
            if working_yaw <= np.pi / 2:
                moving_backwards = not x_pos or not y_pos
            elif working_yaw <= np.pi:
                moving_backwards = x_pos or not y_pos
            elif working_yaw <= (3 * np.pi) / 2:
                moving_backwards = x_pos or y_pos
            elif working_yaw <= 2 * np.pi:
                moving_backwards = not x_pos or y_pos
            else:
                print("ERROR: yaw is outside of range")
                print("original yaw:", curr_yaw, "working_yaw:", working_yaw, end="\n\n")
            
            if moving_backwards:
                distance_moved *= -1

            # self.current_pos[0] += math.cos(working_yaw) * distance_moved
            # self.current_pos[1] += math.sin(working_yaw) * distance_moved

            # Now we see who the closest node is and update current node if necessary
            nodes = [self.current_node, self.current_node.n, self.current_node.s, self.current_node.e, self.current_node.w]
            
            new_node = None
            min_distance = 2 ** 32
            for node in nodes:
                # Skip if no neighbor
                if not node:
                    continue

                # Otherwise, pick closest node - usually will be self.current_node
                print("comparing", node.map_coords, node.real_coords, "to", self.current_pos)
                dist = manhattan_distance(node.real_coords, self.current_pos)
                if dist < min_distance:
                    min_distance = dist
                    new_node = node

            # Sanity check
            if new_node:
                self.current_node = new_node
            else:
                print("ERROR: robot_scan_received; new_node is null")

            # Finally, update the odom info
            self.odom_pose_last_motion_update = self.odom_pose


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckExpress()

    node.run()

