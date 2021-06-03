#!/usr/bin/env python3

import cv2
import cv_bridge
import math
import numpy as np
import rospy
import time
import os

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Header, String

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from pathing import find_path_a_star

PATH_PREFIX = os.path.dirname(__file__) + "/robot_maps/"
MAP_NAME = "neighborhood_simple"
DISTANCE_ALGORITHM = "euclidean"

def distance(x_node, y_node):
    if DISTANCE_ALGORITHM == "euclidean":
        return math.sqrt(((x_node[0] - y_node[0]) ** 2) + ((x_node[1] - y_node[1]) ** 2))
    elif DISTANCE_ALGORITHM == "manhattan":
        return abs(x_node[0] - y_node[0]) + abs(x_node[1] - y_node[1])
    else:
        raise Exception("distance: unkown algorithm:", DISTANCE_ALGORITHM)

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

    def __eq__(self, comparator):
        return self.map_coords == comparator.map_coords

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

        # Initialize a publisher for movement
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.road_msg = Twist()
        self.turn_msg = Twist()

        # Initialize location
        self.current_location = PoseWithCovarianceStamped()
        # rospy.wait_for_service(self.local_topic)
        # self.global_localization = rospy.ServiceProxy(self.local_topic, Empty)
        # localization_empty_msg = EmptyRequest()
        # self.global_localization(localization_empty_msg)

        # self.set_map = rospy.ServiceProxy(self.set_map_topic, Empty)
        # set_map_empty_msg = EmptyRequest()
        # self.set_map(set_map_empty_msg)

        """
        Imaging initialization
        """
        # Subscribe to the image scan from the robot
        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_received)

        # Bridge with OpenCV
        self.bridge = cv_bridge.CvBridge()

        # Variable that tells the image call back whether to store the image
        self.capture_image = False

        self.color_bounds = {
            "red": (np.array([0, 0, 100], dtype = "uint8"), np.array([75, 75, 255], dtype = "uint8")),
            "orange": (np.array([0, 80, 127], dtype = "uint8"), np.array([102, 153, 255], dtype = "uint8")),
            "yellow": (np.array([0, 210, 210], dtype = "uint8"), np.array([102, 255, 255], dtype = "uint8")),
            "green": (np.array([0, 100, 0], dtype = "uint8"), np.array([75, 255, 75], dtype = "uint8")),
            "cyan": (np.array([90, 90, 0], dtype = "uint8"), np.array([255, 255, 75], dtype = "uint8")),
            "blue": (np.array([64, 0, 0], dtype = "uint8"), np.array([255, 75, 75], dtype = "uint8")),
            "magenta": (np.array([100, 0, 100], dtype = "uint8"), np.array([255, 75, 255], dtype = "uint8")),
        }

        self.horizontal_field_percent = 0.10
        self.vertical_field_percent = 0.10

        self.nav_line_horizon = 1

        self.directions = ['n', 'e', 's', 'w']


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

        self.path = find_path_a_star(self.road_map, self.current_node.map_coords, self.node_map['(8, 8)'].map_coords)
        print("PATH:", self.path)
        self.current_dir = "e"

        # Control booleans
        self.ignore_road = False
        
        self.initialized = True

    def get_map(self, data):
        self.map = data

    def get_location(self, data):
        if not self.initialized:
            return

        self.current_location = data

        self.current_pos[0] = data.pose.pose.position.x
        self.current_pos[1] = data.pose.pose.position.y

    def image_received(self, data):
        if self.initialized:
            self.image_capture = data
            self.image_height = data.height
            self.image_width = data.width
            self.get_nav_line_moment()

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
        if not self.initialized:
            return

        # print("Current node is:", str(self.current_node))
        # print("Location:", self.current_location)
        # print("")
        
        self.estimate_node()
        self.adjust_movement()

    def estimate_node(self):
        # See who the closest node is and update current node if necessary
        nodes = [self.current_node.n, self.current_node.s, self.current_node.e, self.current_node.w]
        
        new_node = None
        min_distance = 0.3
        old_node = self.current_node
        for node in nodes:
            # Skip if no neighbor
            if not node:
                continue

            # Otherwise, pick closest node - usually will be self.current_node
            # print("comparing", node.map_coords, node.real_coords, "to", self.current_pos)
            dist = distance(node.real_coords, self.current_pos)
            print(self.current_pos, "vs", node.real_coords, "=", dist)
            if dist <= min_distance:
                min_distance = dist
                new_node = node

        # Sanity check
        if new_node:
            self.current_node = new_node
        else:
            # print("ERROR: robot_scan_received; new_node is null")
            self.current_node = old_node

        if old_node != self.current_node:
            print("Current node is:", str(self.current_node))

    def direction_to_turn(self, curr_dir: str, new_dir: str):

        try:
            curr_idx = self.directions.index(curr_dir)
        except ValueError:
            print("Direction \'" + curr_dir + "\' is not valid!")

        try:
            new_idx = self.directions.index(new_dir)
        except ValueError:
            print("Direction \'" + new_dir + "\' is not valid!")

        turn = (new_idx - curr_idx) % 4

        if turn == 0:
            return "forward"
        elif turn == 1:
            return "right"
        elif turn == 3:
            return "left"
        else:
            raise Exception("This turn shouldn't be valid!")

    def adjust_movement(self):
        next_coords = self.path[0]
        current_coords = self.current_node.map_coords

        # test= Twist()
        # test.linear.x = 0.5
        # test.angular.z = 0.1
        # self.movement_pub.publish(test)


        # Check if we have entered next node in the path
        # print("Current:", current_coords, "Next:", next_coords)
        if next_coords == current_coords:
            if len(self.path) == 1:
                print("Reached destination!")
                self.road_msg.linear.x = 0
                self.road_msg.angular.z = 0
                self.movement_pub.publish(self.road_msg)
                self.ignore_road = True
                return

            self.path.pop(0)
            print("Current node:", self.current_node)
            print("Path:", self.path)
            print("Next node:", self.path[0])
            print("")

            next_coords = self.path[0]

            # Determine which direction to go
            res = (current_coords[0] - next_coords[0], current_coords[1] - next_coords[1])

            new_cardinal = ''
            if res == (1, 0):
                new_cardinal = 'n'
            elif res == (-1, 0):
                new_cardinal = 's'
            elif res == (0, 1):
                new_cardinal = 'w'
            elif res == (0, -1):
                new_cardinal = 'e'
            else:
                raise Exception("adjust_movement: invalid res:", res)

            print("res is", res)
            print("new cardinal is", new_cardinal)

            direction = self.direction_to_turn(self.current_dir, new_cardinal)

            print("Moving", direction)

            if direction == "forward":
                print("Nothing (forward)")
                return

            self.ignore_road = True

            self.turn_msg.linear.x = 0
            self.turn_msg.angular.z = 0
            self.movement_pub.publish(self.turn_msg)

            if direction == "right":
                print("Turning right")
                self.turn_msg.linear.x = 0
                self.turn_msg.angular.z = -0.4
                curr_idx = self.directions.index(self.current_dir) + 1
                self.current_dir = self.directions[(curr_idx % 4)]

            elif direction == "left":
                print("Turning left")
                self.turn_msg.linear.x = 0
                self.turn_msg.angular.z = 0.4
                curr_idx = self.directions.index(self.current_dir) - 1
                self.current_dir = self.directions[(curr_idx % 4)]

            print("Publishing")
            self.movement_pub.publish(self.turn_msg)
            rospy.sleep(4)
    
            self.turn_msg.linear.x = 0
            self.turn_msg.angular.z = 0
            self.movement_pub.publish(self.turn_msg)

            self.ignore_road = False
            print("Done")

    def get_nav_line_moment(self):
        if self.ignore_road:
            return

        # Take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')

        # Get the horizon mask
        # Cite: https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        horizon_mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(horizon_mask, (0, self.image_height - int(self.nav_line_horizon * self.image_height)), (self.image_width, self.image_height), 255, -1)
        img = cv2.bitwise_and(img, img, mask=horizon_mask)

        # Get the yellow pixels in the image
        yellow_mask = cv2.inRange(img, self.color_bounds["yellow"][0], self.color_bounds["yellow"][1])
        yellow_target = cv2.bitwise_and(img, img, mask=yellow_mask)

        # Get the moment of the yellow lines
        gray_img = cv2.cvtColor(yellow_target, cv2.COLOR_BGR2GRAY)
        M = cv2.moments(gray_img)
        if M['m00'] > 0:
            # Center of the yellow pixels in the image (the moment)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Uncomment the following code to see the moment in a window!
            cv2.circle(img, (cx, cy), 20, (0, 0, 0), -1)
            cv2.circle(img, (int(self.image_width / 2), int(self.image_height / 2)), 10, (255, 0, 0), -1)
            cv2.imshow("window", img)
            cv2.waitKey(3)

            err = (self.image_width / 2) - cx
            k_p = 1.0 / 500.0
            # twist = Twist()
            self.road_msg.linear.x = 0.26
            self.road_msg.angular.z = k_p * err
            print("Updating road")
            if not self.ignore_road:
                self.movement_pub.publish(self.road_msg)

            return cx, cy
        else:
            return None
        

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckExpress()

    node.run()

