#!/usr/bin/env python3

import cv2
import cv_bridge
import math
import moveit_commander
import numpy as np
import os
import rospy
import time
import json

from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Header, String

# import tf
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

"""
The Node object serves as a point of reference for the robot. Nodes are spaced approximately
1.5 meters apart from one another and are present anywhere the robot can drive unobstructed
on a road. There are no off-road nodes.
"""
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

        """
        AMCL Initialization
        """
        # Subscribe to particle filter
        rospy.Subscriber(self.amcl_topic, PoseWithCovarianceStamped, self.get_location)

        # Initialize location
        self.current_location = PoseWithCovarianceStamped()

        """
        Map initialization
        """
        # Initialize our map
        self.map = OccupancyGrid()

        # Subscribe to the map server
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.get_map)

        # Node publisher for debugging
        self.node_pub = rospy.Publisher("particle_cloud", PoseArray, queue_size=10)

        """
        Positioning initialization - odometry, lidar, and translations
        """
        # enable listening for and broadcasting corodinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        # set threshold values for linear and angular movement before we preform an update
        self.lin_mvmt_threshold = 0.2        
        self.ang_mvmt_threshold = (np.pi / 6)

        self.odom_pose_last_motion_update = None

        # Misc. movement variables
        self.linear = 0
        self.last_cx = None

        # Initialize lidar sub
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)

        # Initialize a publisher for movement
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize Twist messages
        self.road_msg = Twist()
        self.turn_msg = Twist()

        """
        Imaging initialization
        """
        # Subscribe to the image scan from the robot
        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_received)

        # Bridge with OpenCV
        self.bridge = cv_bridge.CvBridge()

        # Variable that tells the image call back whether to store the image
        self.capture_image = False

        """
        The upper and lower bounds for the BGR values for the color detection (i.e.
        what is considered blue, green, and red by the robot).
        """
        self.color_bounds = {
            "red": (np.array([0, 0, 100], dtype = "uint8"), np.array([75, 75, 255], dtype = "uint8")),
            "orange": (np.array([0, 80, 127], dtype = "uint8"), np.array([102, 153, 255], dtype = "uint8")),
            "yellow": (np.array([0, 210, 210], dtype = "uint8"), np.array([102, 255, 255], dtype = "uint8")),
            "green": (np.array([0, 100, 0], dtype = "uint8"), np.array([75, 255, 75], dtype = "uint8")),
            "cyan": (np.array([90, 90, 0], dtype = "uint8"), np.array([255, 255, 75], dtype = "uint8")),
            "blue": (np.array([64, 0, 0], dtype = "uint8"), np.array([255, 75, 75], dtype = "uint8")),
            "magenta": (np.array([100, 0, 100], dtype = "uint8"), np.array([255, 75, 255], dtype = "uint8")),
        }

        """
        What percentage of the FOV, in the center, that the robot uses to
        determine the color in front of it
        """
        self.horizontal_field_percent = 0.10
        self.vertical_field_percent = 0.10

        """
        What percentage of the vertical FOV, from the bottom, the robot searches
        for yellow pixels of the navigation line. Set to 1 to include the whole
        image.
        """
        self.nav_line_horizon = 1

        """
        Compass directions as strings. These are the only four accepted values.
        """
        self.directions = ['n', 'e', 's', 'w']

        """
        Map alignment
        """
        # Load road map
        self.road_map = np.loadtxt(PATH_PREFIX + MAP_NAME + ".txt")
        rospy.sleep(1)

        self.origin_index = None
        self.origin_coords = None
        self.node_map = {}
        self.current_node = None
        
        # Align the road map to the ocupancy grid to create the node map
        self.align_occupancy_grid()

        # Robot faces East to start
        self.current_dir = "e"

        # Load color maps
        with open(PATH_PREFIX + MAP_NAME + ".json", "r") as infile:
            color_maps = json.load(infile)
        self.active_color = "green"
        self.pickup_map = color_maps['pickup']
        self.dropoff_map = color_maps['dropoff']

        self.path = find_path_a_star(np.flip(self.road_map, 0), self.current_node.map_coords, (1, 2))
        print("current_path:", self.path)

        # Control booleans
        self.on_road = True
        self.ignore_road = False
        self.ignore_object = True
        self.turning_towards_dumbbell = False
        self.driving_towards_dumbbell = False
        self.picked_up_dumbbell = False
        self.can_see_dumbbell = False
        self.dropping_off_dumbbell = False

        # the farthest and closest distance a robot can be to pick up an object
        self.db_prox_high = 0.21
        self.db_prox_low = 0.16
        self.house_prox_high = 0.48
        self.house_prox_low = 0.43


        """
        MoveIt initialization
        """
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # move arm to ready position
        self.move_to_ready()

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
        if not self.initialized:
            return
        # Take the ROS message with the image and turn it into a format cv2 can use
        self.image_capture = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        self.image_height = data.height
        self.image_width = data.width
        
        if self.on_road and not self.ignore_road:
            self.get_nav_line_moment()

        if self.turning_towards_dumbbell:
            self.orient_to_dumbbell(self.active_color)

    def robot_scan_received(self, data):
        if not self.initialized:
            return 

        if self.driving_towards_dumbbell:
            front_scan = data.ranges[0]
            self.drive_to_dumbbell(front_scan)

        elif self.on_road:
            self.estimate_node()
            self.adjust_movement()

        elif self.dropping_off_dumbbell:
            self.drive_to_house(front_scan)

        else:
            self.return_to_road()

    """""""""""""""""""""""""""""""""
          ROBOT IMAGING FUNCTIONS
    """"""""""""""""""""""""""""""""" 

    """
    get_nav_line_moment: Get the center position of the blob of the yellow
    navigation line on the screen. Retuns None if there are no yellow pixels.
    """
    def get_nav_line_moment(self):
        if self.ignore_road:
            return

        img = self.image_capture
        
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
            twist = Twist()
            twist.linear.x = 0.26
            twist.angular.z = k_p * err
            if not self.ignore_road:
                self.movement_pub.publish(twist)

            return cx, cy
        else:
            return None

    """
    orient_to_dumbbell: finds the moment of a dumbbell determined by the color
    string. Assumes that the dumbbell is visible (i.e. it's not in front of a
    block of matching color).
    """
    def orient_to_dumbbell(self, color):
        if self.on_road:
            return

        img = self.image_capture

        # color_recog_fov is the percentage of the image that is used to find
        # the colored object - put "blinders" on if going for dumbbell
        if self.picked_up_dumbbell:
            self.color_recog_fov = 1.0
        else:
            self.color_recog_fov = 0.5

        # Reduce the horizontal FOV (to avoid looking at adjacent objects of
        # the same color
        w, h = self.image_width, self.image_height
        cv2.rectangle(img, (-w, 0), (int(w * self.color_recog_fov) // 2, h), (0, 0, 0), -1)
        cv2.rectangle(img, (w - (int(w * self.color_recog_fov) // 2), 0), (2 * w, h), 0, -1)

        # Get the colored pixels in the image
        color_mask = cv2.inRange(img, self.color_bounds[color][0], self.color_bounds[color][1])        
        color_target = cv2.bitwise_and(img, img, mask=color_mask)

        # Get the moment of the dumbbell
        gray_img = cv2.cvtColor(color_target, cv2.COLOR_BGR2GRAY)

        # The default angular velocity. Is zero when an object of a given color
        # can't be found on screen
        ang_vel = 0
        
        M = cv2.moments(gray_img)
        if M['m00'] > 0:
            if not self.picked_up_dumbbell:
                self.can_see_dumbbell = True
                self.driving_towards_dumbbell = True
            
            # Center of the colored pixels in the image (the moment)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # print("midpoint:", str(w // 2), "cx:", cx)
            # Uncomment the following code to see the moment in a window!
            cv2.line(color_target, (cx, 0), (cx, self.image_height), (128, 128, 128), 3)
            cv2.putText(color_target, "(" + str(cx) + ", " + str(cy) + ")", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128))
            cv2.imshow("window", color_target)
            cv2.waitKey(3)

            self.last_cx = cx
            err = (w / 2) - cx
            k_p = 1.0 / 500.0
            ang_vel = err * k_p
        else:
            if not self.picked_up_dumbbell:
                self.can_see_dumbbell = False

        # Turn and drive towards object. "linear" is decided by lidar.
        if (self.turning_towards_dumbbell or self.picked_up_dumbbell) and self.last_cx != None and not self.on_road and not self.ignore_object:
            twist = Twist()
            if self.last_cx > (w / 2) + 5:
                twist.angular.z = ang_vel
            elif self.last_cx < (w / 2) - 5:
                twist.angular.z = ang_vel

            twist.linear.x = self.linear
            if not self.ignore_object:
                self.movement_pub.publish(twist)

    """
    drive_to_house: When the robot has reached a dropoff node, it must offroad to the house's 
    dropoff zone. This function moves the robot there.
    """
    def drive_to_house(self, front_scan):
        print("Driving to house")
        if front_scan == math.inf:
            print("ERROR: drive_to_house: robot sees nothing in front")
            self.linear = 0
        elif front_scan > self.house_prox_high:
            self.linear = 0.1
        elif front_scan < self.house_prox_low:
            self.linear = -0.1
        else:
            self.ignore_object = True
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = 0
            self.movement_pub.publish(twist)
            self.picked_up_dumbbell = False
            self.move_to_released()

    """""""""""""""""""""""""""""""""
          ROBOT MOVEMENT FUNCTIONS
    """""""""""""""""""""""""""""""""
    
    """
    direction_to_turn: determines if the robot should turn left, right, or continue
    forward given current cardinal direction and destination cardinal direction.
    """
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

    """
    adjust_movement: if the robot has crossed into a "new" node, determines if the robot
    should continue forward or turn. 
    """
    def adjust_movement(self):
        next_coords = self.path[0]
        current_coords = self.current_node.map_coords

        # Check if we have entered next node in the path
        # print("Current:", current_coords, "Next:", next_coords)
        if next_coords == current_coords:
            if len(self.path) == 1:
                print("Reached destination!")
                self.road_msg.linear.x = 0
                self.road_msg.angular.z = 0
                self.movement_pub.publish(self.road_msg)
                self.ignore_road = True
                self.ignore_object = False
                self.on_road = False

                # Make one more 90 degree turn
                self.turn_msg.linear.x = 0

                if self.picked_up_dumbbell:
                    self.turn_msg.angular.z = -0.4
                    curr_idx = self.directions.index(self.current_dir) + 1
                
                    self.dropping_off_dumbbell = True
                else:
                    self.turn_msg.angular.z = 0.4
                    curr_idx = self.directions.index(self.current_dir) - 1

                    self.turning_towards_dumbbell = True

                self.current_dir = self.directions[(curr_idx % 4)]
                self.movement_pub.publish(self.turn_msg)

                rospy.sleep(4)

                self.turn_msg.linear.x = 0
                self.turn_msg.angular.z = 0
                self.movement_pub.publish(self.turn_msg)
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
                self.turn_msg.angular.z = -0.4
                curr_idx = self.directions.index(self.current_dir) + 1

            elif direction == "left":
                print("Turning left")
                self.turn_msg.angular.z = 0.4
                curr_idx = self.directions.index(self.current_dir) - 1
            
            self.turn_msg.linear.x = 0
            self.current_dir = self.directions[(curr_idx % 4)]

            print("Publishing")
            self.movement_pub.publish(self.turn_msg)
            rospy.sleep(4)
    
            self.turn_msg.linear.x = 0
            self.turn_msg.angular.z = 0
            self.movement_pub.publish(self.turn_msg)

            self.ignore_road = False
            print("Done")

    """
    estimate_node: Uses robot's current estimated position to check what the closest 
    node is. This node becomes the new current node.
    """
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
            # print(self.current_pos, "vs", node.real_coords, "=", dist)
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

    """
    drive_to_dumbbell: Once a dumbbell has been identified, this function drives the robot within 
    arm's reach of it
    """
    def drive_to_dumbbell(self, front_scan):
        print("Driving to dumbbell")
        if not self.can_see_dumbbell or front_scan == math.inf:
            self.linear = 0
        elif front_scan > self.db_prox_high:
            self.linear = 0.1
        elif front_scan < self.db_prox_low:
            self.linear = -0.1
        else:
            self.ignore_object = True
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = 0
            self.movement_pub.publish(twist)
            self.driving_towards_dumbbell = False
            self.picked_up_dumbbell = True
            self.move_to_grabbed()

   
    

    """
    return_to_road: Once a dumbbell has been picked or delivered, this function returns the robot
    to the road
    """
    def return_to_road(self):
        print("Returning to the road!")
        # Back up to the road
        self.road_msg.linear.x = -0.26
        self.road_msg.angular.z = 0.0

        self.movement_pub.publish(self.road_msg)
        rospy.sleep(3)


        self.road_msg.linear.x = 0.0
        self.road_msg.angular.z = 0.0

        self.movement_pub.publish(self.road_msg)

        # Choose a new destination
        if self.picked_up_dumbbell:
            dest_coords = self.dropoff_map[self.active_color]
        else:
            dest_coords = self.pickup_map[self.active_color]
            self.move_to_ready()

        # Recalculate path
        print("Current node:", self.current_node)
        print("Destination node:", dest_coords)
        self.path = find_path_a_star(self.road_map, self.current_node.map_coords, tuple(dest_coords))
        print("New path:", self.path)
        self.ignore_road = False
        self.on_road = True

            
    """""""""""""""""""""""""""""""""
          ROBOT ARM FUNCTIONS
    """""""""""""""""""""""""""""""""
    def move_arm(self, goal):
        self.move_group_arm.go(goal, wait=True)
        self.move_group_arm.stop()
        
    def move_gripper(self, goal):
        self.move_group_gripper.go(goal, wait=True)
        self.move_group_gripper.stop()
                
    def move_to_ready(self):
        self.move_arm([0.0, 0.55, 0.3, -0.85])
        self.move_gripper([0.018, 0.018])

        print("Arm ready to grab!")
        rospy.sleep(3)

    def move_to_grabbed(self):
        self.move_gripper([0.008, -0.008])     
        self.move_arm([0, 0.45, -0.75, -0.3])
        rospy.sleep(1)

        self.move_arm([0, -1.18, 0.225, -0.3])

        print("Dumbbell is grabbed!")
        rospy.sleep(3)

    def move_to_release(self):
        self.move_arm([0, -0.35, -0.15, 0.5])
        self.move_gripper([0.01, 0.01])

        print("Dumbbell has been released!")
        rospy.sleep(3)


    """""""""""""""""""""""""""""""""
             MISC FUNCTIONS
    """""""""""""""""""""""""""""""""

    """ 
    align_occupancy_grid: maps the loaded OccupancyGrid map onto the loaded road map
    to create a node map. There is one node for each '0' item in the road map.
    """
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

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = DuckExpress()
    node.run()

