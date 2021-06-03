#!/usr/bin/env python3

import cv2
import cv_bridge
import math
import moveit_commander
import numpy as np
import os
import rospy
import time

from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist
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

"""
The Node object serves as a point of reference for the robot. Nodes are spaced approximately
3 meters apart from one another and are present anywhere the robot can drive unobstructed
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

        # Initialize lidar sub
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)

        # Initialize a publisher for movement
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

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

        self.current_path = find_path_a_star(np.flip(self.road_map, 0), self.current_node.map_coords, (6, 4))
        print("current_path:", self.current_path)

        # State for grabbing dumbbell
        self.turning_towards_dumbbell = True
        self.driving_towards_dumbbell = True
        self.picking_up_dumbbell = False

        # the farthest and closest distance a robot can be to pick up an object
        self.db_prox_high = 0.23
        self.db_prox_low = 0.18

        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.linear = 0
        self.last_cx = None

        self.initialized = True

    def get_map(self, data):
        self.map = data

    def translate_map(self):
        if not self.initialized:
            return

    def image_received(self, data):
        if self.initialized:
            # Take the ROS message with the image and turn it into a format cv2 can use
            self.image_capture = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
            self.image_height = data.height
            self.image_width = data.width
            # self.get_nav_line_moment()
            self.grab_dumbbell("green")

    def robot_scan_received(self, data):
        if self.initialized:
            self.update_robot_pos(data)

            if self.driving_towards_dumbbell:
                front_scan = data.ranges[0]

                if not self.can_see_dumbbell or front_scan == math.inf:
                    self.linear = 0
                elif front_scan > self.db_prox_high:
                    self.linear = 0.1
                elif front_scan < self.db_prox_low:
                    self.linear = -0.1
                else:
                    self.linear = 0
                    self.driving_towards_dumbbell = False
                    self.move_to_grabbed()
        
    """
    get_nav_line_moment: Get the center position of the blob of the yellow
    navigation line on the screen. Retuns None if there are no yellow pixels.
    """
    def get_nav_line_moment(self):
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
            # cv2.circle(img, (cx, cy), 20, (0, 0, 0), -1)
            # cv2.circle(img, (int(self.image_width / 2), int(self.image_height / 2)), 10, (255, 0, 0), -1)
            # cv2.imshow("window", img)
            # cv2.waitKey(3)

            err = (self.image_width / 2) - cx
            k_p = 1.0 / 500.0
            twist = Twist()
            twist.linear.x = 0.26
            twist.angular.z = k_p * err
            self.movement_pub.publish(twist)

            return cx, cy
        else:
            return None

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
    align_occupancy_grid: maps the loaded OccupancyGrid map onto the loaded road map
    to create a node map. There is one node for each '0' item in the road map.
    """
    def align_occupancy_grid(self):
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

    """
    update_robot_pos: Uses odometry to update the robot's estimated current positions
    Most of this code is from the particle filter project
    """
    def update_robot_pos(self, data):
        # we need to be able to transfrom the laser frame to the base frame
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated) 
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, data.header.stamp, rospy.Duration(0.5))
        if not(self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp)):
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

            self.current_pos[0] += math.cos(working_yaw) * distance_moved
            self.current_pos[1] += math.sin(working_yaw) * distance_moved

            # Update the robot's node
            self.estimate_node()
            # self.move_to_node()

            # Finally, update the odom info
            self.odom_pose_last_motion_update = self.odom_pose

    """
    estimate_node: Uses robot's current estimated position to check what the closest 
    node is. This node becomes the new current node.
    """
    def estimate_node(self):
        # See who the closest node is and update current node if necessary
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

    """
    move_to_node: Determines if the robot is sufficiently close to the next node in 
    its path. If so, it decides whether it should turn left, right, or continue
    forwards and does so. Otherwise does nothing.
    """
    def move_to_node(self):
        threshold = 0.2
        return

    """
    grab_dumbbell: finds the moment of a dumbbell determined by the color
    string. Assumes that the dumbbell is visible (i.e. it's not in front of a
    block of matching color).
    """
    def grab_dumbbell(self, color):
        
        img = self.image_capture

        # color_recog_fov is the percentage of the image that is used to find
        # the colored object
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
            self.can_see_dumbbell = True
            
            # Center of the colored pixels in the image (the moment)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            print("midpoint:", str(w // 2), "cx:", cx)
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
            self.can_see_dumbbell = False

        # Turn and drive towards dumbbell. "linear" is decided by lidar.
        if self.turning_towards_dumbbell and self.last_cx != None:
            twist = Twist()
            if self.last_cx > (w / 2) + 5:
                twist.angular.z = ang_vel
            elif self.last_cx < (w / 2) - 5:
                twist.angular.z = ang_vel

            twist.linear.x = self.linear
            self.movement_pub.publish(twist)
            
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
        self.move_arm([0, -1.18, 0.225, 0.035])

        print("Dumbbell is grabbed!")
        rospy.sleep(3)

    def move_to_release(self):
        self.move_arm([0, -0.35, -0.15, 0.5])
        self.move_gripper([0.01, 0.01])

        print("Dumbbell has been released!")
        rospy.sleep(3)
        
    def run(self):
        self.move_to_ready()
        rospy.spin()

if __name__ == "__main__":
    node = DuckExpress()
    node.run()

