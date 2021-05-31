#!/usr/bin/env python3

import cv2
import cv_bridge
import math
import numpy as np
import rospy
import time

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped, Twist
from sensor_msgs.msg import Image, LaserScan
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

        # Subscribe to the image scan from the robot
        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_received)

        # Initialize a publisher for movement
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

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

        rospy.sleep(1)
        self.initialized = True

    def get_map(self, data):
        self.map = data

    def translate_map(self):
        if not self.initialized:
            return

    def image_received(self, data):
        if self.initialized:
            self.image_capture = data
            self.image_height = data.height
            self.image_width = data.width
            self.get_nav_line_moment()
        
    """
    save_next_img: Temporarily subscribes to the camera topic to get and store
    the next available image.
    """
    def save_next_img(self):
        self.capture_image = True
        while self.capture_image:
            time.sleep(1)
        
    """
    get_nav_line_moment: Get the center position of the blob of the yellow
    navigation line on the screen. Retuns None if there are no yellow pixels.
    """
    def get_nav_line_moment(self):
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
            k_p = 1.0 / 100.0
            twist = Twist()
            twist.linear.x = 0.2
            twist.angular.z = k_p * err
            # self.movement_pub.publish(twist)

            return cx, cy
        else:
            return None

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
        
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = DuckExpress()
    
    node.run()

