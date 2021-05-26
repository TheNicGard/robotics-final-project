#!/usr/bin/env python3

import cv2
import cv_bridge
import math
import numpy as np
import rospy
import time

from sensor_msgs.msg import Image, LaserScan

class Recognition(object):
    def __init__(self):
        self.initialized = False

        # set up ROS / cv bridge
        rospy.init_node('recognition')
        
        self.bridge = cv_bridge.CvBridge()

        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        
        # variable that tells the image call back whether to store the image
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
        
        self.initialized = True

    def scan_callback(self, data):
        if self.initialized:
            return

    def image_callback(self, data):
        if self.initialized:
            if self.capture_image:
                print("Capturing image...")
                self.image_capture = data
                self.image_height = data.height
                self.image_width = data.width
                self.capture_image = False

    """
    save_next_img: Temporarily subscribes to the camera topic to get and store
    the next available image.
    """
    def save_next_img(self):
        self.capture_image = True
        while self.capture_image:
            time.sleep(1)

    """
    search_view_for_color: Returns the color in front of the robot.
    Adjust horizontal_field_percent and vertical_field_percent to adjust the
    amount of the FOV used in color detection.
    """
    def search_view_for_color(self):
        self.save_next_img()

        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')

        masks = {}
        targets = {}
        for c in self.color_bounds.keys():
            masks[c] = cv2.inRange(img, self.color_bounds[c][0], self.color_bounds[c][1])
            targets[c] = cv2.bitwise_and(img, img, mask=masks[c])

        all_targets = targets["red"]
        for c in ["orange", "green", "cyan", "blue", "magenta"]:
            all_targets = cv2.bitwise_or(targets[c], all_targets)

        # cv2.imshow("image", all_targets)
        # cv2.waitKey(0)
        # cv2.imwrite("./mask.png", all_targets)
        
        # check the center of the image for the most common color
        img_center_w = int(self.image_width / 2)
        img_start_x = int(img_center_w - ((self.image_width * self.horizontal_field_percent) / 2))
        img_end_x = int(img_center_w + ((self.image_width * self.horizontal_field_percent) / 2))

        img_center_h = int(self.image_height / 2)
        img_start_y = int(img_center_h - ((self.image_height * self.vertical_field_percent) / 2))
        img_end_y = int(img_center_h + ((self.image_height * self.vertical_field_percent) / 2))
        
        totals = {
            "red": 0,
            "orange": 0,
            "green": 0,
            "cyan": 0,
            "blue": 0,
            "magenta": 0
        }
                    
        for i in range(img_start_y, img_end_y):
            for j in range(img_start_x, img_end_x):
                for k in totals.keys():
                    if not np.array_equal(targets[k][i, j], [0, 0, 0]):
                        totals[k] += 1

        # guess the color in front of the robot        
        if max(totals.values()) == 0:
            print("Didn't detect a defined color!")
            return None
        for k in totals.keys():
            if max(totals.values()) == totals[k]:
                print("Detected:", k)
                return k

    """
    get_nav_line_moment: Get the center position of the blob of the yellow
    navigation line on the screen. Retuns None if there are no yellow pixels.
    """
    def get_nav_line_moment(self):
        self.save_next_img()

        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')

        # get the horizon mask
        # cite: https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        horizon_mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(horizon_mask, (0, self.image_height - int(self.nav_line_horizon * self.image_height)), (self.image_width, self.image_height), 255, -1)
        img = cv2.bitwise_and(img, img, mask=horizon_mask)

        # get the yellow pixels in the image
        yellow_mask = cv2.inRange(img, self.color_bounds["yellow"][0], self.color_bounds["yellow"][1])
        yellow_target = cv2.bitwise_and(img, img, mask=yellow_mask)

        # get the moment of the yellow lines
        gray_img = cv2.cvtColor(yellow_target, cv2.COLOR_BGR2GRAY)
        M = cv2.moments(gray_img)
        if M['m00'] > 0:
            # center of the yellow pixels in the image
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return cx, cy
        else:
            return None

if __name__=="__main__":
    rec = Recognition()
    rec.search_view_for_color()
    print(rec.get_nav_line_moment())
