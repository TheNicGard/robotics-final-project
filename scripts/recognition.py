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
        what percentage of the horizontal FOV the robot uses to determine the color
        in front of it
        """
        self.horizontal_field_percent = 0.25
        
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
    
    def search_view_for_color(self):
        self.save_next_img()

        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')

        masks = {}
        targets = {}
        for c in self.color_bounds.keys()
            masks[c] = cv2.inRange(img, self.color_bounds[c][0], self.color_bounds[c][1])
            targets[c] = cv2.bitwise_and(img, img, mask=masks[c])

        all_targets = targets["red"]
        for c in ["orange", "yellow", "green", "cyan", "blue", "magenta"]:
            all_targets = cv2.bitwise_or(targets[c], all_targets)

        cv2.imshow("image", all_targets)
        cv2.waitKey(0)

        return
        
        # check the center of the image for the most common color
        red_total, green_total, blue_total = 0, 0, 0
        img_center_w = int(img.shape[1] / 2)
        img_start_x = int(img_center_w - ((img.shape[1] * self.horizontal_field_percent) / 2))
        img_end_x = int(img_center_w + ((img.shape[1] * self.horizontal_field_percent) / 2))
        
        for i in range(img.shape[0]):
            for j in range(img_start_x, img_end_x):
                if not np.array_equal(blue_target[i, j], [0, 0, 0]):
                    blue_total += 1
                if not np.array_equal(green_target[i, j], [0, 0, 0]):
                    green_total += 1
                if not np.array_equal(red_target[i, j], [0, 0, 0]):
                    red_total += 1

if __name__=="__main__":
    rec = Recognition()
    rec.search_view_for_color()
