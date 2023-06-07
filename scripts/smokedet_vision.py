import argparse
import torch
import os
import platform
import sys
import math
import time
import statistics as stat
import queue
import yaml
import sys
#from helpers import path
from transformations import *

import cv2
import numpy as np


##################################################################
import rospy
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
from std_msgs.msg import Bool, String
from proxmsg.msg import PandaProx
from std_srvs.srv import SetBool 
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge   
import colorama
from colorama import Fore, Style
##################################################################

class SmokeDetectorDetector():
    def __init__(self):
        self.IMAGE_HEIGHT = None
        self.IMAGE_WIDTH = None
        self.colorTopic = "/basler/image_rect_color"
        self.depthTopic = None
        self.camSel = False
        self.enableDetection = False
        self.enableNode = '/basler/smokedetection/enable'
        rospy.Service(self.enableNode, SetBool, self.enableDetection_CB)
        self.initRosPy()
        rospy.spin()

    def colorCB(self, input):
        if self.enableDetection:
            self.img_basler = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
            print('kosamona')

            gray = cv2.cvtColor(self.img_basler, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 5)
            rows = gray.shape[0]
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                    param1=100, param2=30,
                                    minRadius=100, maxRadius=150)
            
            print(f"{Fore.RED}IMG size: {circles}")
            rot_basler = matrix(x = [0.,1.,0.,1.,0.,0.,0.,0.,-1.], shape=(3,3))
            #rot_basler = np.eye(3)
            #rot_basler = rot_z(math.pi/2)
            rot_basler_q = r2q(rot_basler)
            print(f"{Fore.MAGENTA} rotmat: {rot_basler} | Quaternion: {rot_basler_q}")
            self.SendTransform2tf(p = [0.251, 0.272, 1.34], q = rot_basler_q, parent_frame= 'vision_table_zero', child_frame="basler")
                
            if circles is not None:

                circles = np.uint16(np.around(circles))
                print(f"{Fore.LIGHTCYAN_EX}Circles: {circles}")
                for index, i in enumerate(circles[0, :]):
                    center = (i[0], i[1])
                    
                    print(center)
                    print(f"{Fore.CYAN}Barva: {self.img_basler[center]}")
                    meanbright = np.mean(self.img_basler[center])
                    print(f"{Fore.GREEN} meanbright: {meanbright}")
                    if meanbright >= 150:
                        #center
                        cv2.circle(self.img_basler, center, 1, (0, 255, 0), 3)
                        # circle outline
                        radius = i[2]
                        print(f"{Fore.LIGHTMAGENTA_EX} circle index: {index}")
                        cv2.circle(self.img_basler, center, radius, (0, 255, 0), 3)
                        worldpos = self.uv_to_XY(center[0], center[1], 1.34)
                        self.SendTransform2tf(p = worldpos, parent_frame="basler", child_frame=("smokedet_"+str(index)))
                    else:
                        #center
                        cv2.circle(self.img_basler, center, 1, (255, 0, 0), 3)
                        # circle outline
                        radius = i[2]
                        print(radius)
                        cv2.circle(self.img_basler, center, radius, (255, 0, 0), 3)
            
            self.out = CvBridge().cv2_to_imgmsg(self.img_basler, encoding = 'rgb8')
            self.pub_OUTPUT_IMG.publish(self.out)
    

    def initRosPy(self):
        """
        This function initialises all of the publishers and subsribers
        required by the program.

        Args
        ----
            color_topic(str) : ROS topic publishing color images, used by AlphaPose

            depth_topic(str) : ROS topic publishing depth images, used to determine 3D location of joints
        """
        rospy.init_node("basler", anonymous = True)

        self.pub_LOCATION = tf2_ros.TransformBroadcaster()
        self.transmsg = geometry_msgs.msg.TransformStamped()
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
        
        while True:
            if self.colorTopic != None:
                break
        
        self.sub_INPUT_IMG = rospy.Subscriber(name = self.colorTopic, data_class = Image, callback = self.colorCB)
        self.pub_OUTPUT_IMG = rospy.Publisher("/vision/smokedetector/detected", Image, queue_size=1)

    def SendTransform2tf(self, p:list=[0,0,0],q:list=[1,0,0,0], parent_frame:str= "panda_2/realsense",child_frame:str="Human_Pose"):
        """
        This functions publishes a point to a TF topic, the point can be seen in RViz

        Args
        ----
            p(list) : Translation vector [x, y, z]

            q(list) : Quaternion rotation vector [w, Rot_X, Rot_Y, Rot_Z]

            parent_frame(str) : The point with respect to which we specify the translation vector

            child_frame(str) : The name of the resulting TF point
        """
        self.transmsg.header.stamp = rospy.Time.now()
        self.transmsg.header.frame_id = parent_frame

        self.transmsg.child_frame_id = child_frame

        self.transmsg.transform.translation.x = p[0]
        self.transmsg.transform.translation.y = p[1]
        self.transmsg.transform.translation.z = p[2]

        self.transmsg.transform.rotation.w = q[0]
        self.transmsg.transform.rotation.x = q[1]
        self.transmsg.transform.rotation.y = q[2]
        self.transmsg.transform.rotation.z = q[3]

        self.pub_LOCATION.sendTransform(self.transmsg)

    def uv_to_XY(self, u:int,v:int, z:int) -> list:
        """
        Convert pixel coordinated (u,v) from realsense camera
        into real world coordinates X,Y,Z 

        Args
        ----
            u(int) : Horizontal coordinate

            v(int) : Vertical coordinate

            z(int) : Depth coordinate

        Returns
        -------
            worldPos(list) : Real world position (in respect to camera)
        """
        
        x = (u - (729.8347850687095)) / 2980.820556640625
        y = (v - (714.869506082948)) / 3036.631219185893

        #x = (u - (322.41944)) / 625.11944
        #y = (v - (243.40506)) / 623.68514

        X = (z * x)
        Y = (z * y)
        
        Z = z

        worldPos = [X, Y, Z]
        return worldPos
        
    def create_service_client(self, node):
        try:
            print("waiting for service:" + node)
            rospy.wait_for_service(node, 2) # 2 seconds
        except rospy.ROSException as e:
            print("Couldn't find service! " + node)
        self.camera_service = rospy.ServiceProxy(node, SetBool)

    def enableDetection_CB(self, req):
        state = req.data
        if state:
            print("Smoke detector detection: starting...")
            self.enableDetection = True
            msg = self.enableNode + " started."
        else:
            print("Smoke detector detection: stopping...")
            self.enableDetection = False
            msg = self.enableNode + " stopped."
        return True, msg
    
if __name__ == "__main__":
    SmokeDetectorDetector()