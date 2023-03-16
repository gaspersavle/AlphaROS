import cv2
from cv_bridge import CvBridge
import time
import rospy
from std_msgs.msg import String, Float32, Int16, Bool
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge   

from PIL import Image as PILimage
from PIL import ImageFilter
        

class TestCamROSpy():
    def __init__(self):
        # Inicializacija nodea
        rospy.init_node("vision", anonymous = True)

        # Definicija odjemalca
        self.watcher = rospy.Subscriber("/realsense/color/image_raw", Image, self.transimg)

        # Definicija publisherja
        self.poser = rospy.Publishe


    def transimg(self, input: Image) -> Image:
        print(input)
        image = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        ros_edited_img = CvBridge().cv2_to_imgmsg(image, encoding = 'rgb8')

        input.data = ros_edited_img
        self.poser.publish(input)

    


if __name__ == "__main__":
    TestCamROSpy()