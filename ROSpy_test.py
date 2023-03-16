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
        self.poser = rospy.Publisher("/alphapose", Image, queue_size = 1)

        rospy.spin()

    def transimg(self, input: Image) -> Image:
        print(input)
        image = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
        pil_image = PILimage.fromarray(image)
        pil_image.save("slika.jpg")
        receivedImg = pil_image
        editedImg = receivedImg.filter(ImageFilter.BLUR)

        ros_edited_img = CvBridge().cv2_to_imgmsg(editedImg, encoding = 'rgb8')

        input.data = ros_edited_img
        
        self.poser.publish(input)

    


if __name__ == "__main__":
    TestCamROSpy()