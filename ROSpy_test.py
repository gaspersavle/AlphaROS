import cv2
from cv_bridge import CvBridge
import time
import rospy
from std_msgs.msg import String, Float32, Int16, Bool
from sensor_msgs.msg import Image, Imagefilter
from std_srvs.srv import Trigger
from cv_bridge import CvBridge   

from PIL import Image
        

class TestCamROSpy():
    def __init__(self):
        # Inicializacija nodea
        rospy.init_node("vision", anonymous = True)

        # Definicija odjemalca
        self.watcher = rospy.Subscriber("/realsense/color/image_raw", Image, self.transimg)

        # Definicija publisherja
        self.poser = rospy.Publisher("/realsense/color/image_pose", Image, queue_size = 1)

        rospy.spin()

    def transimg(self, input: Image) -> Image:
        image = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
        pil_image = Image.fromarray(image)
        pil_image.save("slika.jpg")
        self.receivedImg = pil_image
        self.editedImg = self.receivedImg.filter(Imagefilter.BLUR)
        
        self.poser.publish(self.editedImg)

    


if __name__ == "__main__":
    TestCamROSpy()