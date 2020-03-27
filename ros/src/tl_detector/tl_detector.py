#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import os
import sys
import math

# This calibration paramter debounces the light state
# received from the camera, such that toggeling between
# different states is avoided in case the tl_classifier
# is not sure
STATE_COUNT_THRESHOLD = 3
# This calibration paramter decides if images are saved
# to the linux-filesystem. This may sacrifice some computational
# power in favour of having the images for later analysis.
SAVE_CAMERA_IMAGES_IS_ACTIVE = True
# This calibration paramter decides if the traffic classifer
# light classifer is used or the state of the traffic light
# is taken from the simulator. Turn this to True only when
# using the code in the simulator!
USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR = False
# This calibration paramter renders the rate for
# proceesing images and detecting traffic lights
# It should be chosen by ansering the question how fast
# do images change and traffic lights disappear?
# Unit is Hz
TRAFFIC_LIGHT_DETECTION_UPDATE_FREQUENCY = 2
# This calibration parameter allwos to tune the threshold in meters for paying
# attention to the state of traffic light. Below that threshold, camea images
# are processed, above this is not done.
SAFE_DISTANCE_TO_TRAFFIC_LIGHT = 80


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()
        #self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.directory_for_images = '/data/'
        self.image_counter = 0
        self.loop() #rospy.spin()

    def loop(self):
        """
        This member function manages all threads inside the tl_detector node and
        makes the execution deterministic.
        """
        rate = rospy.Rate(TRAFFIC_LIGHT_DETECTION_UPDATE_FREQUENCY)
        while not rospy.is_shutdown():
            if not None in (self.waypoints, self.pose, self.camera_image):
                light_wp, state = self.process_traffic_lights()
                '''
                Publish upcoming red lights at camera frequency.
                Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''
                if self.state != state:
                    self.state_count = 0
                    self.state = state

                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))

                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))

                self.state_count += 1

            else:
                rospy.loginfo("tl_detector: Missing information, traffic light detection aborted.")

            rate.sleep()


    def pose_cb(self, msg):
        """
        This member function is called when pose is published in order to keep
        the current pose as a member variable.
        """
        self.pose = msg


    def waypoints_cb(self, waypoints):
        """
        This member function is called when waypoints is published in order to keep
        the waypoints as a member variable.
        """
        self.waypoints = waypoints
        number_of_waypoints = len(self.waypoints.waypoints)
        rospy.loginfo("tl_detector: Catched %d waypoints", number_of_waypoints)


    def traffic_cb(self, msg):
        """
        This member function is called when the state of the traffic lights are published in order to keep
        is as a member variable.
        """
        self.lights = msg.lights


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        rospy.loginfo("tl_detector: Catched an image.")
        self.has_image = True
        self.camera_image = msg
        if SAVE_CAMERA_IMAGES_IS_ACTIVE:
            self.save_image(msg)


    def save_image(self, img):
        """
        This member function catches images and saves them to disc.
        Arguments:
            img: The image from the simulator.
        """
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        img.encoding = "rgb8"
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
        file_name = curr_dir + self.directory_for_images+ 'img_'+'%06d'% self.image_counter +'.png'
        cv2.imwrite(file_name, cv_image)
        self.image_counter += 1
        rospy.loginfo("tl_detector.py: Camera image saved to %s!", file_name)


    def get_light_state(self, light):
        """
        This member function determines the current color of the traffic light.
        Arguments:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # Return the state of the light for testing
        if USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR:
            rospy.loginfo("tl_detector.py: Traffic light state taken from simulator!")
            return light.state

        else:
            rospy.loginfo("tl_detector.py: Traffic light classification not yet ready, publishing: %s", str(TrafficLight.UNKNOWN))
            TrafficLight.UNKNOWN
            #THIS IS WHERE THE CLASSIFIER GOES


        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False
        #
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #
        # #Get classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """
        This member function finds the closest visible traffic light, if one
        exists, and determines its location and color.
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None
        stop_line_position = None
        stop_line_waypoint_index = None
        distance = lambda a,b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        if self.pose is not None:
            vehicle_index = self.get_index_of_closest_waypoint_to_current_pose(self.pose.pose.position)
            vehicle_position = self.waypoints.waypoints[vehicle_index].pose.pose.position
            traffic_light_index = self.get_index_of_closest_traffic_light_to_current_pose(vehicle_position)

            if traffic_light_index >= 0:
                traffic_light_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(self.lights[traffic_light_index].pose.pose.position)
                traffic_light_position = self.waypoints.waypoints[traffic_light_waypoint_index].pose.pose.position

                if traffic_light_waypoint_index > vehicle_index:
                    distance_to_traffic_light = distance(vehicle_position, traffic_light_position)

                    if distance_to_traffic_light < SAFE_DISTANCE_TO_TRAFFIC_LIGHT:
                        rospy.loginfo("tl_detector: Traffic light ahead: {}".format(distance_to_traffic_light))
                        light = self.lights[traffic_light_index]
                        stop_line_index = self.get_index_of_closest_stop_line_to_current_pose(traffic_light_position)
                        stop_line_position = self.get_stop_line_positions()[stop_line_index].pose.pose
                        stop_light_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(stop_line_position.position)
                        state_of_traffic_light = self.get_light_state(light)
                        rospy.loginfo("tl_detector: Traffic light has state: {}".format(state_of_traffic_light))
                        return stop_line_waypoint_index, state_of_traffic_light

            else:
                rospy.loginfo("tl_detector: Stop light detection failed.")
                return -1, TrafficLight.UNKNOWN

    def get_index_of_closest_waypoint_to_current_pose(self, pose):
        """
        This member functions returns the index of the waypoint that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.waypoints.waypoints)


    def get_index_of_closest_traffic_light_to_current_pose(self, pose):
        """
        This member functions returns the index of the traffic light that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.lights)


    def get_index_of_closest_stop_line_to_current_pose(self, pose):
        """
        This member functions returns the index of the stop line that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.get_stop_line_positions())

    def get_stop_line_positions(self):
        """
        This member function returns a vector-of-vectors of stop-light-positions.
        Returns:
            stop_line_positions: array of 2-d-arrays
        """
        stop_line_positions = []
        for position in self.config['stop_line_positions']:
            current_point = Waypoint()
            current_point.pose.pose.position.x = position[0]
            current_point.pose.pose.position.y = position[1]
            current_point.pose.pose.position.z = 0.0
            stop_line_positions.append(current_point)

        return stop_line_positions

    def get_index_of_closest_point_to_current_pose(self, pose, positions):
        """
        This member function finds the index of that point in the set of points
        positions, which is closest to the pose.
        Arguments:
            pose: pose
            positions: positions given in "waypoints-structure"
        Returns:
            minimum_distance_idx: an integer, -1 means the search has not been succesfull
        """
        # default value for id
        minimum_distance_idx = -1
        # initialize this with a very large value
        minimum_distance_value = sys.float_info.max
        # define lambda-function for euclidan distance
        distance = lambda a,b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        # loop over all waypoints
        if positions is not None:
            for i in range(len(positions)):
                value = distance(pose, positions[i].pose.pose.position)
                if value < minimum_distance_value:
                    minimum_distance_value = value
                    minimum_distance_idx = i

        return minimum_distance_idx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
