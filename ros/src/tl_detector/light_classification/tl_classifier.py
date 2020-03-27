#from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import glob
from styx_msgs.msg import TrafficLight
import collections

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

#plt.style.use('ggplot')

class TLClassifier(object):
    """
    This class implements a very simple traffic light classifier.
    The classifier looks at a picture and counts the pixels in a specific color range.
    To be effective, the colorspace is HSV; here, red and yellow can be distinguished
    with ease. Green traffic lights are neglected because these can be passed.
    """
    def __init__(self):
        """
        This member function initializes the classifier.
        It sets the bounds for image classification and intializes the
        state of a possible traffic light in an image.
        """
        self.image = None
        # Lower bound for color in image to be "valid red" in HSV-color-space (!)
        self.HSV_bound_red_low = np.array([0, 120, 120],np.uint8)
        # Upper bound for color in image to be "valid red" in HSV-color-space (!)
        self.HSV_bound_red_high = np.array([10, 255, 255],np.uint8)
        # Lower bound for color in image to be "valid yellow" in HSV-color-space (!)
        self.HSV_bound_yellow_low = np.array([25, 120, 120],np.uint8)
        # Upper bound for color in image to be "valid yellow" in HSV-color-space (!)
        self.HSV_bound_yellow_high = np.array([45.0, 255, 255],np.uint8)
        # Constant defining how many pixels of certain color must
        # be present to be detected as a valid red or yellow
        # traffic light
        self.number_of_pixels_tolerance = 60
        # Member variable indicating a red traffic light
        self.red_light = False
        # Member variable indicating a red yellow traffic light
        self.yellow_light = False


        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        self.SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        self.RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
        self.FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'
        # Colors (one for each class)
        cmap = ImageColor.colormap
        print("Number of colors =", len(cmap))
        self.COLOR_LIST = sorted([c for c in cmap.keys()])



    def get_classification(self, image):
        """
        This member function determines the color of the traffic
        light in the image. It requires an image as input.
        It returns the state of a traffic light as an enumerted type.
        """
        self.red_light = False
        self.yellow_light = False
        self.image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        in_red_range_frame = cv2.inRange(self.image, self.HSV_bound_red_low, self.HSV_bound_red_high)
        number_of_red_pixels = cv2.countNonZero(in_red_range_frame)
        if number_of_red_pixels > self.number_of_pixels_tolerance:
            self.red_light = True
            self.yellow_light = False

        in_yellow_range_frame = cv2.inRange(self.image, self.HSV_bound_yellow_low, self.HSV_bound_yellow_high)
        number_of_yellow_pixels = cv2.countNonZero(in_yellow_range_frame)
        if number_of_yellow_pixels > self.number_of_pixels_tolerance:
            self.red_light = False
            self.yellow_light = True

        if self.red_light:
            return TrafficLight.RED

        if self.yellow_light:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN


    def vanilla_conv_block(self, x, kernel_size, output_channels):
        """
        Vanilla Conv -> Batch Norm -> ReLU
        """
        x = tf.layers.conv2d(
            x, output_channels, kernel_size, (2, 2), padding='SAME')
        x = tf.layers.batch_normalization(x)
        return tf.nn.relu(x)

    # TODO: implement MobileNet conv block
    def mobilenet_conv_block(self, x, kernel_size, output_channels):
        """
        Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
        """
        pass

    def run(self):
        # constants but you can change them so I guess they're not so constant :)
        INPUT_CHANNELS = 32
        OUTPUT_CHANNELS = 512
        KERNEL_SIZE = 3
        IMG_HEIGHT = 256
        IMG_WIDTH = 256

        with tf.Session(graph=tf.Graph()) as sess:
            # input
            x = tf.constant(np.random.randn(1, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), dtype=tf.float32)

            with tf.variable_scope('vanilla'):
                vanilla_conv = vanilla_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)
            with tf.variable_scope('mobile'):
                mobilenet_conv = mobilenet_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)

            vanilla_params = [
                (v.name, np.prod(v.get_shape().as_list()))
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla')
            ]
            mobile_params = [
                (v.name, np.prod(v.get_shape().as_list()))
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mobile')
            ]

            print("VANILLA CONV BLOCK")
            total_vanilla_params = sum([p[1] for p in vanilla_params])
            for p in vanilla_params:
                print("Variable {0}: number of params = {1}".format(p[0], p[1]))
            print("Total number of params =", total_vanilla_params)
            print()

            print("MOBILENET CONV BLOCK")
            total_mobile_params = sum([p[1] for p in mobile_params])
            for p in mobile_params:
                print("Variable {0}: number of params = {1}".format(p[0], p[1]))
            print("Total number of params =", total_mobile_params)
            print()

            print("{0:.3f}x parameter reduction".format(total_vanilla_params /
                                                     total_mobile_params))


            detection_graph = load_graph(self.SSD_GRAPH_FILE)
            # detection_graph = load_graph(RFCN_GRAPH_FILE)
            # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

            # The input placeholder for the image.
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

            # The classification of the object (integer id).
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    #
    # Utility funcs
    #
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = self.COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def detect(self, path_to_image):
        # Load a sample image.
        image = Image.open(path_to_image)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                                feed_dict={image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image.size
            box_coords = to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            draw_boxes(image, box_coords, classes)

            plt.figure(figsize=(12, 8))
            plt.imshow(image)

    def time_detection(self, sess, img_height, img_width, runs=10):
        image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

        # warmup
        gen_image = np.uint8(np.random.randn(1, img_height, img_width, 3))
        sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: gen_image})

        times = np.zeros(runs)
        for i in range(runs):
            t0 = time.time()
            sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np})
            t1 = time.time()
            times[i] = (t1 - t0) * 1000
        return times

        def get_times(self):
            with tf.Session(graph=detection_graph) as sess:
                times = time_detection(sess, 600, 1000, runs=10)

        def visualize_times(self):
            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)
            plt.title("Object Detection Timings")
            plt.ylabel("Time (ms)")

            # Create the boxplot
            plt.style.use('fivethirtyeight')
            bp = ax.boxplot(times)


if __name__ == '__main__':
    labels=[]
    f = open("labels.csv", "w")
    print(f)
    cls = TLClassifier()
    for name in glob.glob('data/*'):
        #print(name)
        labels.append(cls.get_classification(cv2.imread(name)))
        f.write(str(labels[-1])+",\n")

    print(labels)
    counter = collections.Counter(labels)
    print(counter)
    print("UNKNOWN=4, GREEN=2, YELLOW=1, RED=0")
