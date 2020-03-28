
from __future__ import absolute_import, division, print_function, unicode_literals
#import matplotlib
#matplotlib.use('Agg')
#from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import glob
#from styx_msgs.msg import TrafficLight
import collections

import tensorflow as tf

#import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

# Import everything needed to edit/save/watch video clips
#import imageio
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#import IPython.display as display

#plt.style.use('ggplot')

from keras.preprocessing.image import ImageDataGenerator
#from keras_applications.mobilenet import MobileNetV2
from keras.models import Model
from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense
import os
import pathlib


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
        #self.RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
        #self.FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'
        # Colors (one for each class)
        cmap = ImageColor.colormap
        #print("Number of colors =", len(cmap))
        self.COLOR_LIST = sorted([c for c in cmap.keys()])

        self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')


    def load_and_train(self, data_dir):
        '''
        '''

        # Constants
        BATCH_SIZE = 32
        IMG_HEIGHT = 224
        IMG_WIDTH = 224

        # Folder structure needs to be data/<class>/*.png
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*.jpg')))
        print("image_count", image_count)

        # Get classes/labels from folder structure
        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
        print("CLASS_NAMES", self.CLASS_NAMES)


        # Create train generator
        train_datagen = ImageDataGenerator(rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)

        # Create test generator
        test_datagen = ImageDataGenerator(rescale=1./255)


        STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

        train_data_gen = train_datagen.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(self.CLASS_NAMES))

        test_data_gen = test_datagen.flow_from_directory(directory="test/",
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(self.CLASS_NAMES))

        base_model=MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dense(512,activation='relu')(x) #dense layer 3
        preds=Dense(4,activation='softmax')(x) #final layer with softmax activation

        model=Model(inputs=base_model.input,outputs=preds)

        # Print architecture
        for i,layer in enumerate(model.layers):
            print(i,layer.name)

        for layer in model.layers:
            layer.trainable=False
        # or if we want to set the first 20 layers of the network to be non-trainable
        # for layer in model.layers[:20]:
        #     layer.trainable=False
        # for layer in model.layers[20:]:
        #     layer.trainable=True
        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        # Adam optimizer
        # loss function will be categorical cross entropy
        # evaluation metric will be accuracy

        step_size_train=train_data_gen.n//train_data_gen.batch_size
        model.fit_generator(generator=train_data_gen,
                           steps_per_epoch=step_size_train,
                           epochs=10,
                           validation_data=test_data_gen,
                           validation_steps=800)


    def predict(self, model, image):

        return model.predict(image)


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

    def mobilenet_conv_block(self, x, kernel_size, output_channels):
        """
        Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
        """
        # assumes BHWC format
        input_channel_dim = x.get_shape().as_list()[-1]
        W = tf.Variable(tf.truncated_normal((kernel_size, kernel_size, input_channel_dim, 1)))

        # depthwise conv
        x = tf.nn.depthwise_conv2d(x, W, (1, 2, 2, 1), padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        # pointwise conv
        x = tf.layers.conv2d(x, output_channels, (1, 1), padding='SAME')
        x = tf.layers.batch_normalization(x)

        return tf.nn.relu(x)


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
        #image = Image.fromarray(image)
        ###
        # For reversing the operation:
        # im_np = np.asarray(im_pil)
        ###
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = self.COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

            #image = cv2.rectangle(image, start_point, end_point, color, thickness)
            #cv2.rectangle(image, (left, top), (right, bot), (128,128,128), 2)

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
        #image = Image.open(path_to_image)
        image = cv2.imread(path_to_image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image)
        #image_np = np.asarray(image, dtype=np.uint8)
        image_np = np.asarray(image)
        print("After asarray")
        print(image_np.shape)
        image_np = np.expand_dims(image_np, 0)

        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            height = image_np.shape[0]
            width = image_np.shape[1]
            box_coords = self.to_image_coords(boxes, height, width)
            print("Coordinates", box_coords)
            print("classes", classes)
            # Each class with be represented by a differently colored box
            self.draw_boxes(image, box_coords, classes)

            #plt.figure(figsize=(12, 8))
            #plt.imshow(image)


    def get_labels(self, filename="labels.csv", image_folder="data/*"):
        labels=[]
        f = open(filename, "w")

        for name in glob.glob(image_folder):
            labels.append(self.get_classification(cv2.imread(name)))
            f.write(str(labels[-1])+",\n")
        print(labels)
        counter = collections.Counter(labels)
        print(counter)
        print("UNKNOWN=4, GREEN=2, YELLOW=1, RED=0")


    def pipeline(self,img):
        draw_img = Image.fromarray(img)
        boxes, scores, classes = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = draw_img.size
        box_coords = self.to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        self.draw_boxes(draw_img, box_coords, classes)
        return np.array(draw_img)


if __name__ == '__main__':

    cls = TLClassifier()

    #for name in glob.glob("data/left0025.jpg"):
    # image = cls.detect("data/left0025.jpg")
    # #print(image)
    # plt.imshow(image)
    # cv2.imwrite("sample1_boxes.jpg", image)
    # plt.show()


    #### Video
    # clip = VideoFileClip('driving.mp4')
    #
    # with tf.Session(graph=cls.detection_graph) as sess:
    #     image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    #     detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    #     detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    #     detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    #
    #     new_clip = clip.fl_image(cls.pipeline)
    #
    #     # write to file
    #     new_clip.write_videofile('result.mp4')

    images = cls.load_and_train("train/")


    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # for image, label in labeled_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())
    #display.display(Image.open(str(image_path)))
