# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys
import time

# For arduino serial comms
import serial

import RPi.GPIO as GPIO


# Set up serial communication with Arduino Uno
ports = ['/dev/ttyUSB0', '/dev/ttyUSB1']
def connect_to_serial(ports, baud_rate=9600):
    for port in ports:
        try:
            ser = serial.Serial(port, 9600)
            return ser
        except:
            print(f'Error connecting to {port}')
ser = connect_to_serial(ports)

# Set up camera constants
IM_WIDTH = 640
IM_HEIGHT = 480

# Select camera type (use USB webcam)
camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilities
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'Trash2'
#initail = 'ssdlite_mobilenet_v2_coco_2018_05_09lol','card_model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','label_mapTrash2.pbtxt')
#initial='mscoco_label_mapTrash.pbtxt','card_labelmap.pbtxt'

# Number of classes the object detector can identify
NUM_CLASSES = 5
#initila=90 , 13

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')

# Create a TensorFlow 2.x session
sess = tf.compat.v1.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

camera = cv2.VideoCapture(0)
ret = camera.set(3, IM_WIDTH)
ret = camera.set(4, IM_HEIGHT)

# Add variables to track the last sent detection and the time of the last sent detection
prev_class_name = None
last_sent_time = 0

while True:
    t1 = cv2.getTickCount()

    ret, frame = camera.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)
        
    max_confidence = 0
    max_confidence_index = -1
    for i in range(len(np.squeeze(scores))):
        if np.squeeze(scores)[i] > 0.85:  # Confidence threshold
            if np.squeeze(scores)[i] > max_confidence:
                max_confidence = np.squeeze(scores)[i]
                max_confidence_index = i

    if max_confidence_index != -1:
        class_id = int(np.squeeze(classes)[max_confidence_index])
        class_name = category_index[class_id]['name']

        # Only send if it's a new detection or cooldown period has elapsed (THE SEND O ARDUINO PART)
        current_time = time.time()
        if class_name != prev_class_name or current_time - last_sent_time > 2:  # Change cooldown period as needed
            print('Sending class name to Arduino:', class_name)
            ser.write(f'{class_name}\n'.encode('utf-8'))
            prev_class_name = class_name
            last_sent_time = current_time
                
    cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object Detection', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
ser.close()
