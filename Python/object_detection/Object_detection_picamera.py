# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys
import threading
import RPi.GPIO as GPIO
import time
# Initialize GPIO pins for servo control
servo_pin1 = 17  # Change this to the actual GPIO pin you're using for the first servo
servo_pin2 = 18  # Change this to the actual GPIO pin you're using for the second servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin1, GPIO.OUT)
GPIO.setup(servo_pin2, GPIO.OUT)

# Create PWM objects for controlling the servos
servo_pwm1 = GPIO.PWM(servo_pin1, 50)  # 50 Hz PWM frequency
servo_pwm2 = GPIO.PWM(servo_pin2, 50)  # 50 Hz PWM frequency
servo_pwm1.start(0)  # Start PWM for the first servo with duty cycle 0
servo_pwm2.start(0)  # Start PWM for the second servo with duty cycle 0

last_detection_time = None
last_detection_time2 = None  # Add this line to initialize the variable for PaperCup

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

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

DESIRED_WIDTH, DESIRED_HEIGHT = 320, 240

class WebcamVideoStream:
    def __init__(self, src=0, width=DESIRED_WIDTH, height=DESIRED_HEIGHT):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, width)
        self.stream.set(4, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def run_inference(stream):
    global last_detection_time  # Declare the use of the global variable
    global last_detection_time2  # Declare the use of the global variable


    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                frame = stream.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_expanded = np.expand_dims(frame_rgb, axis=0)

                # Perform detection
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                    
                plastic_bottle_detected = False
                paper_cup_detected = False
                
                # Iterate through detected objects
                for i in range(len(classes[0])):
                    class_id = int(classes[0][i])
                    class_name = category_index[class_id]['name']
                    score = scores[0][i]

                    # Check if the detected class is the one you want to trigger the servo for
                    if class_name == 'PlasticBottle' and score > 0.85:
                        plastic_bottle_detected = True
                        last_detection_time = time.time()  # Update the last detection time
                        # Control the servo motor
                        servo_pwm1.ChangeDutyCycle(5.5)  # Move servo to a specific angle
                        time.sleep(2)
                        servo_pwm1.ChangeDutyCycle(0)  # Stop servo
                        
                    if class_name == 'PaperCup' and score > 0.85:
                        # Control the second servo motor for PaperCup
                        paper_cup_detected = True
                        last_detection_time2 = time.time()  # Update the last detection time
                        servo_pwm2.ChangeDutyCycle(5.5)  # Move servo to a specific angle for PaperCup
                        time.sleep(2)
                        servo_pwm2.ChangeDutyCycle(0)  # Stop servo for PaperCup

                    if last_detection_time and (time.time() - last_detection_time > 2):
                        # Move the first servo to a different position for PlasticBottle
                        servo_pwm1.ChangeDutyCycle(2.5)  # Adjust this value as needed for PlasticBottle
                        time.sleep(2)
                        last_detection_time = None  # Reset the last detection time for PlasticBottle
                        servo_pwm1.ChangeDutyCycle(0)  # Adjust this value as needed for PlasticBottle

                    # Check if more than 2 seconds have passed since the last detection for PaperCup
                    if last_detection_time2 and (time.time() - last_detection_time2 > 2):
                        # Move the second servo to a different position for PaperCup
                        servo_pwm2.ChangeDutyCycle(2.5)  # Adjust this value as needed for PaperCup
                        time.sleep(2)
                        last_detection_time2 = None  # Reset the last detection time for PaperCup
                        servo_pwm2.ChangeDutyCycle(0)  # Adjust this value as needed for PaperCup


                # Visualization of the results of a detection
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    min_score_thresh=0.85)

                cv2.imshow('Team etneb', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
# Start video stream
video_stream = WebcamVideoStream(src=0, width=DESIRED_WIDTH, height=DESIRED_HEIGHT).start()

# Start inference
run_inference(video_stream)

servo_pwm1.stop()
servo_pwm2.stop()
GPIO.cleanup()
video_stream.stop()
cv2.destroyAllWindows()
