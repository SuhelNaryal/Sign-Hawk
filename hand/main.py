#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Video) From TF2 Saved Model
=====================================
"""
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]
POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = "models/hand_detector/label_map.pbtxt"

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.4

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = "models/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

#landmark = tf.lite.Interpreter('hand_landmark.tflite')
#landmark.allocate_tensors()

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


print('Running inference for Webcam', end='')

# Initialize Webcam
videostream = cv2.VideoCapture(0)
ret = videostream.set(3, 1280)
ret = videostream.set(4, 720)

while True:
    tic = time.time()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = videostream.read()
    frame = cv2.flip(frame, 1)
    frame_raw = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    imH, imW, _ = frame.shape

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            # increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            """
            crop_img = frame_raw[ymin:ymax][xmin:xmax]
            #crop_img = cv2.resize(crop_img, dsize=(256, 256))
            crop_img = np.ascontiguousarray(2 * ((crop_img / 255) - 0.5).astype('float32'))
            landmark_tensor = tf.convert_to_tensor(crop_img)
            landmark_tensor = landmark_tensor[tf.newaxis, ...]
            landmark.set_tensor(landmark.get_input_details()[0]['index'], landmark_tensor)
            landmark.invoke()
            joints = landmark.get_tensor(landmark.get_output_details()[0]['index'])
            joints = joints.reshape(-1, 3)
            joints[:, 0] = (joints[:, 0]*256 / (ymax-ymin) ) + ymin
            joints[:, 1] = (joints[:, 1]*256 / (xmax-xmin) ) + xmin
            print(joints)

            for point in joints:
                x, y, z = point
                cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            for connection in connections:
                x0, y0, z0 = joints[connection[0]]
                x1, y1, z1 = joints[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
            """
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            # Draw label
            object_name = category_index[int(classes[i])][
                'name']  # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  # Draw label text

    cv2.putText(frame, 'Objects Detected : ' + str(count), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 235, 52), 2,
                cv2.LINE_AA)
    cv2.imshow('Objects Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    toc = time.time()
    print('time:', (toc-tic)*1000)
    print('fps:', 1/(toc-tic))

cv2.destroyAllWindows()
print("Done")