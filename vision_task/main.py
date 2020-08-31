
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

from img_list import img_list
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_st'
NUM_CLASSES = 8
IMAGE_DIR = 'productionpanos'
SAVE_DIR = 'productionpanos_out'

SCALE_PERCENT = 50 # percent of original size, if we want to resize the image
SCORE_THRESH = 0.7

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    try:
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
    except:
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
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

image_list = img_list(IMAGE_DIR)
for img in image_list:
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(img)

    width = int(image.shape[1] * SCALE_PERCENT / 100)
    height = int(image.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Mostly for Goal 1
    test_boxes = np.squeeze(boxes)
    test_scores = np.squeeze(scores)
    test_classes = np.squeeze(classes)
    temp_class = []
    name = ''
    for i in range(test_boxes.shape[0]):
        if test_scores is None or test_scores[i] > SCORE_THRESH:
            box = tuple(test_boxes[i].tolist())
            if not test_classes[i] in temp_class:
                temp_class.append(test_classes[i])
                name =  name + '_' + (category_index[test_classes[i]]["name"])


    # Draw the results of the detection (aka 'visulaize the results')
    # for Goal 2
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=10,
        min_score_thresh=SCORE_THRESH)

    # All the results have been drawn on image. Now display the image.
    # following might not work with windows, works with mac os
    image_name = img.split("/") #getting image name, from absolute img name
    image_name = image_name[-1].split(".")
    image_name = image_name[0] + name + ".jpg" # new image name to be saved with
    print(image_name)
    cv2.imwrite(os.path.join(SAVE_DIR, image_name), image)
