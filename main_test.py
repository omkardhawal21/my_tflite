# import cv2

# # Pretrained classes in the model
# classNames = {1: 'fish'}


# def id_class_name(class_id, classes):
#     for key, value in classes.items():
#         if class_id == key:
#             return value

# print('hello')
# # Loading model
# model = cv2.dnn.readNetFromTensorflow('/home/omkar/machine-learning/fish_detection/fish_final/tflite/frozen_inference_graph.pb',
#                                       '/home/omkar/machine-learning/datasets/animals/fish/label_map.pbtxt')
# print(model)
# image = cv2.imread("/home/omkar/machine-learning/datasets/animals/fish/test_img/45.jpg")
# print(image)

# image_height, image_width, _ = image.shape

# model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))

# output = model.forward()
# print(output[0,0,:,:].shape)

# for detection in output[0, 0, :, :]:
#     print(detection)
#     confidence = detection[2]
#     if confidence > .5:
#         class_id = detection[1]
#         class_name=id_class_name(class_id,classNames)
#         print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
#         box_x = detection[3] * image_width
#         box_y = detection[4] * image_height
#         box_width = detection[5] * image_width
#         box_height = detection[6] * image_height
#         cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
#         cv2.putText(image,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))





# cv2.imshow('image', image)
# # cv2.imwrite("image_box_text.jpg",image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()





import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv2




# Checks and deletes the output file
# You cant have a existing file or it will through an error


# Playing video from file
cap = cv2.VideoCapture(0)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.


sys.path.append("..")

# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# Model preparation
MODEL_NAME = '/home/omkar/machine-learning/road_sign_detection/tflite'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', '/home/omkar/machine-learning/datasets/road_signs/label_map.pbtxt')
NUM_CLASSES = 6

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            if ret == True:
                # Saves for video

                # Display the resulting frame
                cv2.imshow('Charving Detection', frame)

                # Close window when "Q" button pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()