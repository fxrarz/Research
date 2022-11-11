'''
POSE ESTIMATION

Platform: Colab

Requirement: CPU

Written on: 14 September 2021

Tested on: 22 November 2021

Author: A.S. Faraz Ahmed

Description: 
    Deploy in a Edge Computer
'''

import tensorflow as tf
import numpy as np
import cv2
import os

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

if 'pose_model' in os.listdir('/content/'):
  pose_estimation = tf.saved_model.load('pose_model')
else:
  import tensorflow_hub as hub
  pose_estimation = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

input_size = 192

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=-2.50)
  # fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=6, linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)
  edge_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'r', 'w', 'c', 'm','b', 'g', 'r', 'c', 'm', 'y', 'g', 'w']
  
  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot
  
def _keypoints_and_edges_for_display(keypoints_with_scores, height, width,
                                     keypoint_threshold=0.11):
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

def movenet(input_image):
    model = module.signatures['serving_default']
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoint_with_scores = outputs['output_0'].numpy()
    return keypoint_with_scores
    
bg_image = tf.io.read_file('/content/bg.jpg')
bg_image = tf.image.decode_jpeg(bg_image)
input_image_bg = tf.expand_dims(bg_image, axis=0)
input_image_bg = tf.image.resize_with_pad(input_image_bg, input_size, input_size)

def crop(output_overlay):
    row = []
    indexs = []
    for i in range(output_overlay.shape[0]):
        index = 0
        for x in output_overlay[i]:
            if x[0] != 0 or x[1] != 0 or x[2] != 0:
                row.append(i)
                indexs.append(index)
                break
            index+=1
    sx = min(indexs)
    sy = row[0]
    row = []
    indexs = []
    for i in range(output_overlay.shape[0]):
        index = 0
        for x in output_overlay[i]:
            if x[0] != 0 or x[1] != 0 or x[2] != 0:
                row.append(i)
                indexs.append(index)
            index+=1
    dx = max(indexs)
    dy = row[-1]
    cropped_image = output_overlay[sy:dy, sx:dx]
    return cv2.imwrite('output.jpg',cropped_image)
    
def test(image_path):
    image = tf.io.read_file(image_path) # Load the input image.
    image = tf.image.decode_jpeg(image)
    input_image = tf.expand_dims(image, axis=0) # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    keypoint_with_scores = movenet(input_image) # Run model inference.
    display_image = tf.expand_dims(image, axis=0) # Visualize the predictions with image_bg.
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoint_with_scores)

    image = crop(output_overlay)
    image_path = "output.jpg"

    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = sk_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    plt.figure(figsize=(25, 25))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(frame)
    plt.title("Original Image")
    plt.axis("off")

    ax = plt.subplot(1, 3, 3)
    plt.imshow(cropped_image)
    plt.title("Cropped Image")
    plt.axis("off")
 
    ax = plt.subplot(1, 3, 2)
    plt.imshow(output_overlay)
    plt.title("Overlay Image")

img_height = 100
img_width = 100
sk_model = tf.keras.models.load_model('skeleton_tf1_all.h5')
cap = cv2.VideoCapture(0)
while 1:
    _, frame = cap.read()
    if not _:
        break
    cv2.imwrite("test.jpg")
    test('test.jpg')

