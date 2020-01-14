from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def preprocess_image(image, output_height, output_width, is_training=False):	
  resized_image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
  tf.summary.image('image', tf.expand_dims(image, 0))
  return tf.image.per_image_standardization(resized_image) 