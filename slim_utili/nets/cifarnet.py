# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the CIFAR-10 model definition."""

import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import my_slim_layers

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def cifarnet(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
  """Creates a variant of the CifarNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  batch_size = images.get_shape().as_list()[0]
  end_points = {}
 
  decomposed_layer=my_slim_layers.conv2d_mask

  #use original conv2d
  #myconv2d = slim.conv2d
  myconv2d = decomposed_layer
  #myconv2d = my_slim_layers.conv2d_svd_mask
  #use conv2d with masks
  myconv2d1 = decomposed_layer
  #myconv2d1=myconv2d

  #Chong added
  import DnnUtili
  
  with tf.variable_scope(scope, 'CifarNet', [images, num_classes]):
    net = myconv2d1(images, 192, [5, 5], scope='conv1')
    net = myconv2d1(net, 160, [1, 1], scope='cccp1')
    net = myconv2d1(net, 96, [1, 1], scope='cccp2')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')
    net = myconv2d1(net, 192, [5, 5], scope='conv2')
    net = myconv2d(net, 192, [1, 1], scope='cccp3')
    net = myconv2d(net, 192, [1, 1], scope='cccp4')
    net = slim.avg_pool2d(net, [3, 3], 2, scope='pool2')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')
    net = myconv2d(net, 192, [3, 3], scope='conv3')
    net = myconv2d(net, 192, [1, 1], scope='cccp5')
    net = myconv2d(net, num_classes, [1, 1], scope='cccp6')
    net = slim.avg_pool2d(net, [7, 7], 1, scope='pool3')
    
    #logits should be of shape [batch_size, num_classes]
    #using squeeze will sequeeze out the first dimension, which is batch_size when batch_size is 1
    logits = tf.reshape(net, [batch_size, num_classes])
    
    #if batch_size is one, add a vector of length num_classes to the endpoints dict
    if batch_size == 1:
        logits = tf.squeeze(net)
        end_points['logits_vector'] = logits
        
    logits = tf.reshape(logits, [batch_size, num_classes])

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
cifarnet.default_image_size = 32


def cifarnet_arg_scope(weight_decay=0.00012):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      #[myconv2d],
      [slim.conv2d],
      weights_initializer= tf.contrib.layers.xavier_initializer_conv2d(),
      #biases_initializer= tf.contrib.layers.xavier_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu):
    with slim.arg_scope(
        [slim.fully_connected],
        #biases_initializer=tf.constant_initializer(0.0),
        #weights_initializer=trunc_normal(0.04),
        #weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc
