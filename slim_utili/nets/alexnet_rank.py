# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""




import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

from my_slim_layers import RandomMux

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      #Chong edited
      #network topology depends on decomposition scheme
      FLAGS = tf.app.flags.FLAGS

      import pickle
      with open(FLAGS.init_op_dict_path,'rb')as fhandle:
          K_dict = pickle.load(fhandle)

      mux_prob = 0.5

      #DEBUG
      #print('In net construction: %s'%str(K_dict))

      #conv2
      conv2_out_1 = slim.conv2d(net, 192, [5, 5], scope='conv2', activation_fn=None)
      #conv2 decomposed
      K = K_dict['alexnet_v2/conv2']
      conv2_v_out = slim.conv2d(net, K, [1, 5], scope='conv2_v', activation_fn=None)
      conv2_out_2 = slim.conv2d(conv2_v_out, 192, [5, 1], scope='conv2_h', activation_fn=None)
      #conv2 random mux
      conv2_out = RandomMux(conv2_out_1, conv2_out_2, mux_prob, is_training=is_training)
      conv2_out = slim.batch_norm(conv2_out, scope='conv2_bn', activation_fn=tf.nn.relu, is_training=is_training)

      net = slim.max_pool2d(conv2_out, [3, 3], 2, scope='pool2')

      #conv3
      conv3_out_1 = slim.conv2d(net, 384, [3, 3], scope='conv3', activation_fn=None)
      #conv3 decomposed
      K = K_dict['alexnet_v2/conv3']
      conv3_v_out = slim.conv2d(net, K, [1, 3], scope='conv3_v', activation_fn=None)
      conv3_out_2 = slim.conv2d(conv3_v_out, 384, [3, 1], scope='conv3_h', activation_fn=None)
      #conv3 random mux
      conv3_out = RandomMux(conv3_out_1, conv3_out_2, mux_prob, is_training=is_training)
      conv3_out = slim.batch_norm(conv3_out, scope='conv3_bn', activation_fn=tf.nn.relu, is_training=is_training)

      #conv4
      conv4_out_1 = slim.conv2d(conv3_out, 384, [3, 3], scope='conv4', activation_fn=None)
      #conv4 decomposed
      K = K_dict['alexnet_v2/conv4']
      conv4_v_out = slim.conv2d(conv3_out, K, [1, 3], scope='conv4_v', activation_fn=None)
      conv4_out_2 = slim.conv2d(conv4_v_out, 384, [3, 1], scope='conv4_h', activation_fn=None)
      #conv4 random mux
      conv4_out = RandomMux(conv4_out_1, conv4_out_2, mux_prob, is_training=is_training)
      conv4_out = slim.batch_norm(conv4_out, scope='conv4_bn', activation_fn=tf.nn.relu, is_training=is_training)

      #conv5
      conv5_out_1 = slim.conv2d(conv4_out, 256, [3, 3], scope='conv5', activation_fn=None)
      #conv5 decomposed
      K = K_dict['alexnet_v2/conv5']
      conv4_v_out = slim.conv2d(conv4_out, K, [1, 3], scope='conv5_v', activation_fn=None)
      conv5_out_2 = slim.conv2d(conv4_v_out, 256, [3, 1], scope='conv5_h', activation_fn=None)
      #conv5 random mux
      conv5_out = RandomMux(conv5_out_1, conv5_out_2, mux_prob, is_training=is_training)
      conv5_out = slim.batch_norm(conv5_out, scope='conv5_bn', activation_fn=tf.nn.relu, is_training=is_training)
      #Chong edited above
      net = slim.max_pool2d(conv5_out, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        #Chong edited
        #fc6
        fc6_out_1 = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
        #fc6 decomposed
        K = K_dict['alexnet_v2/fc6']
        fc6_v_out = slim.conv2d(net, K, [1, 5], scope='fc6_v', padding='VALID', activation_fn=None)
        fc6_out_2 = slim.conv2d(fc6_v_out, 4096, [5, 1], scope='fc6_h', padding='VALID')
        #fc6 random mux
        net = RandomMux(fc6_out_1, fc6_out_2, mux_prob, is_training=is_training)
        #Chong edited above

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer,
                          scope='fc8')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
alexnet_v2.default_image_size = 224
