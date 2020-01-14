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





import tensorflow as tf

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
  end_points = {}

  with tf.variable_scope(scope, 'CifarNet', [images, num_classes]):

    #Chong edited
    #network topology depends on decomposition scheme
    FLAGS = tf.app.flags.FLAGS

    import pickle
    with open(FLAGS.init_op_dict_path,'rb')as fhandle:
      K_dict = pickle.load(fhandle)

    #DEBUG
    #print('In net construction: %s'%str(K_dict))

    net = slim.conv2d(images, 192, [5, 5], scope='conv1')

    #net = slim.conv2d(net, 160, [1, 1], scope='cccp1')
    K = K_dict['CifarNet/cccp1']
    net = slim.conv2d(net, K, [1, 1], scope='cccp1_v', activation_fn=None)
    net = slim.conv2d(net, 160, [1, 1], scope='cccp1_h')

    #net = slim.conv2d(net, 96, [1, 1], scope='cccp2')
    K = K_dict['CifarNet/cccp2']
    net = slim.conv2d(net, K, [1, 1], scope='cccp2_v', activation_fn=None)
    net = slim.conv2d(net, 96, [1, 1], scope='cccp2_h')

    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout1')

    #net = slim.conv2d(net, 192, [5, 5], scope='conv2')
    K = K_dict['CifarNet/conv2']
    net = slim.conv2d(net, K, [1, 5], scope='conv2_v', activation_fn=None)
    net = slim.conv2d(net, 192, [5, 1], scope='conv2_h')

    #net = slim.conv2d(net, 192, [1, 1], scope='cccp3')
    K = K_dict['CifarNet/cccp3']
    net = slim.conv2d(net, K, [1, 1], scope='cccp3_v', activation_fn=None)
    net = slim.conv2d(net, 192, [1, 1], scope='cccp3_h')

    #net = slim.conv2d(net, 192, [1, 1], scope='cccp4')
    K = K_dict['CifarNet/cccp4']
    net = slim.conv2d(net, K, [1, 1], scope='cccp4_v', activation_fn=None)
    net = slim.conv2d(net, 192, [1, 1], scope='cccp4_h')

    net = slim.avg_pool2d(net, [3, 3], 2, scope='pool2')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout2')

    #net = slim.conv2d(net, 192, [3, 3], scope='conv3')
    K = K_dict['CifarNet/conv3']
    net = slim.conv2d(net, K, [1, 3], scope='conv3_v', activation_fn=None)
    net = slim.conv2d(net, 192, [3, 1], scope='conv3_h')

    net = slim.conv2d(net, 192, [1, 1], scope='cccp5')
    net = slim.conv2d(net, num_classes, [1, 1], scope='cccp6')
    net = slim.avg_pool2d(net, [7, 7], 1, scope='pool3')
    logits = tf.squeeze(net)

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
cifarnet.default_image_size = 32


def cifarnet_arg_scope(weight_decay=0.004):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d],
      #weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
      weights_initializer= tf.contrib.layers.xavier_initializer_conv2d(),
      activation_fn=tf.nn.relu):
    with slim.arg_scope(
        [slim.fully_connected],
        biases_initializer=tf.constant_initializer(0.0),
        #weights_initializer=trunc_normal(0.04),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc
