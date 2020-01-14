from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import my_slim_layers
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
slim = tf.contrib.slim


#Chong added
import DnnUtili
mvm = DnnUtili.MaskingVariableManager()   

decomposed_layer=my_slim_layers.conv2d_mask

#use original conv2d
#myconv2d = slim.conv2d
myconv2d = decomposed_layer
#myconv2d = my_slim_layers.conv2d_svd_mask
#use conv2d with masks
myconv2d1 = decomposed_layer
#myconv2d1=myconv2d

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([myconv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def squeeze(inputs, num_outputs):
    return myconv2d(inputs, num_outputs, [1, 1], stride=1, padding='VALID',scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = myconv2d(inputs, num_outputs, [1, 1], stride=1, padding='VALID', scope='1x1')
        e3x3 = myconv2d(inputs, num_outputs, [3, 3], stride=1, padding='SAME', scope='3x3')
    return tf.concat(values=[e1x1, e3x3], axis=3)


def squeezenet(images,
               num_classes=1000,
               is_training=False,
               scope='squeezenet'):
    """Original squeezenet architecture for 227x227 images."""


    #DEBUG
    print('squeezenet: is_training is %d'%is_training)
    with tf.variable_scope('squeezenet', values=[images]) as sc:
        end_point_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([fire_module, myconv2d,
                             slim.max_pool2d, slim.avg_pool2d],
                            outputs_collections=[end_point_collection]):
            net = myconv2d(images, 64, [3, 3], stride=2, padding='VALID',scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
            net = fire_module(net, 16, 64, scope='fire2')
            net = fire_module(net, 16, 64, scope='fire3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
            net = fire_module(net, 32, 128, scope='fire4')
            net = fire_module(net, 32, 128, scope='fire5')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool5')
            net = fire_module(net, 48, 192, scope='fire6')
            net = fire_module(net, 48, 192, scope='fire7')
            net = fire_module(net, 64, 256, scope='fire8')
            net = fire_module(net, 64, 256, scope='fire9')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='drop9')
            net = myconv2d(net, num_classes, [1, 1], stride=1, padding='VALID',scope='conv10')
            net = slim.avg_pool2d(net, [13, 13], stride=1, padding='VALID', scope='avgpool10')
            logits = tf.squeeze(net, [1, 2], name='logits')
            logits = utils.collect_named_outputs(end_point_collection,
                                                 sc.name + '/logits',
                                                 logits)
        end_points = utils.convert_collection_to_dict(end_point_collection)
        return logits, end_points

squeezenet.default_image_size=227


def squeezenet_arg_scope(weight_decay=0.0002, is_training=None):
    with slim.arg_scope(
        #[myconv2d],
        [slim.conv2d],
        weights_initializer= tf.contrib.layers.xavier_initializer_conv2d(),
        #biases_initializer= tf.contrib.layers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
            return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
