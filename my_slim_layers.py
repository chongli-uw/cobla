#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:35:22 2017
Modifications from 
#/usr/lib/python3.5/site-packages/tensorflow/contrib/layers/python/layers
@author: chongli
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import variables as tf_variables

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

import numpy as np
import sklearn.decomposition
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import fast_dot
#make sure sklearn.utils is using fast BLAS functions
import warnings
from sklearn.exceptions import NonBLASDotWarning
warnings.simplefilter('always', NonBLASDotWarning) 

import scipy
import scipy.io
import scipy.linalg
import math
import random

import os
import pickle
import scipy
import scipy.io
from collections import OrderedDict
import xxhash

import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')
import TFInclude
import DnnUtili

#################################################################
class DecompositionScheme(object):
    '''describe a decomposition scheme in a convolution layer'''
    def __init__(self, weight_4d, scheme, name=None):
        self.weight_4d=None
        self.weight_2d=None
        self.scheme=None
        assert type(name) is str
        self.name=name

        #svd results
        self.P = None
        self.S = None
        self.V = None

        assert weight_4d.ndim == 4, 'input weight_4d error, got shape %s'%weight_4d.shape
        assert type(weight_4d) is np.ndarray
        self.weight_4d=weight_4d

        assert scheme in ('tai','microsoft'), 'unsupported decomposition scheme %s'%scheme
        self.scheme=scheme

    def update_weight_2d(self, weight_2d):
        '''rewrite the internal values from weight_2d'''
        assert weight_2d.ndim == 2, 'input weight_2d error, got shape %s'%weight_2d.shape
        assert type(weight_2d) is np.ndarray

        self.P = None
        self.S = None
        self.V = None

        self.weight_2d = weight_2d
        self.K = min(self.weight_2d.shape)

        self.get_initial_values()

    def get_weight_2d(self):
        '''return the flattened 2D matrix and the maximum rank of the 2D matrix, given the decomposition scheme'''
        height,width,input_channel,output_channel = self.weight_4d.shape
        C = input_channel
        N = output_channel

        if self.scheme=='tai':
            #from [height, width, input_channel, output_channel] to [input_channel,width, height,output_channel]
            weight = self.weight_4d.transpose([2,1,0,3])
            self.weight_2d = weight.reshape((C*width,height*N))
        elif self.scheme=='microsoft':
            #from [height, width, input_channel, output_channel] to [output_channel, height, width, input_channel]
            weight = self.weight_4d.transpose([3, 0, 1, 2])
            self.weight_2d=weight.reshape(output_channel, height*width*input_channel)
        else:
            raise ValueError()

        self.K = min(self.weight_2d.shape)
        
        return self.weight_2d, self.K

    def svd(self):
        '''P is U*S, absorbing the singular values into the left singular vector'''

        if self.P is not None and self.S is not None and self.V is not None:
            return self.P, self.S, self.V

        if self.weight_2d is None:
            self.get_weight_2d()

        weight_2d_hash = xxhash.xxh32(self.weight_2d.tobytes()).hexdigest()

        if not os.path.exists('/home/chongli/ramdisk'):
            os.makedirs('/home/chongli/ramdisk')
        svd_pickle_path = '/home/chongli/ramdisk/svd_result_pickle.pickle'

        #if svd result pickle not exist, initialize a OrderedDict
        if not os.path.isfile(svd_pickle_path):
            svd_result_dict = OrderedDict()
        else:
            with open(svd_pickle_path,'rb') as f:
                svd_result_dict = pickle.load(f)
            assert type(svd_result_dict) is OrderedDict

        if self.name not in svd_result_dict:
            svd_result_dict = OrderedDict()
            #compute svd and save to file
            U,s,V = np.linalg.svd(self.weight_2d)
            svd_result_dict[self.name] = ((U,s,V), xxhash.xxh32(self.weight_2d.tobytes()).hexdigest() )

            with open(svd_pickle_path, 'wb') as f:  
                pickle.dump(svd_result_dict, f, protocol=-1)
        else:
            #name exist in dict, check hash
            if weight_2d_hash==svd_result_dict[self.name][1]:
                U,s,V = svd_result_dict[self.name][0]
            else: 
                #print('my_slim_layer: svd result loaded from %s for layer %s does not match'%(svd_pickle_path,self.name))
                os.remove(svd_pickle_path)
                svd_result_dict = OrderedDict()

                #compute svd and save to file
                U,s,V = np.linalg.svd(self.weight_2d)
                svd_result_dict[self.name] = ((U,s,V), weight_2d_hash)

                with open(svd_pickle_path, 'wb') as f:  
                    pickle.dump(svd_result_dict, f, protocol=-1)

        assert U.shape[0] == self.weight_2d.shape[0] and V.shape[0] == self.weight_2d.shape[1]
        assert U.shape[0] == U.shape[1] and V.shape[0] == V.shape[0]
        assert s.size == U.shape[0] or s.size == V.shape[0]

        #absort the singular values into U to get P
        P = np.array(U, dtype=np.float32)
        for i in range(s.size):
            P[:,i] *= s[i]

        self.P = P
        self.S = s
        self.V = V
        return self.P, self.S, self.V

    def get_initial_values(self, add_and_svd_K = None):
        '''return the initial weight of the decomposed layers'''
        if self.weight_2d is None:
            self.get_weight_2d()

        if self.P is None:
            self.svd()

        height,width,input_channel,output_channel = self.weight_4d.shape
        C = input_channel
        N = output_channel
        if add_and_svd_K is None:
            K = self.K
        else:
            assert add_and_svd_K <= self.K
            K = int(add_and_svd_K)

        P = self.P
        V = self.V
       
        if self.scheme=='tai':
            v_init_value = P[:, :K]
            #important: notice that W is (C*width, height*N), and the reshape is (C, width, 1, K)
            v_init_value = np.reshape(v_init_value, (C, width, 1, K))
            #reshape to [K, width, 1, C]
            v_init_value = np.transpose(v_init_value, [3, 1, 2, 0])
            #reshape to tensorflow data format [height, width, input_channel, output_channel] 
            v_init_value = np.transpose(v_init_value, [2,1,3,0])

            #initial numpy value of the decomposed h
            h_init_value = V[:K, :]
            #important: notice that W is (C*width, height*N),the last 2 dimensions of the reshape matches the original tensor
            h_init_value = np.reshape(h_init_value, (K, 1, height, N))
            #reshape to [N,1,height,K],
            h_init_value = np.transpose(h_init_value, [3,1,2,0])
            #reshape to tensorflow data format [height, width, input_channel, output_channel] 
            h_init_value = np.transpose(h_init_value, [2,1,3,0])
        elif self.scheme=='microsoft':
            #initial numpy value of the decomposed v
            v_init_value = V[:K, :]
            #[K1, height*width*input_channel]
            v_init_value = v_init_value.reshape((K, height, width*input_channel))
            v_init_value = v_init_value.reshape((K, height, width, input_channel))
            #[output_channel, height, width, input_channel] to [height, width, input_channel, output_channel]
            v_init_value = v_init_value.transpose([1, 2, 3, 0])

            #initial numpy value of the decomposed h
            h_init_value =P[:, :K]
            #[output_channel, K1] to [1, 1, output_channel, K1]
            h_init_value = np.reshape(h_init_value, [1, 1, output_channel, K])
            #[1, 1, output_channel, K1] to [1, 1, K1, output_channel]
            h_init_value = np.transpose(h_init_value, [0,1,3,2])
        else:
            raise ValueError()

        self.v_init_value=v_init_value
        self.h_init_value=h_init_value
        
        return v_init_value, h_init_value
   
    @staticmethod    
    def get_filter_weight(v, h, S_effective, scheme):
        '''given the h and v initial value in np array, return the filter 
        weights with the masking variable multiplied'''
        if scheme=='tai':
            v = tf.multiply(v, S_effective)
            h = h
        elif scheme=='microsoft':
            v = tf.multiply(S_effective, v)
            h = h
        else:
            raise ValueError()
            
        return v,h

    @staticmethod
    def get_stride(stride, scheme):
        '''given the stride from the argument of the convolution layer, 
        return the stride of the decomposed layers'''

        assert len(stride) == 2, 'expected square stride, got %s'%strides
        assert stride[0] == stride[1], 'expected square stride, got %s'%strides
        
        if scheme=='tai':
            mystride = stride[0]
            stride_v = (1,mystride)
            stride_h = (mystride,1)
        elif scheme=='microsoft':
            stride_v = stride
            stride_h = (1,1)
        else:
            raise ValueError()
            
        return stride_v, stride_h

    @staticmethod
    def get_K_from_decomposed_weight(v_init_value, h_init_value, scheme):
        '''given the np value of the decomposed weight, get the K value of the original weight'''
        assert v_init_value.ndim == 4,'v: %s, h: %s'%(v_init_value.shape, h_init_value.shape)
        assert h_init_value.ndim == 4,'v: %s, h: %s'%(v_init_value.shape, h_init_value.shape)

        if scheme == 'tai' or scheme == 'microsoft':
            K = v_init_value.shape[3]
            assert K == h_init_value.shape[2], 'v: %s, h: %s'%(v_init_value.shape, h_init_value.shape)
        else:
            raise ValueError()

        return K

    def num_operation_per_K(self,inputs, arguments):
        '''compute the number of operations in a conv2d'''

        rate = arguments['rate']
        stride = arguments['stride']
        padding=arguments['padding']

        v, h = self.get_initial_values()
        stride_v,stride_h = DecompositionScheme.get_stride(stride, self.scheme)
        _,K = self.get_weight_2d()

        self.original_op,_ = DnnUtili.num_operation_conv2d(DnnUtili.get_shape(inputs), DnnUtili.get_shape(self.weight_4d), stride, padding, rate)

        num_op_1, output1_size = DnnUtili.num_operation_conv2d(DnnUtili.get_shape(inputs), DnnUtili.get_shape(v), stride_v, padding, rate)
        num_op_2, _ = DnnUtili.num_operation_conv2d(output1_size, DnnUtili.get_shape(h), stride_h, padding, rate)
        self.num_op_per_K = float(num_op_1 + num_op_2)/K

        scale_factor = 1e-10

        return self.num_op_per_K*scale_factor, self.original_op*scale_factor

#################################################################
@add_arg_scope
def conv2d_mask(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    """
    Adds an N-D convolution followed by an optional batch_norm layer.

    It is required that 1 <= N <= 3.

    `convolution` creates a variable called `weights`, representing the
    convolutional kernel, that is convolved (actually cross-correlated) with the
    `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
    provided (such as `batch_norm`), it is then applied. Otherwise, if
    `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
    variable would be created and added the activations. Finally, if
    `activation_fn` is not `None`, it is applied to the activations as well.

    Performs a'trous convolution with input stride/dilation rate equal to `rate`
    if a value > 1 for any dimension of `rate` is specified.  In this case
    `stride` values != 1 are not supported.

    Args:
    inputs: a Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: integer, the number of output filters.
    kernel_size: a sequence of N positive integers specifying the spatial
      dimensions of of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: a sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: one of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
      N=3, currently the only valid value is "NDHWC".
    rate: a sequence of N positive integers specifying the dilation rate to use
      for a'trous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

    Returns:
    a tensor representing the output of the operation.

    Raises:
    ValueError: if `data_format` is invalid.
    ValueError: both 'rate' and `stride` are not uniformly 1.
    """

    #has to be at the very beginning of the function, otherwise the local variables will be included too
    arguments = locals()

    assert data_format is None or data_format == 'NHWC'

    if FLAGS.eval_fine_tuned_decomposition:
        assert FLAGS.is_training is False
        return conv2d_mask_eval_fine_tuned(**arguments)
    else:
        return conv2d_svd_mask(**arguments)

#################################################################
#@add_arg_scope
def conv2d_svd_mask(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
    raise ValueError('Invalid data_format: %r' % (data_format,))
  with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    input_rank = inputs.get_shape().ndims
    if input_rank is None:
      raise ValueError('Rank of inputs must be known')
    if input_rank < 3 or input_rank > 5:
      raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                       input_rank)
    conv_dims = input_rank - 2
    kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
    stride = utils.n_positive_integers(conv_dims, stride)
    rate = utils.n_positive_integers(conv_dims, rate)

    if data_format is None or data_format.endswith('C'):
      num_input_channels = inputs.get_shape()[input_rank - 1].value
    elif data_format.startswith('NC'):
      num_input_channels = inputs.get_shape()[1].value
    else:
      raise ValueError('Invalid data_format')

    if num_input_channels is None:
      raise ValueError('Number of in_channels must be known.')

    weights_shape = (
        list(kernel_size) + [num_input_channels, num_outputs])
    weights_collections = utils.get_variable_collections(variables_collections,
                                                         'weights')
    #Chong added
    #compute memory and computation cost
    #has to be at the very beginning of the function, otherwise the local variables will be included too
    arguments = dict(locals())
    #import mvm
    mvm = DnnUtili.mvm

    #original weights names in the un-decomopsed layer
    original_weights_name = '%s/weights'%(sc.name)
    weights_value = DnnUtili.get_tensor_from_checkpoint(original_weights_name)

    try:
        original_biases_name = '%s/biases'%(sc.name)
        biases_value = DnnUtili.get_tensor_from_checkpoint(original_biases_name)
    except:
        #if original weight is succesfully read from checkpoint, but bias is not, probably because no bias was used
        #this is an identity function, which addes no bias
        #print('my_slim_layer: no bias added in layer %s'%sc.name)
        if normalizer_fn == None:
            normalizer_fn=lambda arg: arg
     
    #initialize a DecompositionScheme object
    ds = DecompositionScheme(weights_value, scheme=FLAGS.decomposition_scheme, name=sc.name)

    #compute the computation cost and memory cost
    num_computation_per_K, original_computation_cost = ds.num_operation_per_K(inputs,arguments)
    filter_size = kernel_size + (num_input_channels, num_outputs)
    num_memory_per_K, original_memory_cost = DnnUtili.num_memory_per_K(filter_size, decomposition=FLAGS.decomposition_scheme)

    W,K = ds.get_weight_2d()

    #do the svd
    P,S,V = ds.svd() 

    #create the mask for the singular values
    #by default, mask variable is zero, meaning this singular component is used
    #mask variable being one meaning the singular component is not used'
    reduced_index = mvm.compute_reduced_index(S,op_name='%s/S_masks'%sc.name)
    assert reduced_index[0] >= 0, 'reduced_index error: %s, K: %d, %s'%(str(reduced_index), K, sc.name)
    assert reduced_index[1] <= K, 'reduced_index error: %s, K: %d, %s'%(str(reduced_index), K, sc.name)
    #use a placeholder, so everytime the network is used a feed_dict has to be provided
    #top part, fixed to 0 (include)
    S_top = np.full(fill_value=0.0, shape= [reduced_index[0]], dtype=np.float32)
    #bottom part, fixed to 1 (excluded)
    S_bottom = np.full(fill_value=1.0, shape = [K-reduced_index[1]], dtype=np.float32)

    #the free variables
    if FLAGS.is_training:
        if not FLAGS.add_and_svd_rounding:
            #if is training, need to provide mask value here for some reason, feed_dict does not work
            #need to run a evaluation script first, to dump the pickle file containing name_value_dict
            assert FLAGS.train_feed_dict_path is not None, 'if training, then need to provide the train_feed_dict_path'
            assert os.path.isfile(FLAGS.train_feed_dict_path), 'FLAGS.train_feed_dict_path not valid %s'%FLAGS.train_feed_dict_path
            with open(FLAGS.train_feed_dict_path, "rb") as f:
                #the name_value_dict contains un-duplicated copy of mask variable values
                name_value_dict = pickle.load(f)

            assert (sc.name)+'/S_masks_reduced' in name_value_dict, 'name: %s, keys: %s'%(sc.name+'/S_masks_reduced' , name_value_dict.keys())
            default_value = name_value_dict[(sc.name)+'/S_masks_reduced']
            assert list(default_value.shape) == list([reduced_index[1]-reduced_index[0]]), 'layer: %s, default_value: %s, reduced_index: %s'%(sc.name, default_value.shape, str(reduced_index))
            reduced_S_masks = default_value.astype(np.float32)
        #if using add_and_svd, the mask variables will be calculated using the sqp_solution, there is no need to load reduced_S_masks here
    else:
        #not training, then register mask variable with mvm, a feed_dict will be provided for evaluation
        reduced_S_masks = tf.placeholder(name='S_masks_reduced',
                                           dtype=tf.float32,
                                           shape=[reduced_index[1]-reduced_index[0]])

    if not FLAGS.add_and_svd_rounding:
        S_masks = tf.concat([S_top, reduced_S_masks, S_bottom], axis=0, name='S_masks')
        assert S_masks.get_shape().as_list()[0] == K 

        #if a singular value is not used, then it is masked by multiplying with zero
        S_effective = tf.subtract(tf.ones(shape=[K], dtype=S.dtype), S_masks)

        #add the declared masking variable to the masking variable manager declared in the network file
        mvm.add_variable(S_masks, singular_values=S, flattened_shape=W.shape, 
                reduced_index=reduced_index, reduced_variable=reduced_S_masks, decomposition_scheme=ds)
        v_init_value, h_init_value = ds.get_initial_values()
    else:
        #if choose to do add_and_svd rounding
        #load the non-integer mask variable solution for all the mask_variables in the network
        assert FLAGS.solution_path is not None
        assert str(FLAGS.solution_path).endswith('.mat'), 'FLAGS.solution_path is %s'%FLAGS.solution_path
        assert os.path.isfile(FLAGS.solution_path)
        non_integer_solution = np.squeeze(scipy.io.loadmat(FLAGS.solution_path)['x'])

        assert mvm.add_and_svd_index >=0
        #get the mask variables for this layer only, 0 means included and 1 means excluded
        layer_mask_variable = non_integer_solution[mvm.add_and_svd_index: mvm.add_and_svd_index+reduced_index[1]-reduced_index[0]]

        #increment the index pointer
        mvm.add_and_svd_index += layer_mask_variable.shape[0] 
        #have to do this because if mutiple tower are used, add_and_svd_index will overflow (larger than the size of non_integer_solution)
        mvm.add_and_svd_index = mvm.add_and_svd_index%(non_integer_solution.size)

        S_masks = np.concatenate((S_top, layer_mask_variable, S_bottom), axis=0)
        assert S_masks.shape[0] == K, 'solution read from FLAGS.solution_path does not match with reduced_index.pickle'

        S_effective = np.ones(shape=[K], dtype=np.float32) - S_masks
        assert S_effective.shape[0] == K
        assert S_effective.ndim == 1

        S_matrix = np.zeros_like(W) 
        S_matrix[:K, :K] = np.diag(S_effective)

        #the W matrix reconstructed using the 
        W_sqp = P@S_matrix@V

        #update the ds
        ds.update_weight_2d(W_sqp)
        P, S, V = ds.svd()

        if FLAGS.load_add_and_svd_K:
            #DEBUG
            #Not sure if should add this
            #assert not FLAGS.cost_saturation, 'should not set cost_saturation if loading add_and_svd_K'
            #load add_and_svd_K from pickle
            assert os.path.isfile('/tmp/add_and_svd_K.pickle'),'FLAGS.load_add_and_svd_K is on but /tmp/add_and_svd_K.pickle is not found'
            with open('/tmp/add_and_svd_K.pickle','rb') as f:
                add_and_svd_K_dict = pickle.load(f)
                assert sc.name in add_and_svd_K_dict,'layer %s not found in add_and_svd_K %s'%(sc.name, str(list(add_and_svd_K_dict.keys())))

                #in the dict is add_and_svd_K, num_computation_per_K, original_computation_cost,num_memory_per_K, original_memory_cost]
                add_and_svd_K = add_and_svd_K_dict[sc.name][0]
        else:     
            #do random rounding and add to mvm
            add_and_svd_K = layer_mask_variable.shape[0] - sum(layer_mask_variable)
            add_and_svd_K = int(add_and_svd_K) + (0.99*random.random() < add_and_svd_K - int(add_and_svd_K))
            #add_and_svd_K = int(add_and_svd_K) + (random.random() < add_and_svd_K - int(add_and_svd_K))

            #DEBUG
            #add_and_svd_K = layer_mask_variable.shape[0] - math.ceil(sum(layer_mask_variable))

            if FLAGS.cost_saturation:
                number_of_singular_values_used = reduced_index[0] + add_and_svd_K
                if number_of_singular_values_used*num_computation_per_K >= original_computation_cost or number_of_singular_values_used*num_memory_per_K >= original_memory_cost:
                    add_and_svd_K = reduced_index[1] - reduced_index[0]
                    #DEBUG
                    #print('my_slim_layer: %s mask set to all due to cost saturation'%sc.name)

            mvm.add_and_svd_K(sc.name, (add_and_svd_K, num_computation_per_K, original_computation_cost,num_memory_per_K, original_memory_cost, add_and_svd_K+reduced_index[0]))

        assert 0 <= add_and_svd_K <= reduced_index[1] - reduced_index[0]

        v_init_value, h_init_value = ds.get_initial_values(add_and_svd_K=add_and_svd_K + reduced_index[0])

    if FLAGS.is_training:
        #if is training, create variables v and h for the decomposed weights
        v = variables.model_variable('weights_v', 
                                           #shape=v_init_value.shape,
                                           dtype=dtype,
                                           #initializer=weights_initializer,
                                           initializer=v_init_value,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        h = variables.model_variable('weights_h', 
                                           #shape=h_init_value.shape,
                                           dtype=dtype,
                                           #initializer=weights_initializer,
                                           initializer=h_init_value,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
    else:
        v,h = v_init_value, h_init_value

    if not FLAGS.add_and_svd_rounding:
        #if not doing add_and_svd, then need to multiply the v and h weight with the masking variables
        v,h = DecompositionScheme.get_filter_weight(v, h, S_effective, FLAGS.decomposition_scheme)

    stride_v,stride_h = DecompositionScheme.get_stride(stride, FLAGS.decomposition_scheme)

    #call the convolution layers

    #DEBUG
    #print('my_slim_layer: input %s'%inputs.shape)
    outputs = nn.convolution(input=inputs,
                             filter=v,
                             dilation_rate=rate,
                             strides=stride_v,
                             padding=padding,
                             data_format=data_format)
    #print('my_slim_layer: outputs1 %s'%outputs.shape)
    outputs = nn.convolution(input=outputs,
                             filter=h,
                             dilation_rate=rate,
                             strides=stride_h,
                             padding=padding,
                             data_format=data_format)
    #print('my_slim_layer: outputs2 %s'%outputs.shape)

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        if FLAGS.is_training:
            #if training, add bias variable
            biases_collections = utils.get_variable_collections(
                variables_collections, 'biases')
            biases = variables.model_variable('biases',
                                              #shape=[num_outputs],
                                              dtype=dtype,
                                              #initializer=biases_initializer,
                                              initializer=biases_value,
                                              regularizer=biases_regularizer,
                                              collections=biases_collections,
                                              trainable=trainable)
        else:
            #if not training just use the numpy value
            biases = biases_value
        outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    
    if not FLAGS.add_and_svd_rounding:
        #register the amount of memory and computation per singular component for this layer
        mvm.add_num_computation(S_masks, num_computation_per_K)
        mvm.add_total_computation(S_masks, original_computation_cost)
        mvm.add_num_memory(S_masks, num_memory_per_K)
        mvm.add_total_memory(S_masks, original_memory_cost)

    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


#################################################################
#@add_arg_scope
def conv2d_mask_eval_fine_tuned(inputs,
                num_outputs,
                kernel_size,
                trainable=True,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                scope=None):
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
    raise ValueError('Invalid data_format: %r' % (data_format,))
  with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    input_rank = inputs.get_shape().ndims
    if input_rank is None:
      raise ValueError('Rank of inputs must be known')
    if input_rank < 3 or input_rank > 5:
      raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                       input_rank)
    conv_dims = input_rank - 2
    kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
    stride = utils.n_positive_integers(conv_dims, stride)
    rate = utils.n_positive_integers(conv_dims, rate)

    if data_format is None or data_format.endswith('C'):
      num_input_channels = inputs.get_shape()[input_rank - 1].value
    elif data_format.startswith('NC'):
      num_input_channels = inputs.get_shape()[1].value
    else:
      raise ValueError('Invalid data_format')

    if num_input_channels is None:
      raise ValueError('Number of in_channels must be known.')

    weights_shape = (
        list(kernel_size) + [num_input_channels, num_outputs])
    weights_collections = utils.get_variable_collections(variables_collections,
                                                         'weights')
    #Chong added
    assert FLAGS.eval_fine_tuned_decomposition, 'this layer is for evaluating the performance of a decomposed layer that has been fine tuned'
    assert FLAGS.is_training is False
    #import mvm
    mvm = DnnUtili.mvm

    #weights names in the decomopsed layer
    original_weights_name_v = '%s/weights_v'%(sc.name)
    original_weights_name_h = '%s/weights_h'%(sc.name)
    weights_value_v = DnnUtili.get_tensor_from_checkpoint(original_weights_name_v)
    weights_value_h = DnnUtili.get_tensor_from_checkpoint(original_weights_name_h)

    try:
        original_biases_name = '%s/biases'%(sc.name)
        biases_value = DnnUtili.get_tensor_from_checkpoint(original_biases_name)
    except:
        #if original weight is succesfully read from checkpoint, but bias is not, probably because no bias was used
        #this is an identity function, which addes no bias
        print('my_slim_layer: no bias added in layer %s'%sc.name)
        normalizer_fn=lambda arg: arg

    K = DecompositionScheme.get_K_from_decomposed_weight(weights_value_v, weights_value_h, FLAGS.decomposition_scheme)
    #only using the shape of the S
    S = np.full(fill_value=np.NaN, shape=[K], dtype= np.float32)

    if not FLAGS.add_and_svd_rounding:
        #create the mask for the singular values
        #by default, mask variable is zero, meaning this singular component is used
        #mask variable being one meaning the singular component is not used'
        reduced_index = mvm.compute_reduced_index(S,op_name='%s/S_masks'%sc.name)

        #top part, fixed to 0 (include)
        S_top = np.full(fill_value=0.0, shape=[reduced_index[0]], dtype=np.float32)
        
        #need to run a evaluation script first, to dump the pickle file containing name_value_dict
        assert FLAGS.train_feed_dict_path is not None
        assert os.path.isfile(FLAGS.train_feed_dict_path), 'FLAGS.train_feed_dict_path not valid %s'%FLAGS.train_feed_dict_path
        with open(FLAGS.train_feed_dict_path, "rb") as f:
            #the name_value_dict contains un-duplicated copy of mask variable values
            name_value_dict = pickle.load(f)

        assert (sc.name)+'/S_masks_reduced' in name_value_dict, 'name: %s, keys: %s'%(sc.name+'/S_masks_reduced' , name_value_dict.keys())
        default_value = name_value_dict[(sc.name)+'/S_masks_reduced']
        assert list(default_value.shape) == list([reduced_index[1]-reduced_index[0]]), 'default value loaded from train_feed_dict_path does not match'

        #bottom part, fixed to 1 (excluded)
        S_bottom = np.full(fill_value=1.0, shape = [K-reduced_index[1]], dtype=np.float32)

        S_masks = tf.concat([S_top, tf.constant(default_value,dtype=tf.float32), S_bottom], axis=0, name='S_masks')
        assert S_masks.get_shape().as_list()[0] == K 

        #if a singular value is not used, then it is masked by multiplying with zero
        S_effective = tf.subtract(tf.ones(shape=[K], dtype=tf.float32), S_masks ) 

        #DEBUG
        #print('name: %s, reduced_index: %s'%(sc.name, str(reduced_index)))
        #print(default_value)

        #DEBUG
        #S_effective = tf.ones(shape=S_effective.shape)
    
        #multiply with the masking variables
        v,h = DecompositionScheme.get_filter_weight(weights_value_v, weights_value_h, S_effective, FLAGS.decomposition_scheme)
    else:
        #in case of add_and_svd_rounding
        v,h = weights_value_v, weights_value_h

    stride_v,stride_h = DecompositionScheme.get_stride(stride, FLAGS.decomposition_scheme)

    #call the convolution layers
    outputs = nn.convolution(input=inputs,
                             filter=v,
                             dilation_rate=rate,
                             strides=stride_v,
                             padding=padding,
                             data_format=data_format)
    outputs = nn.convolution(input=outputs,
                             filter=h,
                             dilation_rate=rate,
                             strides=stride_h,
                             padding=padding,
                             data_format=data_format)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases = biases_value
        outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)
############################################################################
DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'

@add_arg_scope
def my_separable_conv2d(
    inputs,
    num_outputs,
    kernel_size,
    depth_multiplier,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NHWC,
    rate=1,
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a depth-separable 2D convolution with optional batch_norm layer.
  This op first performs a depthwise convolution that acts separately on
  channels, creating a variable called `depthwise_weights`. If `num_outputs`
  is not None, it adds a pointwise convolution that mixes channels, creating a
  variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
  it adds bias to the result, creating a variable called 'biases', otherwise,
  the `normalizer_fn` is applied. It finally applies an activation function
  to produce the end result.
  Args:
    inputs: A tensor of size [batch_size, height, width, channels].
    num_outputs: The number of pointwise convolution output filters. If is
      None, then we skip the pointwise convolution stage.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    rate: A list of length 2: [rate_height, rate_width], specifying the dilation
      rates for atrous convolution. Can be an int if both rates are the same.
      If any value is larger than one, then both stride values need to be one.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is invalid.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  layer_variable_getter = _build_variable_getter({
      'bias': 'biases',
      'depthwise_kernel': 'depthwise_weights',
      'pointwise_kernel': 'pointwise_weights'
  })
  
  #Chong added
  kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
  #num_filters_in = utils.channel_dimension(
  #        inputs.get_shape(), df, min_rank=4)

  image_size = DnnUtili.get_shape(inputs)
  depthwise_filter_size = [kernel_h, kernel_w, image_size[3], depth_multiplier]
  
  depthwise_comp,pointwise_comp,_ = DnnUtili.num_operation_conv2d(image_size, 
    depthwise_filter_size, stride, padding, 1, True, num_outputs)
  
  depthwise_mem, pointwise_mem = DnnUtili.num_memory_separable_conv2d(image_size, 
      depthwise_filter_size, num_outputs)
  
  DnnUtili.mvm.variable_to_sep_info[scope] = np.array([depthwise_comp,pointwise_comp,
                      depthwise_mem, pointwise_mem])
  print('%s\t%.3E\t%.3E\t%.3E\t%.3E\t'%(scope,depthwise_comp,pointwise_comp,depthwise_mem, pointwise_mem))
  #Chong added end

  with variable_scope.variable_scope(
      scope,
      'SeparableConv2d', [inputs],
      reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    df = ('channels_first'
          if data_format and data_format.startswith('NC') else 'channels_last')
    if num_outputs is not None:
      # Apply separable conv using the SeparableConvolution2D layer.
      layer = convolutional_layers.SeparableConvolution2D(
          filters=num_outputs,
          kernel_size=kernel_size,
          strides=stride,
          padding=padding,
          data_format=df,
          dilation_rate=utils.two_element_tuple(rate),
          activation=None,
          depth_multiplier=depth_multiplier,
          use_bias=not normalizer_fn and biases_initializer,
          depthwise_initializer=weights_initializer,
          pointwise_initializer=weights_initializer,
          bias_initializer=biases_initializer,
          depthwise_regularizer=weights_regularizer,
          pointwise_regularizer=weights_regularizer,
          bias_regularizer=biases_regularizer,
          activity_regularizer=None,
          trainable=trainable,
          name=sc.name,
          dtype=inputs.dtype.base_dtype,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs)

      # Add variables to collections.
      _add_variable_to_collections(layer.depthwise_kernel,
                                   variables_collections, 'weights')
      _add_variable_to_collections(layer.pointwise_kernel,
                                   variables_collections, 'weights')
      if layer.bias is not None:
        _add_variable_to_collections(layer.bias, variables_collections,
                                     'biases')

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      # Actually apply depthwise conv instead of separable conv.
      dtype = inputs.dtype.base_dtype
      kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
      stride_h, stride_w = utils.two_element_tuple(stride)
      weights_collections = utils.get_variable_collections(
          variables_collections, 'weights')

      image_size = DnnUtili.get_shape(inputs)
      depthwise_filter_size = [kernel_h, kernel_w, image_size[3], depth_multiplier]

      depthwise_weights = variables.model_variable(
          'depthwise_weights',
          shape=depthwise_shape,
          dtype=dtype,
          initializer=weights_initializer,
          regularizer=weights_regularizer,
          trainable=trainable,
          collections=weights_collections)
      strides = [1, 1, stride_h,
                 stride_w] if data_format.startswith('NC') else [
                     1, stride_h, stride_w, 1
                 ]

      outputs = nn.depthwise_conv2d(
          inputs,
          depthwise_weights,
          strides,
          padding,
          rate=utils.two_element_tuple(rate),
          data_format=data_format)
      num_outputs = depth_multiplier * num_filters_in

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
      else:
        if biases_initializer is not None:
          biases_collections = utils.get_variable_collections(
              variables_collections, 'biases')
          biases = variables.model_variable(
              'biases',
              shape=[
                  num_outputs,
              ],
              dtype=dtype,
              initializer=biases_initializer,
              regularizer=biases_regularizer,
              trainable=trainable,
              collections=biases_collections)
          outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


#############################################################################
def _sparse_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  inputs_rank = inputs.dense_shape.get_shape().as_list()[0]
  if inputs_rank < new_rank:
    raise ValueError(
        'Inputs has rank less than new_rank. {} must have rank at least'
        ' {}. Received rank {}, shape {}'.format(inputs, new_rank, inputs_rank,
                                                 inputs.get_shape()))

  outer_dimensions = inputs.dense_shape[:new_rank - 1]
  inner_dimensions = inputs.dense_shape[new_rank - 1:]
  new_shape = array_ops.concat(
      (outer_dimensions, [math_ops.reduce_prod(inner_dimensions)]), 0)
  flattened = sparse_ops.sparse_reshape(inputs, new_shape)
  return flattened


def _dense_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  rank_assertion = check_ops.assert_rank_at_least(
      inputs, new_rank, message='inputs has rank less than new_rank')
  with ops.control_dependencies([rank_assertion]):
    outer_dimensions = array_ops.strided_slice(
        array_ops.shape(inputs), [0], [new_rank - 1])
    new_shape = array_ops.concat((outer_dimensions, [-1]), 0)
    reshaped = array_ops.reshape(inputs, new_shape)

  # if `new_rank` is an integer, try to calculate new shape.
  if isinstance(new_rank, six.integer_types):
    static_shape = inputs.get_shape()
    if static_shape is not None and static_shape.dims is not None:
      static_shape = static_shape.as_list()
      static_outer_dims = static_shape[:new_rank - 1]
      static_inner_dims = static_shape[new_rank - 1:]
      flattened_dimension = 1
      for inner_dim in static_inner_dims:
        if inner_dim is None:
          flattened_dimension = None
          break
        flattened_dimension *= inner_dim
      reshaped.set_shape(static_outer_dims + [flattened_dimension])
  return reshaped


@add_arg_scope
def _inner_flatten(inputs, new_rank, output_collections=None, scope=None):
  """Flattens inner dimensions of `inputs`, returns a Tensor with `new_rank`.
  For example:
  '''
      x = tf.random_uniform(shape=[1, 2, 3, 4, 5, 6])
      y = _inner_flatten(x, 4)
      assert y.get_shape().as_list() == [1, 2, 3, (4 * 5 * 6)]
  '''
  This layer will fail at run time if `new_rank` is greater than the current
  rank of `inputs`.
  Args:
    inputs: A `Tensor` or `SparseTensor`.
    new_rank: The desired rank of the returned `Tensor` or `SparseTensor`.
    output_collections: Collection to which the outputs will be added.
    scope: Optional scope for `name_scope`.
  Returns:
    A `Tensor` or `SparseTensor` conataining the same values as `inputs`, but
    with innermost dimensions flattened to obtain rank `new_rank`.
  Raises:
    TypeError: `inputs` is not a `Tensor` or `SparseTensor`.
  """
  with ops.name_scope(scope, 'InnerFlatten', [inputs, new_rank]) as sc:
    if isinstance(inputs, sparse_tensor.SparseTensor):
      flattened = _sparse_inner_flatten(inputs, new_rank)
    else:
      inputs = ops.convert_to_tensor(inputs)
      flattened = _dense_inner_flatten(inputs, new_rank)
  return utils.collect_named_outputs(output_collections, sc, flattened)


def _model_variable_getter(getter,
                           name,
                           shape=None,
                           dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           rename=None,
                           use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      collections=collections,
      trainable=trainable,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=getter,
      use_resource=use_resource)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""

  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)

  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = utils.get_variable_collections(collections_set,
                                               collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)
        
#@add_arg_scope
#def separable_convolution2d(
#    inputs,
#    num_outputs,
#    kernel_size,
#    depth_multiplier,
#    stride=1,
#    padding='SAME',
#    data_format=DATA_FORMAT_NHWC,
#    rate=1,
#    activation_fn=nn.relu,
#    normalizer_fn=None,
#    normalizer_params=None,
#    weights_initializer=initializers.xavier_initializer(),
#    weights_regularizer=None,
#    biases_initializer=init_ops.zeros_initializer(),
#    biases_regularizer=None,
#    reuse=None,
#    variables_collections=None,
#    outputs_collections=None,
#    trainable=True,
#    scope=None):
#  """Adds a depth-separable 2D convolution with optional batch_norm layer.
#  This op first performs a depthwise convolution that acts separately on
#  channels, creating a variable called `depthwise_weights`. If `num_outputs`
#  is not None, it adds a pointwise convolution that mixes channels, creating a
#  variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
#  it adds bias to the result, creating a variable called 'biases', otherwise,
#  the `normalizer_fn` is applied. It finally applies an activation function
#  to produce the end result.
#  Args:
#    inputs: A tensor of size [batch_size, height, width, channels].
#    num_outputs: The number of pointwise convolution output filters. If is
#      None, then we skip the pointwise convolution stage.
#    kernel_size: A list of length 2: [kernel_height, kernel_width] of
#      of the filters. Can be an int if both values are the same.
#    depth_multiplier: The number of depthwise convolution output channels for
#      each input channel. The total number of depthwise convolution output
#      channels will be equal to `num_filters_in * depth_multiplier`.
#    stride: A list of length 2: [stride_height, stride_width], specifying the
#      depthwise convolution stride. Can be an int if both strides are the same.
#    padding: One of 'VALID' or 'SAME'.
#    data_format: A string. `NHWC` (default) and `NCHW` are supported.
#    rate: A list of length 2: [rate_height, rate_width], specifying the dilation
#      rates for atrous convolution. Can be an int if both rates are the same.
#      If any value is larger than one, then both stride values need to be one.
#    activation_fn: Activation function. The default value is a ReLU function.
#      Explicitly set it to None to skip it and maintain a linear activation.
#    normalizer_fn: Normalization function to use instead of `biases`. If
#      `normalizer_fn` is provided then `biases_initializer` and
#      `biases_regularizer` are ignored and `biases` are not created nor added.
#      default set to None for no normalizer function
#    normalizer_params: Normalization function parameters.
#    weights_initializer: An initializer for the weights.
#    weights_regularizer: Optional regularizer for the weights.
#    biases_initializer: An initializer for the biases. If None skip biases.
#    biases_regularizer: Optional regularizer for the biases.
#    reuse: Whether or not the layer and its variables should be reused. To be
#      able to reuse the layer scope must be given.
#    variables_collections: Optional list of collections for all the variables or
#      a dictionary containing a different list of collection per variable.
#    outputs_collections: Collection to add the outputs.
#    trainable: Whether or not the variables should be trainable or not.
#    scope: Optional scope for variable_scope.
#  Returns:
#    A `Tensor` representing the output of the operation.
#  Raises:
#    ValueError: If `data_format` is invalid.
#  """
#  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
#    raise ValueError('data_format has to be either NCHW or NHWC.')
#  layer_variable_getter = _build_variable_getter({
#      'bias': 'biases',
#      'depthwise_kernel': 'depthwise_weights',
#      'pointwise_kernel': 'pointwise_weights'
#  })
#
#  with variable_scope.variable_scope(
#      scope,
#      'SeparableConv2d', [inputs],
#      reuse=reuse,
#      custom_getter=layer_variable_getter) as sc:
#    inputs = ops.convert_to_tensor(inputs)
#
#    df = ('channels_first'
#          if data_format and data_format.startswith('NC') else 'channels_last')
#    if num_outputs is not None:
#      # Apply separable conv using the SeparableConvolution2D layer.
#      layer = convolutional_layers.SeparableConvolution2D(
#          filters=num_outputs,
#          kernel_size=kernel_size,
#          strides=stride,
#          padding=padding,
#          data_format=df,
#          dilation_rate=utils.two_element_tuple(rate),
#          activation=None,
#          depth_multiplier=depth_multiplier,
#          use_bias=not normalizer_fn and biases_initializer,
#          depthwise_initializer=weights_initializer,
#          pointwise_initializer=weights_initializer,
#          bias_initializer=biases_initializer,
#          depthwise_regularizer=weights_regularizer,
#          pointwise_regularizer=weights_regularizer,
#          bias_regularizer=biases_regularizer,
#          activity_regularizer=None,
#          trainable=trainable,
#          name=sc.name,
#          dtype=inputs.dtype.base_dtype,
#          _scope=sc,
#          _reuse=reuse)
#      outputs = layer.apply(inputs)
#
#      # Add variables to collections.
#      _add_variable_to_collections(layer.depthwise_kernel,
#                                   variables_collections, 'weights')
#      _add_variable_to_collections(layer.pointwise_kernel,
#                                   variables_collections, 'weights')
#      if layer.bias is not None:
#        _add_variable_to_collections(layer.bias, variables_collections,
#                                     'biases')
#
#      if normalizer_fn is not None:
#        normalizer_params = normalizer_params or {}
#        outputs = normalizer_fn(outputs, **normalizer_params)
#    else:
#      # Actually apply depthwise conv instead of separable conv.
#      dtype = inputs.dtype.base_dtype
#      kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
#      stride_h, stride_w = utils.two_element_tuple(stride)
#      num_filters_in = utils.channel_dimension(
#          inputs.get_shape(), df, min_rank=4)
#      weights_collections = utils.get_variable_collections(
#          variables_collections, 'weights')
#
#      depthwise_shape = [kernel_h, kernel_w, num_filters_in, depth_multiplier]
#      depthwise_weights = variables.model_variable(
#          'depthwise_weights',
#          shape=depthwise_shape,
#          dtype=dtype,
#          initializer=weights_initializer,
#          regularizer=weights_regularizer,
#          trainable=trainable,
#          collections=weights_collections)
#      strides = [1, 1, stride_h,
#                 stride_w] if data_format.startswith('NC') else [
#                     1, stride_h, stride_w, 1
#                 ]
#
#      outputs = nn.depthwise_conv2d(
#          inputs,
#          depthwise_weights,
#          strides,
#          padding,
#          rate=utils.two_element_tuple(rate),
#          data_format=data_format)
#      num_outputs = depth_multiplier * num_filters_in
#
#      if normalizer_fn is not None:
#        normalizer_params = normalizer_params or {}
#        outputs = normalizer_fn(outputs, **normalizer_params)
#      else:
#        if biases_initializer is not None:
#          biases_collections = utils.get_variable_collections(
#              variables_collections, 'biases')
#          biases = variables.model_variable(
#              'biases',
#              shape=[
#                  num_outputs,
#              ],
#              dtype=dtype,
#              initializer=biases_initializer,
#              regularizer=biases_regularizer,
#              trainable=trainable,
#              collections=biases_collections)
#          outputs = nn.bias_add(outputs, biases, data_format=data_format)
#
#    if activation_fn is not None:
#      outputs = activation_fn(outputs)
#    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)



############################################################################
#the original convolution layer in slim
@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
  """Adds an N-D convolution followed by an optional batch_norm layer.

  It is required that 1 <= N <= 3.

  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.

  Performs a'trous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: a Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: integer, the number of output filters.
    kernel_size: a sequence of N positive integers specifying the spatial
      dimensions of of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: a sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: one of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
      N=3, currently the only valid value is "NDHWC".
    rate: a sequence of N positive integers specifying the dilation rate to use
      for a'trous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if `data_format` is invalid.
    ValueError: both 'rate' and `stride` are not uniformly 1.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
    raise ValueError('Invalid data_format: %r' % (data_format,))
  with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    input_rank = inputs.get_shape().ndims
    if input_rank is None:
      raise ValueError('Rank of inputs must be known')
    if input_rank < 3 or input_rank > 5:
      raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                       input_rank)
    conv_dims = input_rank - 2
    kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
    stride = utils.n_positive_integers(conv_dims, stride)
    rate = utils.n_positive_integers(conv_dims, rate)

    if data_format is None or data_format.endswith('C'):
      num_input_channels = inputs.get_shape()[input_rank - 1].value
    elif data_format.startswith('NC'):
      num_input_channels = inputs.get_shape()[1].value
    else:
      raise ValueError('Invalid data_format')

    if num_input_channels is None:
      raise ValueError('Number of in_channels must be known.')

    weights_shape = (
        list(kernel_size) + [num_input_channels, num_outputs])
    weights_collections = utils.get_variable_collections(variables_collections,
                                                         'weights')
    weights = variables.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    outputs = nn.convolution(input=inputs,
                             filter=weights,
                             dilation_rate=rate,
                             strides=stride,
                             padding=padding,
                             data_format=data_format)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases, data_format=data_format)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)
