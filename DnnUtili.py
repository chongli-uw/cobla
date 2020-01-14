import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util

#from keras import backend as K

import numpy as np
import numpy.random
import scipy
import scipy.io
import scipy.linalg
import statistics
import random
import copy
import re

import sklearn.decomposition
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import fast_dot
#make sure sklearn.utils is using fast BLAS functions
import warnings
from sklearn.exceptions import NonBLASDotWarning
warnings.simplefilter('always', NonBLASDotWarning) 

import pydot_ng as pydot
import matplotlib.pyplot as plt
import matplotlib

import tempfile
import uuid
import os
import os.path
try:
    import matlab
    import matlab.engine
except ImportError:
    print('DnnUtili: cannot import matlab')
    pass
import pickle
import time
import math
import numbers
import bisect
import shutil

############################################################################
def tensor_in_checkpoint(variable_name, checkpoint_path = None):
    if checkpoint_path is None and FLAGS.checkpoint_path is None:
        raise ValueError('checkpoint_path nor reader is provided, or tf.train.latest_checkpoint failed.')
 
    if checkpoint_path is None:
        checkpoint_path = FLAGS.checkpoint_path

    reader = get_checkpoint_reader(checkpoint_path)

    return reader.has_tensor(variable_name)
############################################################################
def get_tensor_from_checkpoint(variable_name, checkpoint_path = None, reader = None):
    """Get a numpy tensor from checkpoint file"""
  
    if checkpoint_path is None and reader is None and FLAGS.checkpoint_path is None:
        raise ValueError('checkpoint_path nor reader is provided, or tf.train.latest_checkpoint failed.')
 
    if checkpoint_path is None:
        checkpoint_path = FLAGS.checkpoint_path

    #if no reader is provided
    if reader is None:
        reader = get_checkpoint_reader(checkpoint_path)

    if reader.has_tensor(variable_name) is False:
        #print('Checkpoint (%s) does not contain variable: %s, variables in checkpoint are %s'%(str(checkpoint_path),variable_name,reader.get_variable_to_shape_map().keys()))
        #print('Checkpoint (%s) does not contain variable: %s, '%(str(checkpoint_path),variable_name))
        raise ValueError('Checkpoint (%s) does not contain variable: %s, '%(str(checkpoint_path),variable_name))
        
    return reader.get_tensor(variable_name)

############################################################################
def get_checkpoint_reader(checkpoint_path = None):
    """Get a checkpoint reader object"""

    #use default checkpoint_path in FLAGS if none provided
    if checkpoint_path is None:
        checkpoint_path = FLAGS.checkpoint_path

    #if a path is provided, find the latest checkpiont in the path
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = latest_checkpoint(checkpoint_path)

    if checkpoint_path is None:
        raise ValueError('checkpoint_path is not provided, or tf.train.latest_checkpoint failed.')
    
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path )
            
    return reader

############################################################################
def split_comma_separated_string(css, expected_type = str):
    '''Given a comma separated string, 
    return a list containing the content which is casted to the expected type'''

    assert type(css) is str
    assert expected_type in [str, int] 

    result = [expected_type(item.strip()) for item in css.split(',')]
    return result

def to_comma_separated_string(from_list):
    '''
    Convert a list to a comma separated list
    '''
    result = ''
    for item in from_list:
        result += str(item)
        result += ','

    #remove the last ,
    return result[:-1]
############################################################################
def inspect_checkpoint_file(checkpoint_path = None, variable_name = None, variable_scope = None, save_path = None, print_rank = False):
    """Prints tensors in a checkpoint file.
    only consider weights variables, biases are ignored
    variable_name: name of request variable, use 'all' for all vaiables
    variable scope: the scope of the request variable
    save_path: path to which the variable values are saved in pickle and mat format
    print_rank: whether the tensor rank is calculated and saved
    """  
    
    #use default checkpoint_path in FLAGS if none provided
    if checkpoint_path is None:
        checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is None:
        raise ValueError('checkpoint_path is not provided')
    
    if variable_name is None and variable_scope is None:
        raise ValueError('Must specify variable name or scope')
        
    #the list of variables to be printed or saved to pickle file
    var_to_print = list()
      
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path )
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
            
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    #if a tensor_name is requested
    if variable_name not in [None, 'all']:
        #but not found
        if variable_name not in var_to_shape_map:
            print(list(var_to_shape_map.keys()))
            raise ValueError('Variable %s not in checkpoint'%variable_name)
        elif variable_name.endswith('biases'):
            raise ValueError('Bias variable %s not important'%variable_name)
        else:
            var_to_print.append(variable_name)
    
    if variable_name == 'all':
        var_to_print = list(var_to_shape_map.keys())
        #remove biases
    var_to_print = [v for v in var_to_print if not v.endswith('biases')]
    
    if variable_scope is not None:
            #check if any variable in the request scope is found
        valid_scope = False
        #add all the variables in the request scope
        for key in var_to_shape_map:
            #key.op.name
            if key.startswith(variable_scope) and not key.endswith('biases'):
                var_to_print.append(key)
                valid_scope = True
            
        if not valid_scope:
            raise ValueError('Cannot find any variable in scope %s'%scope_name)
            
    #maps variable name to a numpy array containing its value
    var_value_map = dict()
    var_rank_map = dict()
    
    #run in current session
    #    sess = tf.get_default_session()
    #    if sess is None:
    with tf.Session().as_default():
        sess=tf.get_default_session()
        
        #print variable information
        for var in var_to_print:
            print(var)
            var_value = reader.get_tensor(var)
            #print(var_value.shape)
            #var_23 = [v for v in tf.all_variables() if v.name == "Variable_23:0"][0]
            var_tf = tf.constant(var_value, dtype=tf.float32)
            var_value_map[var] = (var_value)
            
            if print_rank and var_value.ndim in [2,4]:
                assert var_value.ndim in [2,4], 'only support 2 or 4 dimension tensor'
                var_rank_map[var] = sess.run(tensor_nuclear_norm(var_tf))
                print('Rank is %f'%var_rank_map[var])
            
    if save_path is not None:#
        #check if I can write to the save_path
        assert os.access(save_path, os.W_OK), 'save_path not writeable'
        
        #save pickle
        with open(os.path.join(save_path,'model_variable.pickle'), 'wb') as f:
            pickle.dump((var_value_map, var_rank_map), f, protocol=-1)
        #save mat file
        import scipy.io
        scipy.io.savemat(os.path.join(save_path,'model_variable.mat'), mdict={'variable_value': var_value_map, 'variable_rank':var_rank_map})
        
    return var_value_map, var_rank_map
        
############################################################################
def myGradients(f, variable_list):
    #f is a function, variable_list is a list of one dimensional variables
    #return the gradient of f in respect to a vector which is formed by concatenating 
    #all the variables in variable_list together
    
    #check variable_list 
    assert type(variable_list) is list, 'variable_list is %s, must be list'%str(type(variable_list))
    for v in variable_list:
        v_shape = v.get_shape().as_list()
        assert len(v_shape) == 1, 'gradient to the variables is not a vector: %s'%str(v_shape)


    #compute the gradient to the variables
    gxy = tf.gradients(f, variable_list)
    #concate the gradients of each variable together, as if the variables in variable_list
    #are concatenated together first, and compute the gradient
    gp = tf.concat(gxy, axis=0)

    #ensure the gradient to the variables is a one diemensional vector
    gp_shape = gp.get_shape().as_list()
    assert len(gp_shape) == 1, 'gradient to the variables is not a vector: %s'%str(gp_shape)

    return gp

def myJacobian(f, variable_list):
    #f is a 1-d tensor (generally logits) of shape [num_classes]
    #variable_list is a list of variables
    #return the jacobian of shape [num_variables, num_classes]

    assert f.get_shape().ndims == 1, 'only support 1-d tensorof shape [num_classes], got %s'%f.get_shape()

    num_classes = f.get_shape().as_list()[0]

    columns = []
    for i in range(num_classes):
        col = myGradients(f[i], variable_list)
        #reshape to a column vector
        col = tf.reshape(col, [-1, 1])
        #add to list
        columns.append(col)

    jacob = tf.stack(columns, axis=1)
    jacob = tf.reshape(jacob, [-1, num_classes])

    return jacob

############################################################################
def myHessian(f, variable_list, gradient_vector = None):
    #f is a function, variable_list is a list of one dimensional variables
    #return the hessian of f in respect to a vector which is formed by concatenating 
    #all the variables in variable_list together
    #optionally, a gradient vector can be provided
   
    #if no gradient vector is provied, compute
    if gradient_vector is None:
        gp = myGradients(f, variable_list)
    else:
        #assert the gradient_vector is a vector
        assert len(gradient_vector.get_shape().as_list()) == 1, 'gradient_vector must be a vector, %s'%gradient_vector
        assert type(variable_list) is list, 'variable_list is %s, must be list'%str(type(variable_list))
        #ensure the length of the concatenated variable formed by the variables in the variable_list is the same as the gradient_vector
        cat_variable_length = 0
        for v in variable_list:
            cat_variable_length += v.get_shape().as_list()[0]
        assert cat_variable_length == gradient_vector.get_shape().as_list()[0], 'length of concateated variable is not the same as gradient_vector'

        gp = gradient_vector
       
    num_variables = gp.get_shape().as_list()[0]

    #initialize 
    hp = []
    
    for i in range(num_variables):
        column = tf.gradients(gp[i], variable_list)
        column = tf.concat(column, axis=0)
        
        hp.append(column)
    
    hp = tf.stack(hp)
    return hp
  
def hessian_vector_product(ys, xs, v):
    """
    copied from 
    /usr/lib/python3.5/site-packages/tensorflow/python/ops/gradients_imply.py
    Multiply the Hessian of `ys` wrt `xs` by `v`.

    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.

    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.

    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.

    Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.

    Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.

    Raises:
    ValueError: `xs` and `v` have different length.

    """

    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = tf.gradients(ys, xs)

    assert len(grads) == length
    elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    return tf.gradients(elemwise_products, xs)

###########################################################################
def my_hessian_vector_product(f, variable_list, v):
    """
    modified from 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients_impl.py
    
    Multiply the Hessian of `ys` wrt `xs` by `v`.
    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.
    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.
    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.
    Args:
      f: A scalar value, or a tensor or list of tensors to be summed to
          yield a scalar.
      xs: A list of tensors that we should construct the Hessian over.
      v: A list of tensors, with the same shapes as xs, that we want to
          multiply by the Hessian.
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    Raises:
      ValueError: `xs` and `v` have different length.
    """
   
    #Chong edited
    assert np.prod(f.get_shape().as_list()) == 1, 'f should be a scalar variable, got %s'%f
    #Chong edited ends
    
    #ensure the length of the concatenated variable formed by the variables in the variable_list is the same as the gradient_vector
    cat_variable_length = 0
    for var in variable_list:
        cat_variable_length += var.get_shape().as_list()[0]
    assert len(v.get_shape().as_list()) == 1, 'v should be a vector, got %s'%v.get_shape().as_list()
    assert v.get_shape().as_list()[0] == cat_variable_length, 'length of v should match the length of concatenated vector'%v.get_shape().as_list()

    # First backprop
    #put grads and v in a list so they are enumerable in the zip in the next line
    grads = [myGradients(f, variable_list)]
    v = [v]
    
    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    
    # Second backprop
    return myGradients(elemwise_products, variable_list)

############################################################################
import time
class Timer(object):
    '''with Timer('name'):
            do something
    '''
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('%s took %.6f'%(self.name, time.time() - self.tstart))
        else:
            print('Elapsed %.6f'%(time.time() - self.tstart))


############################################################################
def get_shape(inputs):
    '''return the shape of a tensor or a numpy ndarray in numerical values'''

    if type(inputs) is np.ndarray:
        return inputs.shape
    else:
        try:
            return inputs.get_shape().as_list()
        except:
            raise ValueError('unknown input of type %s'%type(inputs))

############################################################################
#http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
#http://cs231n.github.io/convolutional-networks/
#from skimage.util import view_as_windows as viewW
def num_operation_conv2d(image_size, filter_size, strides, padding, rate=1, separable=False, num_outputs=None):
    '''
    calculate the number of operations of the convolution operation
    
    image_size: size of image (batch, x,y, channel)
    filter_size: size of filter kernel (filter_height, filter_width, input_channel,  output_channel)
    
    '''
    assert len(image_size) == 4,'inputs_size must be (batch, x,y, channel), got %s'%str(image_size)
    assert len(filter_size) == 4,'filter_size must be 4d (filter_height, filter_width, input_channel, output_channel), got %s'%str(filter_size)

    #do not consider batch size
    image_size = image_size[1:]

    num_input_channel = image_size[2]
    assert num_input_channel == filter_size[2], 'number of input channel does not match'

    num_output_channel = filter_size[3]

    if type(strides) is int:
        strides = (strides,strides)
    else:
        assert len(strides)==2
    if type(rate) is int:
        assert rate == 1, 'case for rate not 1 has not been checked %s'%str(rate)
        rate = (rate, rate)
    else:
        assert len(rate)==2
        assert all(r == 1 for r in rate)

    #compute the size of a column representation
    size_column = np.prod(filter_size[0:3])
  
    #compute number of columns, or the number of observations through sliding window on the image
    #according to nn.convolution document
    #If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    #If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).
    assert padding in ('VALID','SAME')
    if padding == "VALID":
        x_direction = math.ceil((image_size[0]-filter_size[0])/strides[0]) + 1
        y_direction = math.ceil((image_size[1]-filter_size[1])/strides[1]) + 1
    else:
        x_direction = math.ceil((image_size[0])/strides[0])
        y_direction = math.ceil((image_size[1])/strides[1])
    num_columns = x_direction*y_direction
    
    output_size = [1,x_direction,y_direction,num_output_channel]
    
    if separable:
        #each patch is just the size of [filter_height, filter_width]
        size_column = np.prod(filter_size[0:2])
        
        #filter_size is [kernel_h, kernel_w, num_input_channel, depth_multiplier]
        depth_multiplier = filter_size[3]
        assert depth_multiplier<=5, 'depth_multiplier so large? %d'%filter_size[3]
        depthwise_output_channel = num_input_channel*depth_multiplier
        depthwise_cost = depthwise_output_channel*size_column*num_columns
        
        #add a leading 1 to account for batch size
        output_size = list(output_size)
        
        if num_outputs is None:
            output_size[3] = num_input_channel*depth_multiplier
            pointwise_cost = 0
        else:
            #with pointwise convolution, matrix multiplication
            #at each point operation is
            #[1, depthwise_output_channel]*[depthwise_output_channel, num_outputs]
            pointwise_cost = (x_direction*y_direction)*depthwise_output_channel*num_outputs
            output_size[3] = num_outputs
            
        return depthwise_cost, pointwise_cost, output_size
        
    ##DEBUG
    #print('image_size: %s'%str(image_size))
    #print('filter_size: %s'%str(filter_size))
    #print('num_output_channel:%s'%num_output_channel)
    #print('size_column:%s'%size_column)
    #print('num_columns:%s'%num_columns)
    #total_operation = num_output_channel*size_column*num_columns
    #print('total_operation:%s'%total_operation)

    #return total number and the output image size
    return num_output_channel*size_column*num_columns, output_size

def num_memory_separable_conv2d(image_size, filter_size, num_outputs=None):
    '''compute the memory cost of a separable conv2d'''
    assert len(image_size) == 4,'inputs_size must be (batch, x,y, channel), got %s'%str(image_size)
    assert len(filter_size) == 4,'filter_size must be 4d (filter_height, filter_width, input_channel, output_channel), got %s'%str(filter_size)

    #do not consider batch size
    image_size = image_size[1:]

    num_input_channel = image_size[2]
    assert num_input_channel == filter_size[2], 'number of input channel does not match'

    #filter_size is [kernel_h, kernel_w, num_input_channel, depth_multiplier]
    depth_multiplier = filter_size[3]
    assert depth_multiplier<=5, 'depth_multiplier so large? %d'%filter_size[3]
    
    depthwise_output_channel = num_input_channel*depth_multiplier
    
    depthwise_cost = np.prod(filter_size)
    if num_outputs is None:
        pointwise_cost = 0
    else:
        #[depthwise_output_channel, num_outputs]
        pointwise_cost = depthwise_output_channel*num_outputs
        
    #TODO, in num_memory_per_K, divide factor is 1e7
    
    return depthwise_cost, pointwise_cost
    
    
############################################################################
def num_memory_per_K(filter_size, decomposition='tai'):
    '''Compute the memory requirement of a decomposed conv2d layer per singular component
    '''
    assert len(filter_size) ==4, 'filter_size must be 4d (filter_height, filter_width, input_channel,  output_channel), got %s'%filter_size
    assert filter_size[0] == filter_size[1], 'filter kernel should be square, got %s'%filter_size
 
    #make constraint smaller in range for numerical reasons
    divide_factor = 1e7

    original_cost = np.prod(filter_size)/divide_factor
    if decomposition== 'tai': 
        #[input_channel*width, height*output_channel]
        shape = [filter_size[2]*filter_size[1], filter_size[0]*filter_size[3]]
        per_K = sum(shape)
       
        #TODO, not sure if correct, decomposition may actually increase the memory cost
        return per_K/ divide_factor, original_cost
    elif decomposition == 'microsoft':
        #(k^2*c+d)*d'
        cost = filter_size[0]*filter_size[1]*filter_size[2] + filter_size[3] 
        return cost/divide_factor, original_cost
    else:
        raise ValueError('unknown decomposition %s'%decomposition)

############################################################################
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


from collections import OrderedDict
import os
import pickle
import warnings
from numbers import Number
class MaskingVariableManager(object, metaclass=Singleton):
    def __init__(self):
        #an ordered map from the name of the variable, to the index of that variable
        #in the long concatenated variable
        self.variable_to_index = OrderedDict()

        #mapping from the masking variable to the number of computations per singular component
        self.variable_to_num_computation = OrderedDict()
        self.variable_to_total_computation = OrderedDict()
        #mapping from the masking variable to the number of memory per singular component
        self.variable_to_num_memory = OrderedDict()
        self.variable_to_total_memory = OrderedDict()
        
        #class members to handle fine-tuning and heuristic
        #mapping from the masking variable to the numerical singular value
        self.variable_to_singular_values = OrderedDict()

        #mapping from the masking variable to the size of the flattened conv kernel (for Tai's paper only?)
        self.variable_to_flattened_shape = OrderedDict()

        #mapping from the name of the trainable variables to their initial values, used for fine-tuning
        self.variable_name_to_initial_value = OrderedDict()

        #for each variable,  [0...size] (following python index convention, the last index is one past the last element),
        #only consider a subset [reduced_start...reduced_end] (following python convention) to be free variable, [0...reduced_start] is fixed to 0 (always include)
        #[reduced_end...size] is fixed to 1 (always discard)
        #an ordered map from the name of the variable, to the index of the reduced variables in each variable
        self.variable_to_reduced_index = OrderedDict()

        #mapping from the masking variable to the reduced(free) variables
        self.variable_to_reduced_variable = OrderedDict()

        #a pointer to keep track of which segment of the sqp solution should be used to reconstruct the W_sqp, used in my_slim_layer.py
        self.add_and_svd_index = 0
        #a dict mapping from the name of the layer (sc.name) to the add_and_svd_K used in that layer
        self.add_and_svd_K_dict = OrderedDict()

        #a dict mapping from the masking variable to its corresponding DecompositionScheme object
        self.variable_to_decomposition_scheme = OrderedDict()

        #DEBUG
        #this is temporary, record the computation cost and the memory cost of separable conv2d layer
        self.variable_to_sep_info = OrderedDict()
        self.sep_comp_divide_factor = 1e10
        self.sep_mem_divide_factor = 1e7


    ##################################
    @property
    def num_masking_variables(self):
        #total number of binary masking varibales
        return self.get_num_mask_variables()

    ##################################
    def add_variable(self, tf_variable, num_computation=None, num_memory=None, singular_values=None, flattened_shape=None, 
            initial_value=None, reduced_index=None, reduced_variable=None, decomposition_scheme=None):
        #register a new tensorflow variable with the manager

        #get the shape of the variable
        variable_shape = tf_variable.get_shape().as_list()
        assert len(variable_shape) == 1, 'expecting that the variables is a vector of binary masking variables, input is %s '%(variable_shape)

        #the range of the binary masks in the long concatenated variable
        index_range = (self.num_masking_variables, self.num_masking_variables + variable_shape[0])
        #self.variable_to_index[tf_variable.op.name] = index_range
        self.variable_to_index[tf_variable] = index_range

        if num_computation is not None:
            self.add_num_computation(tf_variable, num_computation)
        if num_memory is not None:
            self.add_num_memory(tf_variable, num_memory)
        if singular_values is not None:
            self.add_singular_values(tf_variable, singular_values)
        if flattened_shape is not None:
            self.add_flattened_shape(tf_variable, flattened_shape)
        if initial_value is not None:
            self.add_initial_value(tf_variable, initial_value)
        if reduced_index is not None:
            self.set_reduced_index(tf_variable, reduced_index)
        else:
            #initialize the reduced index of the variable to be the entireity, all binary mask variables are free
            self.variable_to_reduced_index[tf_variable] = [0, variable_shape[0]]
        if decomposition_scheme is not None:
            self.variable_to_decomposition_scheme[tf_variable]=decomposition_scheme

        assert reduced_variable is not None
        self.variable_to_reduced_variable[tf_variable] = reduced_variable


    def get_variables(self, include_fixed_variables=False):
        '''return a ordered list of the tensorflow variables registered with this mvm.
           include_fixed_variables is false: return the reduced_variables
           include_fixed_variables is true: return the full variables, including the top and bottom portion that are fixed to 0 and 1 by the reduced_index
        
        '''
        if include_fixed_variables:
            return list(self.variable_to_index.keys())
        else:
            return list(self.variable_to_reduced_variable.values())


    def get_num_mask_variables(self):
        '''return the total number of binary mask variables'''

        total_num_mask_variables = 0

        for var in self.variable_to_index.keys():
            assert var.get_shape().ndims == 1, 'variable should be vector, got %s'%var.get_shape().as_list()
            total_num_mask_variables += var.get_shape().as_list()[0]

        return total_num_mask_variables

    def get_num_reduced_mask_variables(self):
        '''return the total number of reduced(free) binary mask variables'''
        total_num_reduced_mask_variables = 0

        for var in self.variable_to_index.keys():
            assert var.get_shape().ndims == 1, 'variable should be vector, got %s'%var.get_shape().as_list()
            total_num_reduced_mask_variables += (self.variable_to_reduced_index[var][1] -  self.variable_to_reduced_index[var][0])

        return total_num_reduced_mask_variables

    def set_reduced_index(self, var, index):
        '''set self.variable_to_reduced_index, the index of the free binary mask variables'''

        assert var in self.variable_to_index, 'variable must have already registered with mvm, got %s'%var.name
        assert index[1] >= index[0], 'index must be a range %s'%str(index)
        assert index[0] >= 0, 'index must be a range %s'%str(index)
        assert index[1] <= var.get_shape().as_list()[0], 'index must be a range %s'%str(index)

        self.variable_to_reduced_index[var] = index

    def get_mask_variable_value_using_heuristic(self, heuristic=None, computation_max=None, memory_max=None, 
                    monotonic=False, timelimit=None):
        '''
            return a long concatenated value of mask variable, based on heuristic that only considers the singular values
            of each layer, but not the correlation between layers

            when this function is called, the net should have been fully constructed and all the mask variables and their singular values
            are registered with the MaskVariableManager
        '''
       
        self.check_constraint_feasiability_against_reduced_index(computation_max, memory_max)

        #heuristic 2 call a external function
        if heuristic == 2:
            #Option 2: microsoft cvpr paper equation 13 and 14
            #assert not FLAGS.enable_reduction, 'reduction is unsupported for this heuristic, due to 0 free variables for some layers'
            return self.call_cvx_sumlog_heuristics(computation_max=computation_max, memory_max=memory_max, timelimit=timelimit)

        if heuristic == 3:
            return self.greedy_product_sum_heuristic(computation_max=computation_max, memory_max=memory_max)

        if heuristic in (0,1):
            return self.get_mask_variable_value_truncating_singular_value(heuristic, computation_max, memory_max)

        if heuristic in (5,6):
            assert FLAGS.enable_reduction is False
            #initialize a masking variable values for the long concatenated variable
            mask_variable_values_all = np.array([], dtype=np.float32)

            assert len(self.variable_to_singular_values) > 0, 'variable_to_singular_values unregistered'
            assert len(self.variable_to_singular_values) == len(self.variable_to_flattened_shape), 'singular value and flattened shape should both be registered'

            eng = matlab.engine.start_matlab()
            #add path
            eng.addpath(r'/home/chongli/research/sparse',nargout=0)

            for var, singular_values in self.variable_to_singular_values.items():
                S = singular_values

                #initial solution for this layer
                layer_solution = np.ones_like(S)
                #kk is the index of truncating the singular value [0:kk] (actually element 0 to kk-1) are kept
                if heuristic == 5:
                    #Option 3: use the optimal hard threshold value is 4/sqrt(3) (stanford paper)
                    flattened_shape = self.variable_to_flattened_shape[var]
                    m = flattened_shape[0]
                    n = flattened_shape[1]

                    beta = float(m)/float(n)

                    #find the hard threshold
                    threshold = (0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43)*statistics.median(S)

                    for kk in range(0, len(S)):
                        if S[kk] < threshold:
                            break
                elif heuristic == 6:
                    ds = self.variable_to_decomposition_scheme[var]
                    weight_2d, _ = ds.get_weight_2d()

                    weight_2d = weight_2d.tolist()
                    weight_2d = matlab.double(weight_2d)

                    kk = eng.VBMF_wrapper(weight_2d)
                else:
                    raise ValueError()

                layer_solution[0:kk+1] = 0
                mask_variable_values_all = np.hstack([mask_variable_values_all, layer_solution])
            eng.quit()

            solution = mask_variable_values_all
            pickle_path = '/tmp/solution.pickle'
            with open(pickle_path, 'wb') as f:  
                pickle.dump(solution, f, protocol=-1)

            #save to mat format
            mat_path = pickle_path.replace('.pickle','.mat')
            scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

            return mask_variable_values_all

        if 20<=heuristic<=40:
            return self.get_mask_variable_close_original(heuristic)

        #return an solution with all mask variables enabled
        if heuristic == 10:
            solution = np.zeros(self.get_num_mask_variables(), np.float32)

            pickle_path = '/tmp/solution.pickle'
            with open(pickle_path, 'wb') as f:  
                pickle.dump(solution, f, protocol=-1)

            #save to mat format
            mat_path = pickle_path.replace('.pickle','.mat')
            scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

            return solution
        #return an solution with all mask variables disabled, using heuristic 10 and 11 together will set reduce_variables to all variables
        if heuristic == 11:
            solution = np.ones(self.get_num_mask_variables(), np.float32)

            pickle_path = '/tmp/solution.pickle'
            with open(pickle_path, 'wb') as f:  
                pickle.dump(solution, f, protocol=-1)

            #save to mat format
            mat_path = pickle_path.replace('.pickle','.mat')
            scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

            return solution

        raise ValueError('Unknown heuristic %d'%heuristic)

    def get_mask_variable_close_original(self, heuristic):
        '''for each layer, if all the singular values are used in the 2 cascading decomposed layers, it is possible that 
        the computation/memory cost is higher than the original.

        return a masking variable solution, such that for each layer, the computation/memory cost is (100 + (heuristic-20))%
        of the original cost
        '''

        assert 20<=heuristic<=40

        target = (100+heuristic-20)/100

        #initialize a masking variable values for the long concatenated variable
        mask_variable_values_all = np.array([], dtype=np.float32)

        for var, original_computation in self.variable_to_total_computation.items():
            index = self.variable_to_index[var]

            kk = math.floor(original_computation*target/self.variable_to_num_computation[var])
            
            mask_variable_value = np.ones([index[1]-index[0]], dtype=np.float32)
            mask_variable_value[0:kk] = np.float32(0.0)

            #hstack to form the long concatenated variable
            mask_variable_values_all = np.hstack([mask_variable_values_all, mask_variable_value])

        #verify the shape of the returned vector
        assert len(mask_variable_values_all.shape) == 1
        assert mask_variable_values_all.shape[0] == self.num_masking_variables, 'shape of returned vector %s does not match registered mask variables %d'%(mask_variable_values_all.shape, self.num_masking_variables)

        #save to pickle
        solution = mask_variable_values_all
        pickle_path = '/tmp/solution.pickle'
        with open(pickle_path, 'wb') as f:  
            pickle.dump(solution, f, protocol=-1)

        #save to mat format
        mat_path = pickle_path.replace('.pickle','.mat')
        scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

        return solution

    def get_mask_variable_value_truncating_singular_value(self, heuristic=None, computation_max=None, memory_max=None):
        '''heuristic 0: keep the top x percent of the squared sum of singular values
           heuristic 1: keep the top x percent of the sum of the singular values'''

        #################################
        def _compute_cost_using_percentage(heuristic=None, percentage=None, computation_max=None, memory_max=None):
            '''given a percentage, return the computation cost and a solution of masking variables'''

            assert 0 < percentage <= 1, 'invalid percentage %s'%percentage
            assert 0 < computation_max <= 1, 'invalid computation_max %s'%computation_max
            
            #initialize a masking variable values for the long concatenated variable
            mask_variable_values_all = np.array([], dtype=np.float32)

            assert len(self.variable_to_singular_values) > 0, 'variable_to_singular_values unregistered'
            assert len(self.variable_to_singular_values) == len(self.variable_to_flattened_shape), 'singular value and flattened shape should both be registered'

            for var, singular_values in self.variable_to_singular_values.items():
                S = singular_values
                if heuristic == 0:
                    #Option 0: percentage% of the total squared eigenvalue
                    kk = 1
                    threshold = percentage*sum(S**2)
                    while kk < len(S):
                        if sum(S[:kk]**2 ) >= threshold:
                            break
                        kk += 1
                elif heuristic == 1:
                    #Option 1: percentage% of the total singular value
                    kk = 1
                    threshold = percentage*sum(S)
                    while kk < len(S):
                        if sum(S[:kk] ) >= threshold:
                            break
                        kk += 1
                #kk is the index of truncating the singular value [0:kk] (actually element 0 to kk-1) are kept
                #elif heuristic == 2:
                #    #Option 2: keep the top percentage of the singular values
                #    kk = int(len(S)*percentage)
                #elif heuristic == 3:
                #    #Option 3: use the optimal hard threshold value is 4/sqrt(3) (stanford paper)
                #    flattened_shape = self.variable_to_flattened_shape[var]
                #    m = flattened_shape[0]
                #    n = flattened_shape[1]

                #    #TODO swap m,n?
                #    beta = float(m)/float(n)

                #    #find the hard threshold
                #    import statistics
                #    threshold = (0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43)*statistics.median(S)

                #    for kk in range(0, len(S)):
                #        if S[kk] < threshold:
                #            break
                else:
                    raise ValueError('Unknown heuristic %d' % heuristic)

                #the value of the mask variables for this variable alone
                mask_variable_value = np.ones([len(singular_values)], dtype=np.float32)
                mask_variable_value[0:kk+1] = np.float32(0.0)

                #self.variable_to_value_dict[var] = mask_variable_value

                #hstack to form the long concatenated variable
                mask_variable_values_all = np.hstack([mask_variable_values_all, mask_variable_value])

            #self.calculate_percentage_computation_memory_cost(mask_variable_values_all, 'Heuristics %d'%heuristic)
            
            #verify the shape of the returned vector
            assert len(mask_variable_values_all.shape) == 1
            assert mask_variable_values_all.shape[0] == self.num_masking_variables, 'shape of returned vector %s does not match registered mask variables %d'%(mask_variable_values_all.shape, self.num_masking_variables)

            total_computation_cost = self.get_total_computation_cost()
            used_computation_percentage = fast_dot(1-mask_variable_values_all, self.get_computation_cost_coefficient()) / total_computation_cost

            return (used_computation_percentage, mask_variable_values_all)
        #################################

        upper = 0.999
        lower= 0.001
    
        while upper > lower and upper - lower > 0.0001:
            mid = (upper+lower)/2
            used_computation_cost, solution = _compute_cost_using_percentage(heuristic,mid,computation_max) 

            if used_computation_cost > computation_max:
                upper = mid
            elif used_computation_cost < computation_max:
                lower= mid
            else:
                break

        assert abs(used_computation_cost-computation_max)<0.02, 'used computation percentage: %s, computation_max: %s'%(used_computation_cost, computation_max)

        pickle_path = '/tmp/solution.pickle'
        with open(pickle_path, 'wb') as f:  
            pickle.dump(solution, f, protocol=-1)

        #save to mat format
        mat_path = pickle_path.replace('.pickle','.mat')
        scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

        return solution
            
    def greedy_product_sum_heuristic(self, computation_max=None, memory_max=None, pickle_path = '/tmp/solution.pickle'):
        '''implements the greedy heuristic for equation 13 and 14 in microsoft paper'''

        if computation_max is None:
            computation_max = 1
        if memory_max is None:
            memory_max = 1

        total_computation_cost = self.get_total_computation_cost()
        total_memory_cost = self.get_total_memory_cost()

        comp_coeff = np.array(self.get_computation_cost_coefficient())
        mem_coeff = np.array(self.get_memory_cost_coefficient())

        constrained_computation_cost = total_computation_cost*computation_max
        constrained_memory_cost = total_memory_cost*memory_max

        #the initial solution, all singular values are enabled
        solution = np.full(self.num_masking_variables, 1, dtype=np.int64)

        #get all the singular values
        all_singular_values = np.concatenate(list(self.variable_to_singular_values.values()) )

        #TODO manipulate the singular values here
        all_singular_values = [v**2 for v in all_singular_values]
        all_singular_values = np.array(all_singular_values)
        assert all_singular_values.shape[0] == solution.shape[0]

        #compute the Dot product between the Solution and the Singular values
        dss = np.multiply(solution, all_singular_values)
        assert len(dss) == len(solution)

        #compute the original obj
        obj = np.longdouble(0.0)
        for var, index in self.variable_to_index.items():
            obj += np.log(np.sum(dss[index[0]:index[1]]), dtype=np.longdouble)

        #initialize computation cost and memory cost
        computation_cost = fast_dot(solution, comp_coeff) 
        memory_cost = fast_dot(solution, mem_coeff)

        #initialize two list, with the range of the reduced index
        layer_index_upper = list()
        layer_index_lower = list()
        for var in self.variable_to_singular_values.keys():
            var_index = self.variable_to_index[var]
            layer_index_upper.append(var_index[0])
            #-1 because the lower index is one past the item as in python indexing convention
            layer_index_lower.append(var_index[1]-1)

        #initialize the array containing the change in objective if the smallest singular value of a layer is removed
        delta_obj = np.full(len(layer_index_lower), np.float32(None))

        while computation_cost > constrained_computation_cost or memory_cost > constrained_memory_cost:
            #for the smallest singular value in each layer, compute the change in objective
            for layer_idx, bin_idx in enumerate(layer_index_lower):
                #if all but one singular value are discarded, do not consider this layer
                if bin_idx == layer_index_upper[layer_idx]:
                    delta_obj[layer_idx] = np.float32('inf')
                    continue

                #discard this singular value in the dot product between solution and singular values by setting it to 0
                dss_element = dss[bin_idx]
                dss[bin_idx] = 0

                #compute the change in objective
                new_obj = np.longdouble(0.0)
                for var, index in self.variable_to_index.items():
                    new_obj += np.log(np.sum(dss[index[0]:index[1]]), dtype=np.longdouble)

                delta_obj[layer_idx] = obj - new_obj
                assert delta_obj[layer_idx] >= 0, 'delta_obj[layer_idx]: %f, layer_idx: %d, bin_idx: %d'%(delta_obj[layer_idx], layer_idx, bin_idx)
                assert np.isfinite(delta_obj[layer_idx]) 

                #reset the dss, after the change in objective is saved in delta_obj
                dss[bin_idx] = dss_element
            
            assert not np.isnan(np.sum(delta_obj))
            #measure is delta_obj/delta_cost
            #the greedy heuristic cannot handle multiple constraints, only considering computation cost here TODO
            #assert len(delta_obj) == len(comp_coeff)
            #measure = np.divide(delta_obj, comp_coeff)
            #assert len(measure) == len(delta_obj)

            #discard_idx = np.argmin(measure)
            #TODO if use the meaure, then the code breaks down, problem is how the delta cost is calcuated
            discard_layer_idx = np.argmin(delta_obj)
            discard_idx = layer_index_lower[discard_layer_idx]
            #minus 1 because if all variables are discarded, then lower pointer is one past upper pointer
            assert discard_idx > layer_index_upper[discard_layer_idx], 'upper: %s, lower: %s, discard_layer_idx: %d'%(str(layer_index_upper), 
                    str(layer_index_lower), discard_layer_idx)

            ##if there are multiple indices that matches the min value 
            #min_idxs = np.where(delta_obj == delta_obj[discard_idx])[0]
            ##among the indices of the mins, find one that has largest computation cost
            #discard_layer_idx = min_idxs[np.argmax(comp_coeff[min_idxs]) ]

            #discard the the singular value in solution and dss
            solution[discard_idx] = 0
            dss[discard_idx] = 0

            #in next iteration, should check the next larger singular value in the layer that a singular value was just dropped
            layer_index_lower[discard_layer_idx] -= 1

            #update the computation and memory cost
            computation_cost -= comp_coeff[discard_idx]
            memory_cost -= mem_coeff[discard_idx]

        #recompute optimal obj value
        dss = np.multiply(solution, all_singular_values)
        obj = np.longdouble(0.0)
        for var, index in self.variable_to_index.items():
            #solution in this layer
            sl = solution[index[0]:index[1]]
            obj += np.log(np.sum(dss[index[0]:index[1]]), dtype=np.longdouble)
            #the solution for each layer should be descending in this heuristic
            assert all(sl[i] >= sl[i+1] for i in range(len(sl)-1)), 'in layer %s (index: %s), a singular value should not be discarded before all the values smaller than it are discarded %s'%(var.op.name, str(index), str(sl))

        print('----RESULT:greedy heuristic objective value: %.6E'%obj)
        #in other parts of the code, 0 means keeping the singular value and 1 means discarding it, so need to invert 0 and 1
        solution = 1 - solution 
        solution = np.float32(solution)

        #DEBUG
        #self.print_solution(solution, check_monotonic=True, print_value=False)

        #reduce the solution
        reduced_solution = self.reduce_mask_variables_np(solution, exact_size=True)

        with open(pickle_path, 'wb') as f:  
            pickle.dump(reduced_solution, f, protocol=-1)

        #save to mat format
        mat_path = pickle_path.replace('.pickle','.mat')
        scipy.io.savemat(mat_path, {'solution': np.array(reduced_solution, dtype=np.float64)}, do_compression=True)

        print('---greedy solution saved in %s at %s'%(pickle_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))
        return solution
    
    def expand_reduced_mask_variables_np(self, reduced_mask_variables_np, exact_size=False):
        '''given a numpy vector containing the value of the reduced mask variable values that are concatenated, return a np vector of the length
        num_mask_variables with the leading part fill by 0 and trailing part filled by 1, for each binary mask  variable'''

        reduced_mask_variables_np = np.array(reduced_mask_variables_np)
        assert reduced_mask_variables_np.ndim == 1,'input should be 1D vector, got %s'%str(reduced_mask_variables_np.shape)
        assert not np.isnan(np.sum(reduced_mask_variables_np)), 'Cannot have any NaN in the reduced_masking_variables_np'

        #find the total number of reduced variables
        num_reduced_mask_variables = self.get_num_reduced_mask_variables()
        num_mask_variables = self.get_num_mask_variables()
        assert reduced_mask_variables_np.shape[0] == num_mask_variables or reduced_mask_variables_np.shape[0] == num_reduced_mask_variables, 'reduced_mask_variables_np shape: %s, num_mask_variables: %d, num_reduced_mask_variables: %d'%(reduced_mask_variables_np.shape, num_mask_variables, num_reduced_mask_variables)

        #if the size of variable for reduced and full solution vector are the same, then just return
        if num_reduced_mask_variables == num_mask_variables:
            return reduced_mask_variables_np

        #if the input reduced_mask_variables_np is actually the np for the entireity of the mask variables
        #just return without expanding
        if reduced_mask_variables_np.shape[0] == num_mask_variables:
            if exact_size:
                raise ValueError('input vector is not the reduced binary variables')
            return reduced_mask_variables_np

        assert reduced_mask_variables_np.shape[0] == self.get_num_reduced_mask_variables(), 'input size should match the number of reduced_mask_variables, got %s'%str(reduced_mask_variables_np)

        #initialize the expanded_vector with nan
        expanded_vector = np.full([self.num_masking_variables], np.float32(None))

        reduced_np_start = 0
        for var,reduced_index in self.variable_to_reduced_index.items():
            index = self.variable_to_index[var]

            #start and end index of the reduced variables in the expanded vector
            reduced_start = reduced_index[0] + index[0]
            reduced_end = reduced_index[1] + index[0]

            #fill the top portion to 0, meaning to include the singular value
            expanded_vector[index[0]: reduced_start] = np.float32(0.0)
            #copy the values in the reduced_mask_variables_np
            expanded_vector[reduced_start: reduced_end] = reduced_mask_variables_np[reduced_np_start: (reduced_np_start + reduced_index[1] - reduced_index[0])]
            #fill the bottom portion to 1, meaning to discard the singular value
            expanded_vector[reduced_end: index[1]] = np.float32(1.0)

            reduced_np_start += (reduced_index[1] - reduced_index[0])

        assert not np.isnan(np.sum(expanded_vector)), 'Cannot have any NaN in the expanded_vector'
        expanded_vector = np.float32(expanded_vector)
        return expanded_vector

    def reduce_mask_variables_np(self, mask_variables_np, exact_size=False):
        '''given a numpy vector containing the value of the full mask variable values, return a concatenated np vector that only contains
        the reduced variable
        '''

        mask_variables_np = np.array(mask_variables_np)
        assert mask_variables_np.ndim == 1,'input should be 1D vector, got %s'%str(mask_variables_np.shape)
        assert not np.isnan(np.sum(mask_variables_np)), 'Cannot have any NaN in the reduced_masking_variables_np'

        #find the total number of reduced variables
        num_reduced_mask_variables = self.get_num_reduced_mask_variables()
        num_mask_variables = self.get_num_mask_variables()
        assert mask_variables_np.shape[0] == num_mask_variables or mask_variables_np.shape[0] == num_reduced_mask_variables, 'mask_variables_np: %s, num_mask_variables: %s, num_reduced_mask_variables:%s '%(mask_variables_np.shape, num_mask_variables, num_reduced_mask_variables)

        #if the size of variable for reduced and full solution vector are the same, then just return
        if num_reduced_mask_variables == num_mask_variables:
            return mask_variables_np

        #if the input np vector is for the reduced variables already, just return
        if mask_variables_np.shape[0] == num_reduced_mask_variables and mask_variables_np.shape[0] != num_mask_variables:
            if exact_size:
                raise ValueError('input vector is not the entireity of the binary mask variables')
            return mask_variables_np

        assert mask_variables_np.shape[0] == self.num_masking_variables, 'input vector is not of correct length, expected %d, got %d'%(self.get_num_mask_variables(), mask_variables_np.shape[0])

        reduced = list()
        for var,reduced_index in self.variable_to_reduced_index.items():
            index = self.variable_to_index[var]

            var_np_full = mask_variables_np[index[0]: index[1]]
            assert np.array(var_np_full).shape[0] >= reduced_index[1]
            reduced.extend(var_np_full[reduced_index[0]: reduced_index[1]] )

        reduced = np.array(reduced, dtype=np.float32) 
        assert not np.isnan(np.sum(reduced)), 'Cannot have any NaN in the reduced vector'
        assert reduced.shape[0] == num_reduced_mask_variables, 'reduced np shape not right'
        return reduced

    def compute_reduced_index(self, S=None, op_name=None):
        '''compute the reduced_index, the range of the free variables, given the singular values
        consider loading from previous iteration

        input: the singular values of the flattened weight. How the weight is flattened depends on decomposition
        return: reduced_index, the range of the reduced(free) binary variables
        '''

        if FLAGS.enable_reduction is False:
            #do not do any reduction
            assert S is not None
            reduced_index=[0,len(S)]
        elif FLAGS.load_reduced_index:
            assert os.path.isfile('/tmp/reduced_index.pickle'), '/tmp/reduced_index.pickle does not exist'
            #has previously ran mask_compute_reduced_index script
            with open('/tmp/reduced_index.pickle','rb') as f:
                var_name_reduced_index = pickle.load(f)

            assert op_name is not None, 'must provide variable name if want to load reduced_index'
            assert op_name in var_name_reduced_index, '%s is not in %s'%(op_name, str(list(var_name_reduced_index.keys())))

            reduced_index = var_name_reduced_index[op_name]

            #DEBUG
            #print('in compute_reduced_index, var: %s, reduced_index: %s, out of [0,%d]'%(op_name, str(reduced_index), len(S)))
        else:
            raise NotImplementedError('using cumulative eigen energy is not good')

        assert reduced_index[0] <= reduced_index[1], 'has to be a valid range %s'%str(reduced_index)
        assert reduced_index[0] >= 0
        assert reduced_index[1] <= len(S)

        return reduced_index
        
    def get_variable_to_value_dict(self, mask_variables_np=None):
        '''given a numpy vector containing the value of each mask variables, return a map, from the variable, to its value'''
        
        #assert self.num_masking_variables > 0, 'cannot use on a uninitialized object'
        if self.num_masking_variables == 0:
            warnings.warn('~~~~~~~~~~~~~~~~~~~~~~~~No masking variable is registered with the manager~~~~~~~~~')
            return {}

        #all mask enabled if no value is provided
        if mask_variables_np is None:
            mask_variables_np = np.zeros(self.num_masking_variables, np.float32)

        #only consider the reduced(free) variables
        mask_variables_np = self.reduce_mask_variables_np(mask_variables_np)

        assert len(mask_variables_np.shape) == 1, 'expecting input is a vector, but %s' % mask_variables_np
        assert mask_variables_np.shape[0] == self.get_num_reduced_mask_variables(), 'Length of np value vector %d must match the reduced masking variables %d'%(mask_variables_np.shape[0], self.get_num_reduced_mask_variables())
        assert not np.isnan(np.sum(mask_variables_np)), 'Cannot have any NaN in the masking_variables_np'

        #initialize the dict
        variable_to_value_dict = OrderedDict()
       
        start = 0
        for var, index_range in self.variable_to_index.items():
            #the placeholders are the reduced variables
            assert var in self.variable_to_reduced_variable
            reduced_var = self.variable_to_reduced_variable[var]

            reduced_index = self.variable_to_reduced_index[var]
            #number of binary reduced variable for this var(layer)
            num_var = reduced_index[1] - reduced_index[0]

            end = start + num_var
            variable_to_value_dict[reduced_var]=mask_variables_np[start:end]
            start += num_var

        #self.calculate_percentage_computation_memory_cost(mask_variables_np)
            
        return variable_to_value_dict

    def calculate_percentage_computation_memory_cost(self, mask_variables_np, print_prefix = ''):

        #expand the input np if necessary
        mask_variables_np = self.expand_reduced_mask_variables_np(mask_variables_np)
        assert np.array(mask_variables_np).ndim == 1
        assert np.array(mask_variables_np).shape[0] == self.num_masking_variables

        #invert the mask_variables_np, after inverting 1 means a singular value is included, 0 means discarded
        mask_variables_np = 1- np.array(mask_variables_np)

        used_computation_cost = 0.0
        used_memory_cost = 0.0

        for var, index_range in self.variable_to_index.items():
            #calculate the percentage of computation and memory cost with the given masking_variable_np
            layer_coeff = mask_variables_np[index_range[0]:index_range[1]]
            layer_computation_cost = np.sum(layer_coeff)*self.variable_to_num_computation[var]
            layer_memory_cost = np.sum(layer_coeff)*self.variable_to_num_memory[var]

            if FLAGS.cost_saturation: 
                #use the cost of un-decomposed layer(orignal cost)
                if layer_computation_cost >= self.variable_to_total_computation[var] or layer_memory_cost >= self.variable_to_total_memory[var]:
                    layer_computation_cost = self.variable_to_total_computation[var]
                    layer_memory_cost = self.variable_to_total_memory[var]

            used_computation_cost += layer_computation_cost
            used_memory_cost += layer_memory_cost

        used_computation_percentage = used_computation_cost / self.get_total_computation_cost()
        used_memory_percentage = used_memory_cost/ self.get_total_memory_cost()

        print('---%s masking_variable_np used %.3f of computation cost and %.3f of memory cost'%(print_prefix, used_computation_percentage, used_memory_percentage))

        return used_computation_percentage, used_memory_percentage
        
    def print_variable_index(self):
        '''print the index range of the variables in the long masking variable'''
        #assert self.num_masking_variables > 0, 'cannot use on a uninitialized object'
        if self.num_masking_variables == 0:
            warnings.warn('~~~~~~~~~~~~~~~~~~~~~~~~Cannot print variable index. No masking variable is registered with the manager~~~~~~~~~')
            return

        print('----Index of variables in the long concated masking variable vector,in print_variable_index()')
        for var, index_range in self.variable_to_index.items():
            print('---%s: %s, reduce index: %s'%(var.op.name, self.variable_to_index[var], self.variable_to_reduced_index[var]))

    def is_empty(self):
        '''return if any varibles has been registered with the manager'''
        assert self.num_masking_variables >= 0
        return (self.num_masking_variables == 0)

    ##################################
    def add_num_computation(self, variable, num_computation):
        '''register the amount of computation for each singular component corresponding to the masking variable'''

        assert isinstance(num_computation, Number)
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        self.variable_to_num_computation[variable]=num_computation

    def add_num_memory(self, variable, num_memory):
        '''register the amount of memory for each singular component corresponding to the masking variable'''
        
        assert isinstance(num_memory, Number)
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        self.variable_to_num_memory[variable]=num_memory

    def add_total_computation(self, variable, total_computation):
        '''register the total amount of computation for the layer corresponding to the masking variable'''

        assert isinstance(total_computation, Number)
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        self.variable_to_total_computation[variable]=total_computation

    def add_total_memory(self, variable, total_memory):
        '''register the total amount of memory for the layer corresponding to the masking variable'''
        
        assert isinstance(total_memory, Number)
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        self.variable_to_total_memory[variable]=total_memory

    def add_singular_values(self, variable, singular_values):
        '''register the numerical singular values that corresponds to the masking variables'''

        assert singular_values.ndim == 1,'singular value should be a vector, got %s'%singular_values.shape
        assert singular_values.shape[0] == variable.shape[0], 'shape of singular_values vector does not match the corresponding masking variable'
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        assert all(singular_values[i] >= singular_values[i+1] for i in range(len(singular_values)-1)) or np.all(np.isnan(singular_values)),  'S not in descending order %s'%str(singular_values)
        self.variable_to_singular_values[variable] = singular_values

    def add_flattened_shape(self, variable, flattened_shape):
        '''register the numerical singular values that corresponds to the masking variables'''

        assert len(flattened_shape) == 2,'flattened_shape shape error, got %s'%flattened_shape.shape
        assert variable in self.variable_to_index, 'variable %s is not registered'%variable
        self.variable_to_flattened_shape[variable] = flattened_shape

    def add_initial_value(self, variable_name, initial_value):
        '''add a variable name and its initial value to the self.variable_name_to_initial_value'''

        assert variable_name not in self.variable_name_to_initial_value, 'adding %s to variable_name_to_initial_value again'%variable_name
        assert type(initial_value) == np.ndarray, 'initial_value should be a np.array object, got %s'%type(initial_value)

        self.variable_name_to_initial_value[variable_name] = initial_value

    ##################################
    def get_variable_name_to_initial_value_dict(self):
        '''return the dict that maps the variable name to initial values for fine-tuning'''
        return self.variable_name_to_initial_value

    def get_total_computation_cost(self):
        '''report the total computation cost with no approximation'''

        total_computation_cost = 0.0
        for var in self.variable_to_num_computation.keys():
            assert self.variable_to_total_computation[var] > 0
            total_computation_cost += self.variable_to_total_computation[var]

        return total_computation_cost

    def get_total_memory_cost(self):
        '''report the total memory cost with no approximation'''

        total_memory_cost = 0.0
        for var in self.variable_to_num_memory.keys():
            assert self.variable_to_total_memory[var] > 0
            total_memory_cost += self.variable_to_total_memory[var]

        return total_memory_cost

    def get_computation_cost_coefficient(self):
        '''return a list whose length is the total number of masking variables. Each element is per singular value computation cost'''

        coefficient=[]
        for var, index_range in self.variable_to_num_computation.items():
            #number of masking variables of var
            var_index = self.variable_to_index[var]
            var_num_masking_variable = var_index[1] - var_index[0]
            coefficient.extend([self.variable_to_num_computation[var]]*var_num_masking_variable)

        return coefficient

    def get_memory_cost_coefficient(self):
        '''return a list whose length is the total number of masking variables. Each element is per singular value memory cost'''

        coefficient=[]
        for var, index_range in self.variable_to_num_memory.items():
            #number of masking variables of var
            var_index = self.variable_to_index[var]
            var_num_masking_variable = var_index[1] - var_index[0]
            coefficient.extend([self.variable_to_num_memory[var]]*var_num_masking_variable)

        return coefficient

    def compute_unaccounted_cost(self, cost_type, portion, as_list=False):
        '''compute the computation/memory cost of the singular values that are fixed to 0 (include) if portion is top, and those are fixed to 1 (discard) is portion is bottom,
        these variables are not reduced(free) binary variables'''

        assert cost_type in ['computation', 'memory'], 'unknown cost type %s'%cost_type
        assert portion in ['top','bottom', 'both']

        cost = 0.0
        cost_list = []
        for var, var_index in self.variable_to_index.items():
            if cost_type == 'computation':
                unit_cost = self.variable_to_num_computation[var]
            elif cost_type == 'memory':
                unit_cost = self.variable_to_num_memory[var]
            else:
                raise ValueError

            reduced_index = self.variable_to_reduced_index[var]

            if portion == 'top':
                cost += (reduced_index[0]*unit_cost)
                cost_list.append(reduced_index[0]*unit_cost)
            elif portion == 'bottom':
                #find the number of binary variables of the un-reduced variable
                var_length = var_index[1] - var_index[0]
                cost += (var_length - reduced_index[1])*unit_cost
                cost_list.append((var_length - reduced_index[1])*unit_cost) 
            elif portion == 'both':
                #unaccounted cost of both included leading singular values and discarded values
                var_length = var_index[1] - var_index[0]
                cost += (reduced_index[0]*unit_cost) + (var_length - reduced_index[1])*unit_cost
                cost_list.append((reduced_index[0]*unit_cost) + (var_length - reduced_index[1])*unit_cost)
            else:
                raise ValueError

        if as_list:
            return cost, cost_list
        else:
            return cost

    def check_constraint_feasiability_against_reduced_index(self, computation_max=None, memory_max=None, as_error = True):
        '''check if it is possible to satisfy the requested computation/memory max given the reduced_index'''

        computation_lower = self.compute_unaccounted_cost('computation','top') / self.get_total_computation_cost()
        assert computation_lower >=0
        computation_upper = (sum(self.get_computation_cost_coefficient()) - self.compute_unaccounted_cost('computation','bottom')) / self.get_total_computation_cost()
        memory_lower = self.compute_unaccounted_cost('memory','top') / self.get_total_memory_cost()
        memory_upper = (sum(self.get_memory_cost_coefficient())- self.compute_unaccounted_cost('memory','bottom')) / self.get_total_memory_cost()

        print('the range of computation/memory cost is [%.3f, %.3f]/[%.3f, %.3f]'%(computation_lower, computation_upper, memory_lower, memory_upper))
         
        if computation_max is not None:
            #assert 0 < computation_max < 1, 'invalid computation_max %s'%computation_max
            if computation_lower > computation_max:
                warnings.warn('~~~Impossible to satisfy computation_max %.3f, the cost of singular values included due to reduced_cost is %.3f '%(computation_max, computation_lower))
                if as_error:
                    raise ValueError
            if computation_upper < computation_max:
                warnings.warn('~~~Trivial to satisfy computation_max %.3f, the cost of singular values discarded due to reduced_cost is %.3f '%(computation_max, computation_upper))

        if memory_max is not None:
            #assert 0 < memory_max < 1, 'invalid memory_max %s'%memory_max
            if memory_lower > memory_max:
                warnings.warn('~~~Impossible to satisfy memory_max %.3f, the cost of singular values included due to reduced_cost is %.3f '%(memory_max, memory_lower))
                if as_error:
                    raise ValueError
            if memory_upper < memory_max:
                warnings.warn('~~~Trivial to satisfy memory_max %.3f, the cost of singular values discarded due to reduced_cost is %.3f '%(memory_max, memory_upper))

    def call_gurobi_miqp(self, hessian_pickle_path='/tmp/hessian.pickle', computation_max=None, memory_max=None, monotonic=False, timelimit=None):
        '''load necessary data and call gurobi solver, for miqp problem (constrained computation/memory, minimize error)'''

        if hessian_pickle_path.endswith('.pickle'):
            with open(hessian_pickle_path, "rb") as f:
                   data = pickle.load(f)
            hessian = np.float64(data[0])
            gradient = np.float64(data[1])
        elif hessian_pickle_path.endswith('.mat'):
            mdict = scipy.io.loadmat(hessian_pickle_path)
            hessian = mdict['hessian']
            gradient = mdict['gradient']

        #check hessian shape
        assert gradient.shape[0]==hessian.shape[0], 'invalid hessian and gradient %s, %s'%(gradient.shape, hessian.shape)
        assert gradient.shape[0]==hessian.shape[1], 'invalid hessian and gradient %s, %s'%(gradient.shape, hessian.shape)
        assert gradient.shape[0] == self.get_num_reduced_mask_variables(), 'size of hessian loaded from pickle %s does not match num of reduced variables'%(gradient.shape, self.get_num_reduced_mask_variables())

        #check hessian positive definiteness
        #assert is_pd(hessian), 'hessian loaded not positive definite: %s'%(hessian_pickle_path)
        if not is_pd(hessian):
            warnings.warn('~~~Hessian is not positive definite!!!')
        
        num_constraints = int(computation_max is not None) + int(memory_max is not None)
        #assert num_constraints > 0, 'specify at least one of the constraints'
        num_variables = self.get_num_reduced_mask_variables()

        #check if the computation/memory constraint is feasiable given the reduce_index
        self.check_constraint_feasiability_against_reduced_index(computation_max, memory_max, as_error=False)

        A = list()
        sense = list()
        rhs = list()

        #name the pickle file in which the solution is saved
        #find the name of the neural network
        net_name = list(self.variable_to_index.keys())[0].op.name.split('/')[0].strip()
        pickle_solution_path = '/tmp/solution_' + net_name


        if computation_max is not None:
           assert 0 < computation_max < 1, 'invalid computation_max %s'%computation_max
           coeff = self.reduce_mask_variables_np(self.get_computation_cost_coefficient(), exact_size=False)
           assert len(coeff) == gradient.shape[0]
           A.append(coeff)
           sense.append(grb.GRB.GREATER_EQUAL)

           #all_on cost is different from total cost, total cost is the original cost
           #of the undecomposed network, all_on cost is the cost that all the singular
           #values are kept using the selected decomposition
           all_on_cost = sum(coeff)
           rhs.append(all_on_cost - computation_max*self.get_total_computation_cost() + self.compute_unaccounted_cost('computation', 'top'))
           #pickle_solution_path += '_comp_%.2f'%computation_max
        if memory_max is not None:
           assert 0 < memory_max < 1, 'invalid memory_max %s'%memory_max
           coeff = self.reduce_mask_variables_np(self.get_memory_cost_coefficient(), exact_size=False)
           assert len(coeff) == gradient.shape[0]
           A.append(coeff)
           sense.append(grb.GRB.GREATER_EQUAL)

           all_on_cost = sum(coeff)
           rhs.append(all_on_cost - memory_max*self.get_total_memory_cost() + self.compute_unaccounted_cost('memory', 'top') )
           #pickle_solution_path += '_mem_%.2f'%memory_max

        pickle_solution_path += '.pickle'

        print('---Calling miqp solver, num_variables: %d , computation_max: %s, memory_max: %s, solution_path: %s'%(num_variables, computation_max, memory_max, pickle_solution_path))

        # Optimize
        solution = miqp(num_constraints, num_variables, c=gradient, Q=hessian, A=A, sense=sense, rhs=rhs, timelimit=timelimit, solution_pickle_path=pickle_solution_path)

        return solution

    ############################################################################
    def call_cvx_sumlog_heuristics(self, computation_max=None, memory_max=None, timelimit=None, solution_pickle_path='/tmp/solution.pickle'):
        """
        solve Equation 13 and 14 in microsoft cvpr paper. 
        call cvx_sumlog_wrapper.m
        """
        
        #cvx_sumlog_wrapper(idx_vector, obj_coeff, comp_coeff, comp_rhs, mem_coeff, mem_rhs
        num_constraints = int(computation_max is not None) + int(memory_max is not None)
        assert num_constraints > 0, 'specify at least one of the constraints'

        #the number of reduced variables in each layer
        num_var_per_layer = list()
        #the coefficients of the binary decision variables, usually the squared singular value, but can be something else TODO
        obj_coeff = list()
        #the sum of a element-wise function of the singular values that are included due to reduced_index (the leading singular values) for each layer
        unaccounted_sv = list()
        ##in order to include more singular values at top layers, mandate the solution of a layer must be x percent of the total objective
        #layer_contribution_limit = list()

        ##DEBUG
        ##test assign higher weight to the top and bottom levels
        #high_exp = 1
        #low_exp = 3
        #num_layers = len(self.variable_to_singular_values)
        #exps = np.zeros([num_layers])
        #for i in range(len(exps)):
        #    exps[i] = high_exp/(((high_exp/low_exp)**(1/(num_layers-1)))**i)
        #i = 0
        for var, singular_values in self.variable_to_singular_values.items():
            reduced_index = self.variable_to_reduced_index[var]

            num_var_per_layer.append(reduced_index[1] - reduced_index[0])

            #manipulation the singular value, default is elementwise square
            #msv = [s**exps[i] for s in singular_values])
            msv = [s**2 for s in singular_values]
            #TODO in the greedy heuristic, the singular value seem to work better than the square of singular value
            #msv = [s for s in singular_values]

            #the get the singular values of the reduced variables
            obj_coeff.extend(msv[reduced_index[0]:reduced_index[1]])

            #add the unaccounted singular values, corresponding to the leading singular values
            unaccounted_sv.append(sum(msv[0:reduced_index[0]]) )

            #layer_contribution.append(sum(msv))

        #add a eps to avoid numerical issue in MOSEK solver
        unaccounted_sv = [v + 10*np.finfo(np.float32).eps for v in unaccounted_sv]

        ##emphasis the first few layers, by ensure that 
        #layer_contribution_limit = [lc/sum(obj_coeff) for lc in layer_contribution]
        #layer_contribution_limit = np.multiply(layer_contribution_limit, np.logspace(1, -10, num=len(layer_contribution_limit), base=2))


        assert len(num_var_per_layer) == len(unaccounted_sv)
        assert sum(num_var_per_layer) == self.get_num_reduced_mask_variables(), 'number of variable in each layer should match the sum'
        assert len(obj_coeff) == self.get_num_reduced_mask_variables()
        assert all([c >= 0 for c in obj_coeff]), 'modified singular values should all be positive'
        num_var_per_layer = matlab.double(num_var_per_layer)
        obj_coeff = matlab.double(obj_coeff)
        unaccounted_sv = matlab.double(unaccounted_sv)

        #the comp and mem coeff only consider the reduced_variables
        comp_coeff = matlab.double(list(self.reduce_mask_variables_np(self.get_computation_cost_coefficient()) ) )
        mem_coeff = matlab.double(list(self.reduce_mask_variables_np(self.get_memory_cost_coefficient()) ) )
        #-1 indicates the constraint is not active
        comp_rhs = -1
        mem_rhs = -1

        if computation_max is not None:
           #assert 0 < computation_max <= 1, 'invalid computation_max %s'%computation_max
           if not 0 < computation_max <= 1:
               print('DnnUtili: cvx_logsum computation_max %f'%computation_max)
           comp_rhs = computation_max*self.get_total_computation_cost() - self.compute_unaccounted_cost('computation','top')
        if memory_max is not None:
           #assert 0 < memory_max <= 1, 'invalid memory_max %s'%memory_max
           if not 0 < memory_max <= 1:
               print('DnnUtili: cvx_logsum memory_max %f'%memory_max)
           mem_rhs = memory_max*self.get_total_memory_cost() - self.compute_unaccounted_cost('memory','top')
        comp_rhs = float(comp_rhs)
        mem_rhs = float(mem_rhs)

        if timelimit is None:
            #-1 is no time limit in MOSEK
            timelimit = float(-1)
        else:
            timelimit = float(timelimit)

        eng = matlab.engine.start_matlab()
            
        #add path
        eng.addpath(r'/home/chongli/research/sparse',nargout=0)
        eng.addpath(r'/home/cad/gurobi750/linux64/matlab',nargout=0)
        eng.addpath(r'/home/cad/cvx',nargout=0)
        eng.addpath(r'/home/cad/mosek/8/toolbox/r2014a',nargout=0)

        mask_variable_value = eng.cvx_sumlog_wrapper(num_var_per_layer, obj_coeff, unaccounted_sv, comp_coeff, comp_rhs, mem_coeff, mem_rhs, timelimit)

        eng.quit()

        mask_variable_value = np.squeeze(mask_variable_value)
        assert len(mask_variable_value.shape) == 1
        assert mask_variable_value.shape[0] == self.get_num_reduced_mask_variables(), 'shape of returned vector %s does not match registered mask variables %d'%(mask_variable_value.shape, self.num_masking_variables)
        assert all([abs(m)< 1e-8 or abs(m-1) < 1e-8 for m in mask_variable_value]), 'returned vector should be binary values'

        #DEBUG 
        #self.print_solution(mask_variable_value, check_monotonic=True, print_value=False)

        #save result to pickle
        if solution_pickle_path is not None: 
            with open(solution_pickle_path, 'wb') as f:  
                pickle.dump(mask_variable_value, f, protocol=-1)
                
            mat_path = solution_pickle_path.replace('.pickle','.mat')
            scipy.io.savemat(mat_path, {'solution': np.array(mask_variable_value, dtype=np.float32)}, do_compression=True)
            
        #save a copy to /tmp/solution.pickle as well
        with open('/tmp/solution.pickle', 'wb') as f:  
            pickle.dump(mask_variable_value, f, protocol=-1)
        #save to mat format    
        scipy.io.savemat('/tmp/solution.mat', {'solution':np.array(mask_variable_value, dtype=np.float32)}, do_compression=True)

        print('---CVX sum_log solution saved in %s at %s'%(solution_pickle_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))

        #print(self.expand_reduced_mask_variables_np(mask_variable_value) )
        return mask_variable_value

    def save_coefficients_to_pickle(self, pickle_path='/tmp/cost_coeff.pickle'):
        '''save the coefficients of computation and memory cost to pickle.
        order is [computation_coefficient, total_computation, memory_coefficient, total_memory]'''

        #assert self.num_masking_variables > 0, 'cannot save a uninitialized object'
        if self.num_masking_variables == 0:
            warnings.warn('~~~~~~~~~~~~~~~~~~~~~~~~No masking variable is registered with the manager~~~~~~~~~')
            return

        #assuming in the solution 0 means keep the singular value and 1 means discard
        #only considered the reduced variables
        computation_coeff = np.array(self.reduce_mask_variables_np(self.get_computation_cost_coefficient(), exact_size=True), dtype=np.float64)
        memory_coeff = np.array(self.reduce_mask_variables_np(self.get_memory_cost_coefficient(), exact_size=True), dtype=np.float64)

        total_computation_cost = self.get_total_computation_cost()
        total_memory_cost = self.get_total_memory_cost()
        
        unaccounted_computation_top, unaccounted_computation_top_per_layer = self.compute_unaccounted_cost('computation', 'top', as_list=True)
        unaccounted_memory_top, unaccounted_memory_top_per_layer = self.compute_unaccounted_cost('memory', 'top', as_list=True)

        #save the number of (reduced) mask variables 
        num_var_per_layer = list()

        for var, singular_values in self.variable_to_singular_values.items():
            reduced_index = self.variable_to_reduced_index[var]
            num_var_per_layer.append(reduced_index[1] - reduced_index[0])

        #save the original cost of each layer
        original_computation_cost_per_layer = list()
        original_memory_cost_per_layer = list()
        for var, computation_cost in self.variable_to_total_computation.items():
            original_computation_cost_per_layer.append(computation_cost)
            original_memory_cost_per_layer.append(self.variable_to_total_memory[var])

        #save the object to pickle file
        content_to_save = (computation_coeff, memory_coeff, total_computation_cost, total_memory_cost, unaccounted_computation_top, 
                unaccounted_memory_top, num_var_per_layer, original_computation_cost_per_layer, original_memory_cost_per_layer,
                unaccounted_computation_top_per_layer, unaccounted_memory_top_per_layer)
        with open(pickle_path, 'wb') as f:  
            pickle.dump(content_to_save, f, protocol=-1)
        #save to mat as well
        mat_path = pickle_path.replace('.pickle','.mat')
        scipy.io.savemat(mat_path,{'computation_coeff':computation_coeff, 
            'total_computation_cost':total_computation_cost,'memory_coeff':memory_coeff,'total_memory_cost':total_memory_cost,
            'unaccounted_computation_top':unaccounted_computation_top,'unaccounted_memory_top':unaccounted_memory_top, 'num_var_per_layer':num_var_per_layer,
            'original_computation_cost_per_layer':original_computation_cost_per_layer, 'original_memory_cost_per_layer':original_memory_cost_per_layer,
            'unaccounted_computation_top_per_layer':unaccounted_computation_top_per_layer, 'unaccounted_memory_top_per_layer':unaccounted_computation_top_per_layer}, 
            do_compression=True)

    def save_variable_index_to_pickle(self, pickle_path='/tmp/variable_index.pickle'):
        '''save the full index of the variable to pickle, mapping from variable name to the index of the variable in the solution array'''
        var_name_to_index = OrderedDict()

        for var, index in self.variable_to_index.items():
            var_name_to_index[var.op.name] = index

        with open(pickle_path, 'wb') as f:  
            pickle.dump(var_name_to_index, f, protocol=-1)
        print('---variable index saved in %s at %s'%(pickle_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))

    def print_solution(self, solution, check_monotonic=True, print_value=False):
        '''print the solution returned by a heuristic or a solver
            in solution, 0 means include, 1 means discard singular value
        ''' 

        solution = self.expand_reduced_mask_variables_np(solution) 

        #first check if the solution is inverted or not(whether 1 indicates include singular value or 0 indicates include singular value)
        for var, index in self.variable_to_index.items():
            #solution in this layer
            sl = solution[index[0]:index[1]]
            if sl[0] == 1:
                warnings.warn('the first singular value in variable %s is not included'%var.op.name)

        #get all the singular values
        #TODO the singular value here need to match how the singular values are manipuated in call_cvx_sumlog and greedy_heuristic
        all_singular_values = np.concatenate(list(self.variable_to_singular_values.values()) )
        all_singular_values = [v**2 for v in all_singular_values]
        all_singular_values = np.array(all_singular_values)
        #square the singular values, matching the cvx_sumlog heuristic
        #all_singular_value = all_singular_values ** 2
        assert all_singular_values.shape[0] == solution.shape[0]

        #recompute optimal obj value defined by product of sum of select singular values
        #the solution is inverted, because 0 stand for include
        dss = np.multiply(1-solution, all_singular_values)

        print('~~~In print_solution()')
        obj = 1.0
        for var, index in self.variable_to_index.items():
            #solution in this layer
            sl = solution[index[0]:index[1]]
            if check_monotonic:
                assert all(sl[i] <= sl[i+1] for i in range(len(sl)-1)), 'in layer %s (index: %s), a singular value should not be discarded before all the values smaller than it are discarded %s'%(var.op.name, str(index), str(sl))
            obj *= np.sum(dss[index[0]:index[1]])

            #print the index of the first singular value that is discarded
            if np.any(sl==1):
                truncated_index = np.where(sl == 1)[0][0]
            else:
                truncated_index = 'UNTRUNCATED'
            print('~~~Variable: %s, singular value truncated at %s'%(str(var), truncated_index))
            if print_value:
                print('~~~%s'%str(sl))

        #DEBUG TODO obj value depends on how the singular value is manipualted
        print('~~~~obj value computed in print_solution is %.6E'%obj)
        
    def cross_entropy_rounding(self, solution, computation_max=None, memory_max=None, monotonic=False):
        '''use cross entropy rounding to convert a solution from sqplab to binary'''
        self.check_constraint_feasiability_against_reduced_index(computation_max, memory_max, as_error=False)

        solution = matlab.double(solution.tolist())

        comp_coeff = matlab.double(self.reduce_mask_variables_np(self.get_computation_cost_coefficient(), exact_size=False).tolist())
        if computation_max is not None:
            assert 0 < computation_max < 1, 'invalid computation_max %s'%computation_max
            all_on_cost = np.sum(comp_coeff)
            comp_rhs = all_on_cost - computation_max*self.get_total_computation_cost() + self.compute_unaccounted_cost('computation', 'top')
        else:
            comp_rhs = -1.0
        comp_rhs = float(comp_rhs)

        mem_coeff = matlab.double(self.reduce_mask_variables_np(self.get_memory_cost_coefficient(), exact_size=False).tolist())
        if memory_max is not None:
           assert 0 < memory_max < 1, 'invalid memory_max %s'%memory_max
           all_on_cost = np.sum(mem_coeff)
           mem_rhs = all_on_cost - memory_max*self.get_total_memory_cost() + self.compute_unaccounted_cost('memory', 'top')
        else:
           mem_rhs = -1.0
        mem_rhs = float(mem_rhs)

        eng = matlab.engine.start_matlab()
            
        #add path
        eng.addpath(r'/home/chongli/research/sparse',nargout=0)
        eng.addpath(r'/home/cad/gurobi750/linux64/matlab',nargout=0)
        eng.addpath(r'/home/cad/mosek/8/toolbox/r2014a',nargout=0)
        eng.addpath(r'/home/cad/cvx',nargout=0)
        solution_binary = eng.cross_entropy_rounding(solution, comp_coeff, comp_rhs, mem_coeff, mem_rhs)

        eng.quit()

        solution_binary = np.squeeze(solution_binary)

        return solution_binary

    def compute_delta_cost_per_layer(self, solution, solution2=None):
        '''compute the to the reduction in computation cost (reduction from solution in solution 2, to 
        the solution in solution)

        in solution and solution2, 0 means included, 1 means discarded
        '''

        assert FLAGS.enable_reduction is False
        assert solution is not None

        if type(solution) is str:
            solution = scipy.io.loadmat(solution)['x'].squeeze()
        assert type(solution) is np.ndarray

        #compute the computation cost defined by solution2
        if solution2 is None:
            #if no solution 2 is provided, use all on 
            solution2 = np.zeros([mvm.get_num_mask_variables()],dtype=np.float32)
        elif type(solution2) is str:
            solution2 = scipy.io.loadmat(solution2)['x'].squeeze()
        assert type(solution2) is np.ndarray

        solution = self.expand_reduced_mask_variables_np(solution, exact_size=False)
        solution2 = self.expand_reduced_mask_variables_np(solution2, exact_size=False)

        #invert the solution, at this point 1 means included
        solution = 1 - solution;
        solution2 = 1 - solution2;

        assert len(solution.shape) == 1, 'expecting input is a vector, but %s' % solution 
        assert len(solution2.shape) == 1, 'expecting input is a vector, but %s' % solution 
        assert not np.isnan(np.sum(solution)), 'Cannot have any NaN in the solution'
        assert not np.isnan(np.sum(solution2)), 'Cannot have any NaN in the solution'

        #TODO could be memory cost as well 
        cost_coeff = self.get_computation_cost_coefficient()
        delta_cost = np.multiply(solution2 - solution, cost_coeff)

        solution_cost = np.multiply(solution, cost_coeff)

        #initialize the dict mapping from the name of the variable, to the change in cost
        variable_to_delta_cost = OrderedDict()
        variable_to_cost_drop_percentage = OrderedDict()
       
        for var, index_range in self.variable_to_index.items():
            variable_to_delta_cost[var.name] = np.sum(delta_cost[index_range[0]:index_range[1]])

            solution_cost_layer =  np.sum(solution_cost[index_range[0]:index_range[1]])
            #TODO could be memory cost as well 
            original_cost_layer = self.variable_to_total_computation[var]

            variable_to_cost_drop_percentage[var.name] = 1 - solution_cost_layer/original_cost_layer

            if FLAGS.cost_saturation:
                assert np.all(solution2 == 1) , 'comparing two solutions with cost saturation is not supported yet'
                if solution_cost_layer >= original_cost_layer:
                    variable_to_delta_cost[var.name] = 0.0
                    variable_to_cost_drop_percentage[var.name] = 0.0

        #save a copy to pickle
        with open('/tmp/delta_cost_per_layer.pickle', 'wb') as f:  
            pickle.dump((variable_to_delta_cost, variable_to_cost_drop_percentage, solution, solution2),f, protocol=-1)

        print('DnnUtili.compute_delta_cost_per_layer: saving /tmp/delta_cost_per_layer.pickle')

        return variable_to_delta_cost, variable_to_cost_drop_percentage

    def add_and_svd_K(self, name, K_info, pickle_path='/tmp/add_and_svd_K.pickle'):
        #record the add_and_svd_K used for each layer, name of layer (sc.name) maps to
        #[add_and_svd_K, num_computation_per_K, original_computation_cost,num_memory_per_K, original_memory_cost, add_and_svd_K+reduced_index[0]]

        assert len(K_info)==6

        assert name not in self.add_and_svd_K_dict,'layer %s already in add_and_svd_K_dict'%name
        self.add_and_svd_K_dict[name] = K_info

        with open(pickle_path, 'wb') as f:  
            pickle.dump(self.add_and_svd_K_dict, f, protocol=-1)



#initialize a variable manager object
mvm = MaskingVariableManager()
#############################################################################################

def calculate_percentage_add_and_svd(computation_max, memory_max, check_satisfied=False, cost_saturation=None, pickle_path='/tmp/add_and_svd_K.pickle'):
    #in the pickle file, name of layer (sc.name) maps to
    #[add_and_svd_K, num_computation_per_K, original_computation_cost,num_memory_per_K, original_memory_cost, add_and_svd_K+reduced_index[0]]
    #return true if the cost is smaller than the constraints

    with open(pickle_path, 'rb') as f:  
        content = pickle.load(f)
    assert type(content) is OrderedDict, 'got %s from add_and_svd_K.pickle'%type(content)

    used_computation = 0
    total_computation = 0
    used_memory = 0
    total_memory = 0

    try:
        use_cost_saturation = FLAGS.cost_saturation
    except:
        assert cost_saturation is not None
        use_cost_saturation = cost_saturation 

    for key, value in content.items():
        assert len(value)==6
        if use_cost_saturation:
            if value[5]*value[1] >= value[2] or value[5]*value[3] >= value[4]:
                #use the un-decomposed layer if the cost of the decomposed layer is higher than the original
                used_computation +=  value[2]
                used_memory +=  value[4]
            else:                
                used_computation += value[5]*value[1]
                used_memory += value[5]*value[3]
        else:
            used_computation += value[5]*value[1]
            used_memory += value[5]*value[3]

        total_computation += value[2]
        total_memory += value[4]

    computation_percentage = used_computation/total_computation
    memory_percentage = used_memory/total_memory

    print('DnnUtili:check_add_and_svd_constraint: used computation: %.5f, used memory: %.5f'%(computation_percentage, memory_percentage))

    if check_satisfied:
        assert computation_max is not None and memory_max is not None, 'computation_max is not provided: %s'%computation_max
        satisfied =  computation_percentage <= 1.01*computation_max and  memory_percentage <= 1.01*memory_max
        return computation_percentage, memory_percentage, satisfied
    else:
        return computation_percentage, memory_percentage

def draw_delta_cost_per_layer(color_map = 'summer', value_range=None, pickle_path='/tmp/delta_cost_per_layer.pickle', draw_path='/tmp/delta_cost_per_layer.pdf'):
    '''make a plot of the topology of the network, and color each block according to the change in cost 
    across the two solutions. '''

    #python3.5 mask_wrapper.py --model_name squeezenet --enable_reduction 1 --load_reduced_index 1 --compute_delta_cost_per_layer_solution /tmp/sqp_solution.mat --gpu 0,1

    color_map = matplotlib.cm.get_cmap(color_map)
    #color_map = matplotlib.cm.get_cmap('Spectral')
    #color_map = matplotlib.cm.get_cmap('summer')

    #################################
    with open(pickle_path, 'rb') as f:  
        content = pickle.load(f)

    #variable_to_delta_cost, variable_to_cost_drop_percentage = self.compute_delta_cost_per_layer(solution, solution2)
    variable_to_delta_cost, variable_to_cost_drop_percentage = content[0], content[1]

    #find the model_name
    model_name = list(variable_to_delta_cost.keys())[0].split('/')[0].strip()
    assert model_name in ['squeezenet','CifarNet'], 'unknown model'

    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    #control edge length
    dot.set('ranksep', 0.3)

    values = np.array(list(variable_to_cost_drop_percentage.values()))

    #the range for the normalizer
    if value_range is None:
        vmin=min(1-values)
        vmax=max(1-values)
    else:
        vmin = value_range[0]
        vmax = value_range[1]
    #assert vmin < vmax and -1<=vmin and vmax <=1.1

    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    var_name_to_node = OrderedDict()

    for var_name, value in variable_to_cost_drop_percentage.items():
        #example is squeezenet/conv1/S_masks
        #remove leading model name
        label = var_name.lstrip('%s/'%model_name)
        #remove trailing S_masks
        label = label.replace('/S_masks:0','')
        label = '%s\n%.1f%%' % (label,100.0*variable_to_cost_drop_percentage[var_name])

        #DEBUG
        #assert(0<=value<=1)
        
        normalized_value = normalizer(value)
        #normalized_value = normalizer(1-value)
        
        color = color_map(normalized_value)
        color = color[0:3]
        color = [int(c*255) for c in color]
        color = '#%02x%02x%02x' % (color[0], color[1], color[2])
        dot.add_node(pydot.Node(var_name, label=label, style='filled', fillcolor=color) )

    #add edge
    if model_name == 'squeezenet':
        #conv1 to the first fire module
        dot.add_edge(pydot.Edge('squeezenet/conv1/S_masks', 'squeezenet/fire2/squeeze/S_masks')) 

        for i in range(2,10):
            dot.add_edge(pydot.Edge('squeezenet/fire%d/squeeze/S_masks'%i, 'squeezenet/fire%d/expand/1x1/S_masks'%i)) 
            dot.add_edge(pydot.Edge('squeezenet/fire%d/squeeze/S_masks'%i, 'squeezenet/fire%d/expand/3x3/S_masks'%i)) 

            if i < 9:
                dot.add_edge(pydot.Edge('squeezenet/fire%d/expand/1x1/S_masks'%i, 'squeezenet/fire%d/squeeze/S_masks'%(i+1))) 
                dot.add_edge(pydot.Edge('squeezenet/fire%d/expand/3x3/S_masks'%i, 'squeezenet/fire%d/squeeze/S_masks'%(i+1))) 
            else:
                dot.add_edge(pydot.Edge('squeezenet/fire%d/expand/1x1/S_masks'%i, 'squeezenet/conv10/S_masks'))
                dot.add_edge(pydot.Edge('squeezenet/fire%d/expand/3x3/S_masks'%i, 'squeezenet/conv10/S_masks'))
    else:
        raise ValueError('Unknown model_name: %s'%model_name)
    print('DnnUtili:draw_delta_cost_per_layer:  output saved to %s'%draw_path)
    dot.write_pdf(draw_path)


import sys
try:
    #from gurobipy import *
    import gurobipy as grb
except ImportError:
    print('DnnUtili: cannot import gurobipy')
    pass

from datetime import datetime
############################################################################
def miqp(rows, cols, c, Q, A, sense, rhs, timelimit=None, solution_pickle_path='/tmp/solution.pickle'):
    '''
    rows is the number of constraints
    columns is the number of optimization variables
    c is linear term in the objective
    A is the linear constraint term (matrix, not vector, even if with only 1 constraint)
    sense is the relationship (greater or smaller)
    rhs is the right handside of the constraints
    '''
    
    #check input data
    assert len(c) == cols, 'length of linear objective does not match number of variable'
    assert len(Q) == cols, 'size of quadratic objective does not match number of variable'
    assert len(Q[0]) == cols, 'size of quadratic objective does not match number of variable'
    assert len(A) == rows, 'size of linear constraint does not match number of constraint'
    assert len(A[0]) == cols, 'size of linear constraint does not match number of constraint'
    assert len(sense) == rows, 'size of linear constraint does not match number of constraint'
    assert len(rhs) == rows, 'size of linear constraint does not match number of constraint'

    #there seems to be a big advantage in not doing the small value truncation, but why? TODO
    #set small values to zero in Q
    #Q = np.array(Q, dtype=np.float64)
    #tolerance = np.finfo(np.float32).eps
    #Q[np.abs(Q)<tolerance] = np.float64(0.0)

    #add small value to the diagnoal elements
    #Q += (np.identity(len(Q))*tolerance)
        
    model = grb.Model()
    
    #set timelimit in second
    if timelimit is not None:
        model.Params.TIME_LIMIT = timelimit

    #model.Params.Logfile = '/tmp/gurobi.log'

    #set termination criteria, x% of optimality gap
    #model.Params.MIPGap = 0.02
    #model.Params.MIPGap = 0.01

    #DEBUG
    #do not print to stdout
    #model.Params.OutputFlag = 0
    
    # Add variables to model
    vars = []
    for j in range(cols):
        #vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))
        vars.append(model.addVar(vtype=grb.GRB.BINARY))
    
    #with Timer('Populate A matrix'):
    # Populate A matrix
    for i in range(rows):
        expr = grb.LinExpr()
        for j in range(cols):
            if A[i][j] != 0:
                expr += A[i][j]*vars[j]
        model.addConstr(expr, sense[i], rhs[i])
    
    #with Timer('Populate Q matrix'):
    # Populate objective
    obj = grb.QuadExpr()

    coeff_list = []
    v1_list = []
    v2_list = []
    for i in range(cols):
        for j in range(cols):
            if Q[i][j] != 0:
                #/2 because of the 1/2 term due to series expansion
                coeff_list.append(Q[i][j]*0.5)
                v1_list.append(vars[i])
                v2_list.append(vars[j])

    #quadratic objective
    obj.addTerms(coeff_list, v1_list, v2_list)
    #linear objective
    obj.addTerms(c, vars)

    model.setObjective(obj)

    ##tune the model
    ##tune criterion is best feasible solution, -1 to choose automatically
    #model.Params.TuneCriterion = 2
    ##tune time limit
    ##model.Params.TuneTimeLimit = 3600*2
    #model.Params.TuneTimeLimit = 360
    ##keep only best tuning parameters
    #model.Params.TuneResults = 1
    #model.tune()
    #
    ## Solve
    #if model.tuneResultCount > 0:
    #    model.getTuneResult(0)
    #    model.write('/tmp/tune.prm')
    #else:
    #    warnings.warn('---Gurobi tune did not improve over baseline parameter')

    model.optimize()
   
    #initialize the solution vector
    solution = np.full((cols),None, dtype=np.float32)
    
    if model.status in [grb.GRB.Status.OPTIMAL, grb.GRB.Status.INTERRUPTED, grb.GRB.Status.TIME_LIMIT, grb.GRB.Status.SUBOPTIMAL]:
        print('--------------------grb returned %s'%model.status)
        x = model.getAttr('x', vars)
        for i in range(cols):
            solution[i] = x[i]

        # Write model to a file
        #model.write('/tmp/miqp.lp')

        #save result to pickle
        if solution_pickle_path is not None: 
            with open(solution_pickle_path, 'wb') as f:  
                pickle.dump(solution, f, protocol=-1)
            mat_path = solution_pickle_path.replace('.pickle','.mat')
            scipy.io.savemat(mat_path, {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)
            
        #save a copy to /tmp/solution.pickle as well
        with open('/tmp/solution.pickle', 'wb') as f:  
            pickle.dump(solution, f, protocol=-1)
        #save to mat format
        scipy.io.savemat('/tmp/solution.mat', {'solution': np.array(solution, dtype=np.float64)}, do_compression=True)

        print('---gurobi miqp solution saved in %s at %s'%(solution_pickle_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))

        return solution
    else:
        raise ValueError('----Gurobi call failed with exit code %d'%model.status)

def is_pd(K):
    '''test if a matrix is positive semi-definite'''
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in str(err):
            return False
        else:
            raise 

def solution_random_rounding(sqp):
    '''convert the solution from SQP (variables are continuous 0-1) to a binary vector'''

    assert sqp.shape[1] == 1, 'expected a column vector, got %s'%sqp.shape
    sqp = np.reshape(sqp, (sqp.shape[0]))

    #initialize solution
    solution = np.full(sqp.shape, np.float32(None), np.float32)

    #random rounding
    for idx,m in enumerate(sqp):
        solution[idx] = np.float32(np.random.uniform() < m)

    return solution

def copy_rename(src_dir,old_file_name, to_folder, new_file_name):
    '''copy and rename a file.'''

    dst_dir= to_folder
    src_file = os.path.join(src_dir, old_file_name)
    shutil.copy(src_file,dst_dir)

    dst_file = os.path.join(dst_dir, old_file_name)
    new_dst_file_name = os.path.join(dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)

def latest_checkpoint(solution_path):
    '''given a path, return the absolute path of the latest checkpoint in that path'''

    if not os.path.isdir(solution_path):
        return None

    list_file = os.path.join(solution_path,'checkpoint')
    if not os.path.isfile(list_file):
        return None

    with open(list_file,'r') as f:
        lines = f.readlines()
    line = lines[0]
    checkpoint_path_decomposed = re.findall(r'"([^"]*)"', line)
    assert len(checkpoint_path_decomposed) == 1
    checkpoint_path_decomposed = checkpoint_path_decomposed[0]
    checkpoint_path_decomposed = os.path.join(solution_path, os.path.split(checkpoint_path_decomposed)[-1])

    if not os.path.isfile(checkpoint_path_decomposed+'.meta'):
        return None

    return checkpoint_path_decomposed 
