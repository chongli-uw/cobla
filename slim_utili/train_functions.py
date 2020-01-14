#Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic training script that trains a model using a given dataset."""




import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training import saver as tf_saver

import copy
import pickle
import numpy as np

import os
import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import TFInclude
import DnnUtili
import time

slim = tf.contrib.slim

mvm = DnnUtili.mvm

import numpy as np
from collections import OrderedDict
######################
# Chong's Flags #
######################
tf.app.flags.DEFINE_boolean('resume_training_from_train_dir', False , 
                            'True: start training from checkpoint file --checkpoint_path'
                            'False: resume training from --train_dir')
tf.app.flags.DEFINE_boolean('clear_train_dir', False , 'whether to delete the train_dir before start a new training') 
tf.app.flags.DEFINE_string('init_op_dict_path', '/tmp/init_op_dict.pickle','path used by assign_init_op_dict() function to save the initial dict')
tf.app.flags.DEFINE_string('trainable_exclude_scopes', None,'Comma-separated list of scopes to filter the set of variables '
        'to be excluded from train.''By default, None would exclude any variable from training.')

tf.app.flags.DEFINE_boolean('load_solution', False , 'whether to load a gurobi solution in FLAGS.solution_path') 
tf.app.flags.DEFINE_string('solution_path', None, 'the path to the solution of optimal masking variable values')
tf.app.flags.DEFINE_float('computation_max', None, 'percentage of maximum computation cost')
tf.app.flags.DEFINE_float('memory_max', None, 'percentage of maximum memory cost')

tf.app.flags.DEFINE_integer('K_heuristic', None, 'heuristics to use in get_mask_variable_value_using_heuristics() in MaskVariableManager')

tf.app.flags.DEFINE_boolean('enable_reduction', False, 'whether to only consider a subset of the binary values in each variable, and fix the top part to 0, bottom part to 1') 
tf.app.flags.DEFINE_boolean('load_reduced_index', False, 'whether to load reduced_index from /tmp/reduced_index.pickle in compute_reduced_index()') 
tf.app.flags.DEFINE_boolean('add_and_svd_rounding', False, 'convert a solution of continuous to integer using add_and_svd') 
tf.app.flags.DEFINE_boolean('load_add_and_svd_K', False, 'whether to load from /tmp/add_and_svd_K.pickle to decide the add_and_svd_K for each layer') 
tf.app.flags.DEFINE_boolean('eval_fine_tuned_decomposition', False, 'whether to evaluate the decomposition after fine tuning') 
tf.app.flags.DEFINE_boolean('is_training', True, 'indicate whether is training or evaluating') 

tf.app.flags.DEFINE_string('train_feed_dict_path', None, 'the path to the feed_dict for the values of the mask variables, the pickle file is dumped from a previous evaluation')

tf.app.flags.DEFINE_string('decomposition_scheme', 'tai', 'the decomposition scheme to use in my_slim_layer')
#tf.app.flags.DEFINE_string('decomposition_scheme', 'microsoft', 'the decomposition scheme to use in my_slim_layer')

tf.app.flags.DEFINE_boolean('cost_saturation', False, 'if the cost of a layer in the solution is higher than the original cost, then use the un-decomposed layer') 
######################
# Flags #
######################
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 3,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 32,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 32,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 500,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

#tf.app.flags.DEFINE_string(
#    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


######################
# Chong's functions #
######################
def set_train_dataset_flags(model_name, train_dir = None, fine_tune=True, learning_rate=None, batch_size=None, optimizer=None, checkpoint_path=None, number_of_steps=None, number_of_epochs=None):
    '''set dataset dependent flags
    Args: Nothing, reads FLAGS.dataset
    '''
    assert type(model_name) is str, 'model_name has to be a string'
    FLAGS.model_name=model_name

    if train_dir is None:
        FLAGS.train_dir='/tmp/tensorflow_train'
    else:
        FLAGS.train_dir=train_dir
    
    model_name = FLAGS.model_name
    if model_name.startswith('cifarnet'):
        FLAGS.dataset_dir='/home/chongli/dataset/cifar10'
        FLAGS.dataset_name='cifar10'
    else:
        FLAGS.dataset_dir='/home/chongli/dataset/imagenet-data'
        FLAGS.dataset_name='imagenet'

    FLAGS.dataset_split_name='train'
    
    if not FLAGS.model_name:
        raise ValueError('You must set model_name FLAG with --model_name')

    #get the number of GPUs
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(',')
    if gpus == ['']:
        print('train_function: no gpu specified, string is %s'%os.environ["CUDA_VISIBLE_DEVICES"])
        gpus = ['CPU']
    else:
        assert all([g.isdigit() for g in gpus]), 'invalud gpu string : %s'%os.environ["CUDA_VISIBLE_DEVICES"]
    FLAGS.num_clones=len(gpus)
    
    if model_name.startswith('inception_v2'):
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v2.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=64
    elif model_name.startswith('inception_v3'):
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v3.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=64
    elif model_name.startswith('inception_v4'):
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v4.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=64
    elif model_name.startswith('alexnet'):
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name.startswith('cifarnet'):
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/cifar10/model.ckpt-1'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'squeezenet':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/squeezenet/model.ckpt-1'
        FLAGS.batch_size=512
        FLAGS.labels_offset=1
    elif model_name == 'squeezenet_bn':
        #FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/squeezenet/model.ckpt-1'
        FLAGS.checkpoint_path=None
        FLAGS.batch_size=512
        FLAGS.labels_offset=1
    elif model_name == 'mobilenet_v1':
        #INFO, need to specify the filename.ckpt. Specifying the path, orspecifying the file_name.ckpt.data will not work
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet/mobilenet_v1_1.0_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'mobilenet_v1_075':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_075/mobilenet_v1_0.75_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'mobilenet_v1_050':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_050/mobilenet_v1_0.50_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'mobilenet_v1_025':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_025/mobilenet_v1_0.25_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'nasnet_mobile':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/nasnet_mobile/model.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=32
    elif model_name == 'vgg_16':
        FLAGS.labels_offset=1
        FLAGS.batch_size=64
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/vgg_16.ckpt'
    else:
        raise ValueError('Unknown model_name')
#     if model_name.startswith('lenet'):
#         FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v3.ckpt'

    if batch_size is not None:
        FLAGS.batch_size=batch_size

    #assert FLAGS.batch_size % FLAGS.num_clones == 0, 'batch_size: %d, num_clones: %d, cannot be evenly divided'%(FLAGS.batch_size, FLAGS.num_clones)
    FLAGS.batch_size=int(np.round(FLAGS.batch_size/FLAGS.num_clones)*FLAGS.num_clones)

    #according to 
    #https://github.com/tensorflow/models/issues/1428, https://github.com/tensorflow/models/issues/2086
    #batch size is per GPU batch size
    FLAGS.batch_size=int(np.round(FLAGS.batch_size/FLAGS.num_clones))

    if optimizer is not None:
        FLAGS.optimizer=optimizer
    else:
        FLAGS.optimizer='rmsprop'

    if learning_rate is None:
        FLAGS.learning_rate = 0.01
    else:
        FLAGS.learning_rate = learning_rate

    #if not fine tuning, should restart training from scratch and not load any variables
    if fine_tune is False:
        FLAGS.checkpoint_path = None
    else:
    #if fine tuning, decay the learning rate more often
        FLAGS.num_epochs_per_decay = 1.0

    if checkpoint_path is not None:
        FLAGS.checkpoint_path = checkpoint_path

    if number_of_steps is not None and number_of_epochs is not None:
        raise ValueError('Cannot specify both number of step and number of epoch')

    if number_of_steps is not None:
        FLAGS.max_number_of_steps = number_of_steps
    if number_of_epochs is not None:
        #get the dataset
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        FLAGS.max_number_of_steps = max(int(float(number_of_epochs)*dataset.num_samples/(float(FLAGS.batch_size)*float(FLAGS.num_clones)) ) , 1)
        del dataset

def print_training_parameters():
    #print summary of training parameters
    print('-----------------------------------')
    print('-----Summary of Training Parameters')
    print('-----------------------------------')
    print('model_name: %s'%FLAGS.model_name)
    print('train_dir: %s'%FLAGS.train_dir)
    print('checkpoint_path: %s'%FLAGS.checkpoint_path)
    print('optimizer: %s'%FLAGS.optimizer)
    print('learning_rate: %.5E'%FLAGS.learning_rate)
    print('batch_size: %d'% FLAGS.batch_size)
    print('ignore_missing_vars: %d'%FLAGS.ignore_missing_vars)
    print('maximum number of steps: %s'%str(FLAGS.max_number_of_steps))
    print('num_clones: %s'%str(FLAGS.num_clones))

    print('load_solution: %s'%FLAGS.load_solution)
    print('solution_path: %s'% FLAGS.solution_path)
    print('computation_max: %s'% FLAGS.computation_max)
    print('memory_max: %s'% FLAGS.memory_max)
    print('K_heuristic: %s'% FLAGS.K_heuristic)
    print('enable_reduction: %s'% FLAGS.enable_reduction)
    print('-----------------------------------')

######################
def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer=='yellowfin':
    from yellowfin import YFOptimizer
    if learning_rate is None:
        optimizer=YFOptimizer()
    else:
        assert type(learning_rate) is float, 'yellowfin uses a numerical learning_rate'
        optimizer=YFOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.summary.histogram(variable.op.name, variable))
  summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
  return summaries


def _get_init_fn(variable_value_dict = None):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None and variable_value_dict is None:
    return None

# # Warn the user if a checkpoint exists in the train_dir. Then we'll be
# # ignoring the checkpoint anyway.
# if tf.train.latest_checkpoint(FLAGS.train_dir):
#   tf.logging.info(
#       'Ignoring --checkpoint_path because a checkpoint already exists in %s'
#       % FLAGS.train_dir)
#   return None

    if FLAGS.resume_training_from_train_dir:
        assert tf.train.latest_checkpoint(FLAGS.train_dir), 'Asked for resume_training_from_train_dir but there is no valid checkpoint file in train_dir'
        return None
           
  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  #DEBUG
  variables_to_restore_name = [var.op.name for var in variables_to_restore]
  variables_to_restore_name = sorted(variables_to_restore_name)
  print('---------Variables to restore: %s------------\n'%str(variables_to_restore_name))
  #print('---------dict contains %s'% str(sorted(variable_value_dict)))

  #initialize a checkpoint reader
  if FLAGS.checkpoint_path is not None:
      reader = DnnUtili.get_checkpoint_reader(checkpoint_path = FLAGS.checkpoint_path)

      #DEBUG
      #print('--------checkpoint contains %s'% str(sorted(reader.get_variable_to_shape_map() )))
  else:
      reader = None

  #from variable names to values
  var_names_to_values = {}

  for var in variables_to_restore:
      #get the name of the variable
      var_name = var.op.name

      #initialize variable
      var_value = None

      #read from checkpoint file
      if reader is not None:
          if reader.has_tensor(var_name):
              var_value = DnnUtili.get_tensor_from_checkpoint(var_name, reader=reader)
              print('----loading value from checkpoint for variable: %s'%var_name)

      #if the the input variable_value_dict also specified the value of this variable
      if variable_value_dict is not None:
          if var_name in variable_value_dict:
              if var_value is None:
                  print('----loading value from dict for variable: %s'%var_name)
              else:
                  print('----overwriting value from dict over value from checkpoint for variable: %s'%var_name)
              var_value = variable_value_dict[var_name]
      
      #make sure a value is assigned
      if var_value is None:
          if FLAGS.ignore_missing_vars:
              print('--------variable: %s, is initialized using the initializer of the variable'%var_name)
          else:
              raise ValueError('No value for variable: %s in either checkpoint file or input dict'%var_name)
      else: 
          #check the size match
          assert var.get_shape().as_list() == list(var_value.shape), 'size of %s should be %s, got %s'%(var.op.name, 
                  var.get_shape().as_list(), var_value.shape)
          var_names_to_values[var_name] = var_value

  #tf.logging.info('Fine-tuning from %s' % checkpoint_path)
  tf.logging.info('------Fine-tuning---------')
  
  #TODO, need to modify variables loaded from a checkpoint to provide a reasonable training starting point

  return slim.assign_from_values_fn(var_names_to_values)

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """

  if FLAGS.trainable_scopes is not None and FLAGS.trainable_exclude_scopes is not None:
      raise ValueError('Cannot specify trainable_scopes and trainable_exclude_scopes at same time')

  #if both none, return all trainable variables
  if FLAGS.trainable_scopes is None and FLAGS.trainable_exclude_scopes is None:
      print('------------Training all model variables-------')
      return tf.trainable_variables()

  #if only scopes to train is specified
  if FLAGS.trainable_scopes:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_train.extend(variables)
  elif FLAGS.trainable_exclude_scopes:
  #if only variables to be excluded from training is specified
    #start with all trainable variables
    variables_to_train = tf.trainable_variables()

    exclusions = [scope.strip() for scope in FLAGS.trainable_exclude_scopes.split(',')]
  
    # TODO(sguada) variables.filter_variables()
    #make a copy of the variables to train
    variables_to_train_no_exclusion = list(variables_to_train)
    variables_to_train = []
    for var in variables_to_train_no_exclusion:
      excluded = False
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          excluded = True
          break
      if not excluded:
        variables_to_train.append(var)
  else:
      raise ValueError()

  #get the names of the variables to train
  variable_names = [var.op.name for var in variables_to_train]
  print('------------------Variables to train: %s'%str(variable_names))

  return variables_to_train

def train(variable_value_dict=None):

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  assert not (FLAGS.is_training and FLAGS.cost_saturation)

  #Chong added
  print_training_parameters()

  #Chong added
  #get the initial value dict, will be used by assign_from_feed_dict function
  if variable_value_dict is None:
      if not mvm.is_empty():
          #need to change to feed_dict of masking variables?
          raise NotImplementedError
          variable_value_dict = mvm.get_variable_name_to_initial_value_dict()

  #Chong added
  #if user didn't request to resume an ongoing training stored in train_dir
  if FLAGS.resume_training_from_train_dir is not True or FLAGS.K_heuristic is not None: 
      #clear the training folder
      import shutil
      shutil.rmtree(FLAGS.train_dir, ignore_errors=True)
      os.sync()
        
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    ######################
    # Config model_deploy#
    ######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the network #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          shuffle=True,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)

      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      with tf.device(deploy_config.inputs_device()):
        images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        tf.losses.softmax_cross_entropy(
            logits=end_points['AuxLogits'], onehot_labels=labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
      tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      #summaries.add(tf.summary.histogram('activations/' + end_point, x))
      #summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    #for variable in slim.get_model_variables():
      #summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate=(FLAGS.learning_rate if FLAGS.optimizer=='yellowfin' else learning_rate))
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    #session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    session_config = tf.ConfigProto(allow_soft_placement=True)

    #set saver parameters
    saver = tf_saver.Saver(max_to_keep = 2, write_version = 2)

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
    #my_slim_learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        #Chong added
        #init_op = assign_ops,
        #init_feed_dict = assign_feed_dict,
        saver = saver, 
        #Chong added end
        init_fn=_get_init_fn(variable_value_dict),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        session_config = session_config,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)

