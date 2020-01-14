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
"""Generic evaluation script that evaluates a model using a given dataset."""



import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

import numpy as np
import scipy
import scipy.io
import pickle
from collections import OrderedDict
import shutil
import os
import time
import copy

import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')
import DnnUtili
######################
# Chong's Flags #
######################
#MaskingVariableManager
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('call_gurobi', False , 'whether to call gurobi solver to find a new solution') 
tf.app.flags.DEFINE_boolean('load_solution', False , 'whether to load a gurobi solution in FLAGS.solution_path') 
tf.app.flags.DEFINE_string('solution_path', '/tmp/solution.pickle', 'the path to the solution of optimal masking variable values')
tf.app.flags.DEFINE_string('hessian_pickle_path', '/tmp/hesssian.pickle', 'the path to the hessian and gradient pickle file')
tf.app.flags.DEFINE_float('computation_max', None, 'percentage of maximum computation cost')
tf.app.flags.DEFINE_float('memory_max', None, 'percentage of maximum memory cost')
tf.app.flags.DEFINE_integer('timelimit', None, 'The timelimit after which a gurobi call is terminated.')

tf.app.flags.DEFINE_integer('K_heuristic', None, 'heuristics to use in get_mask_variable_value_using_heuristics() in MaskVariableManager')

tf.app.flags.DEFINE_boolean('enable_reduction', False, 'whether to only consider a subset of the binary values in each variable, and fix the top part to 0, bottom part to 1') 
tf.app.flags.DEFINE_boolean('only_compute_mask_variable', False, 'in evaluation function, only compute(and return) the mask variable, without doing the actually evaluation') 
tf.app.flags.DEFINE_boolean('load_reduced_index', False, 'whether to load reduced_index from /tmp/reduced_index.pickle in compute_reduced_index()') 
tf.app.flags.DEFINE_boolean('solution_random_rounding', False, 'convert a solution of continuous variable in [0,1] to a binary solution using random rounding') 
tf.app.flags.DEFINE_boolean('cross_entropy_rounding', False, 'convert a solution of continuous variable in [0,1] to a binary solution by minimizing cross entropy') 
tf.app.flags.DEFINE_boolean('add_and_svd_rounding', False, 'convert a solution of continuous to integer using add_and_svd') 
tf.app.flags.DEFINE_boolean('load_add_and_svd_K', False, 'whether to load from /tmp/add_and_svd_K.pickle to decide the add_and_svd_K for each layer') 
tf.app.flags.DEFINE_boolean('eval_fine_tuned_decomposition', False, 'whether to evaluate the decomposition after fine tuning') 
tf.app.flags.DEFINE_boolean('is_training', False, 'indicate whether is training or evaluating') 
tf.app.flags.DEFINE_string('train_feed_dict_path', None, 'the path to the feed_dict for the values of the mask variables, the pickle file is dumped from a previous evaluation')

tf.app.flags.DEFINE_boolean('cost_saturation', False, 'if the cost of a layer in the solution is higher than the original cost, then use the un-decomposed layer') 

tf.app.flags.DEFINE_string('decomposition_scheme', 'tai', 'the decomposition scheme to use in my_slim_layer')
#tf.app.flags.DEFINE_string('decomposition_scheme', 'microsoft', 'the decomposition scheme to use in my_slim_layer')


def print_mvm_parameters():
    print('-----------------------------------')
    print('-----Summary of MVM Parameters')
    print('-----------------------------------')
    print('call_gurobi: %s'%FLAGS.call_gurobi)
    print('load_solution: %s'%FLAGS.load_solution)
    print('solution_path: %s'% FLAGS.solution_path)
    print('hessian_pickle_path: %s'% FLAGS.hessian_pickle_path)
    print('computation_max: %s'% FLAGS.computation_max)
    print('memory_max: %s'% FLAGS.memory_max)
    print('timelimit: %s'% FLAGS.timelimit)
    print('K_heuristic: %s'% FLAGS.K_heuristic)
    print('enable_reduction: %s'% FLAGS.enable_reduction)
    print('-----------------------------------')
#end MaskingVariableManager

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tensorflow_train',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tensorflow_eval/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 32,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

######################
# Chong's functions #
######################
def set_eval_dataset_flags(checkpoint_path=None, use_training_dataset=False, batch_size=None):
    '''set dataset dependent flags
    Args: Nothing, reads FLAGS.dataset
    '''    
    
    model_name = FLAGS.model_name
    if model_name.startswith('cifarnet'):
        FLAGS.dataset_name = 'cifar10'
        FLAGS.dataset_dir='/home/chongli/dataset/cifar10'
        FLAGS.dataset_split_name='test'
    else:
        FLAGS.dataset_name = 'imagenet'
        FLAGS.dataset_dir='/home/chongli/dataset/imagenet-data'
        FLAGS.dataset_split_name='validation'
        
    #use training dataset if requested
    if use_training_dataset:
        FLAGS.dataset_split_name = 'train'
    
    if not FLAGS.model_name:
        raise ValueError('You must set model_name FLAG with --model_name')
    
    if model_name.startswith('inception_v2'):
        FLAGS.labels_offset=0
        FLAGS.batch_size=256
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v2.ckpt'
    elif model_name.startswith('inception_v3'):
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v3.ckpt'
    elif model_name.startswith('inception_v4'):
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v4.ckpt'
    elif model_name.startswith('cifarnet'):
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/cifar10/model.ckpt-1'
    elif model_name.startswith('squeezenet'):
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/squeezenet/model.ckpt-1'
        FLAGS.batch_size=128
        FLAGS.labels_offset=1
    elif model_name == 'mobilenet_v1':
        #INFO, need to specify the filename.ckpt. Specifying the path, orspecifying the file_name.ckpt.data will not work
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet/mobilenet_v1_1.0_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name == 'mobilenet_v1_075':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_075/mobilenet_v1_0.75_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name == 'mobilenet_v1_050':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_050/mobilenet_v1_0.50_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name == 'mobilenet_v1_025':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/mobilenet_025/mobilenet_v1_0.25_224.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name == 'nasnet_mobile':
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/nasnet_mobile/model.ckpt'
        FLAGS.labels_offset=0
        FLAGS.batch_size=128
    elif model_name == 'alexnet_v2':
        FLAGS.labels_offset=0
        FLAGS.batch_size=512
    elif model_name == 'resnet_v2_50':
        FLAGS.labels_offset=0
        FLAGS.batch_size=512
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/resnet_v2_50.ckpt'
    elif model_name == 'vgg_16':
        FLAGS.labels_offset=1
        FLAGS.batch_size=512
        FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/vgg_16.ckpt'
    else:
        raise ValueError('Unknown model_name')
#     if model_name.startswith('lenet'):
#         FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v3.ckpt'

    gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(',')
    if gpus == ['']:
        print('eval_functions_multi: no gpu specified, string is %s'%os.environ["CUDA_VISIBLE_DEVICES"])
        gpus = ['CPU']
    else:
        assert all([g.isdigit() for g in gpus]), 'invalud gpu string : %s'%os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpus = len(gpus)
    #change batch_size for multiple GPU case
    if num_gpus > 1:
        FLAGS.batch_size = num_gpus*32
    else:
        FLAGS.batch_size = 128

    #if a checkpoint_path is provided as argument
    if checkpoint_path is not None:
        print('----------------------using checkpoint_path %s'%checkpoint_path)
        FLAGS.checkpoint_path = checkpoint_path
    assert FLAGS.checkpoint_path is not None, 'No FLAGS.checkpoint_path provided'

    if batch_size is not None:
        FLAGS.batch_size = batch_size

def print_eval_parameters():
    #print summary of training parameters
    print('-----------------------------------')
    print('-----Summary of Evaluation Parameters')
    print('-----------------------------------')
    print('model_name: %s'%FLAGS.model_name)
    print('checkpoint_path: %s'%FLAGS.checkpoint_path)
    print('enable_reduction: %s'% FLAGS.enable_reduction)
    print('cost_saturation: %s'%FLAGS.cost_saturation)
    print('is_training: %s'%FLAGS.is_training)
    print('batch_size: %d'% FLAGS.batch_size)
    print('-----------------------------------')
      

def eval(eval_op_feed_dict=None, session_config=None, max_num_batches=None, sample_percentage = None, masking_variable_value=None, 
        compute_delta_cost_per_layer_solution=None, compute_delta_cost_per_layer_solution2=None):

  FLAGS.is_training=False
  if not FLAGS.dataset_dir:  
    raise ValueError('You must supply the dataset directory with --dataset_dir')
   
  assert not ((max_num_batches is not None) and (sample_percentage is not None)), 'argument of eval max_num_batches and sample_percentage cannot be both specified'
  shuffle = False
  if sample_percentage is not None:
      if sample_percentage < 0.99:
          shuffle = True
  if max_num_batches is not None:
      shuffle = True

  #set the number of batches to be evalated, None for all the samples
  if max_num_batches is not None:
      FLAGS.max_num_batches=max_num_batches

  #tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.set_verbosity(tf.logging.WARN)
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    tf_global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle= shuffle,
        num_readers=32,
        common_queue_capacity=10 * FLAGS.batch_size,
        common_queue_min=3*FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(',')
    if gpus == ['']:
        gpus = ['CPU']
    else:
        assert all([g.isdigit() for g in gpus]), 'invalud gpu string : %s'%os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpus = len(gpus)

    # Split the batch of images and labels for towers.
    if num_gpus == 0:
        num_splits = 1
    else:
        num_splits = num_gpus

    assert FLAGS.batch_size % num_splits == 0, 'batch_size %d cannot be divided by num_splits %d'%(FLAGS.batch_size, num_splits)

    images_splits = tf.split(axis=0, num_or_size_splits=num_splits, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=num_splits, value=labels)
   
    def _tower_logit(images, labels,reuse_variables=None):
        ####################
        # Define the model #
        ####################
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            logits, _ = network_fn(images)

        return logits

    # Calculate the gradients for each model tower.
    tower_loss_list = []
    tower_top_1_op_list = []
    tower_top_5_op_list = []

    #MaskingVariableManager
    mvm = DnnUtili.mvm

    #each element is the concatenation of all the masking variables in one tower
    variables_list = []

    for idx, gpu_id in enumerate(gpus):
        #with slim.arg_scope([slim.model_variable,slim.variable], device='/gpu:%s' % gpu_id):
        if gpu_id == 'CPU':
            device_string = '/cpu:0'
        else:
            device_string = '/gpu:%d' % idx
            
        with tf.device(device_string):
            # Force all Variables to reside on the CPU.
            #with slim.arg_scope([slim.model_variable,slim.variable], device='/cpu:0'):
            #with tf.device('/cpu:0'):
                # Calculate the loss for one tower of the ImageNet model. This
                # function constructs the entire ImageNet model but shares the
                # variables across all towers.
            logits = _tower_logit(images_splits[idx], labels_splits[idx], reuse_variables=True if idx>0 else None)

            #predictions = tf.argmax(logits, 1)
            labels = tf.squeeze(labels_splits[idx])
            # Specify the loss function
            loss = tf.losses.sparse_softmax_cross_entropy(
                    logits = logits, labels = labels)
            # Calculate predictions.
            top_1_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(logits, labels, 1)))
            top_5_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(logits, labels, 5)))

            tower_loss_list.append(loss)
            tower_top_1_op_list.append(top_1_op)
            tower_top_5_op_list.append(top_5_op)

        #if the mvm is not empty
        if not mvm.is_empty():
            #test get variables
            #variables = slim.get_model_variables()
            variables = DnnUtili.mvm.get_variables()
            variables_list.append(variables)

            #num_mask_variables = DnnUtili.mvm.get_num_reduced_mask_variables()

            #variables_name = [str(v) for v in variables]
            #print(variables_name)

        #compute the value of the feed_dict for masking variables
        #only run for the first tower, and if the mvm is not empty
        if idx == 0 and not mvm.is_empty():
            if FLAGS.only_compute_mask_variable is False:
                print_mvm_parameters()

            #print the information of the masking variables
            #if FLAGS.only_compute_mask_variable is False:
            if FLAGS.K_heuristic is None:
                #don't print when computing reduced index
                mvm.print_variable_index()

            assert bool(FLAGS.call_gurobi) + bool(FLAGS.solution_path is not None) + bool(FLAGS.K_heuristic is not None) <= 1, 'no more than one of these options can be true, got %s, %s, %s'%(FLAGS.call_gurobi, FLAGS.load_solution, FLAGS.K_heuristic)

            #if only call the compute_delta_cost_per_layer function in mvm, dump /tmp/delta_cost_per_layer.pickle
            if compute_delta_cost_per_layer_solution is not None:
                mvm.compute_delta_cost_per_layer(compute_delta_cost_per_layer_solution, compute_delta_cost_per_layer_solution2)
                return

            if FLAGS.call_gurobi:
                masking_variable_value = mvm.call_gurobi_miqp(hessian_pickle_path=FLAGS.hessian_pickle_path, 
                        computation_max=FLAGS.computation_max, memory_max=FLAGS.memory_max, 
                        monotonic=False, timelimit=FLAGS.timelimit)
            elif FLAGS.solution_path is not None:
                print('---Loading solution from %s'%FLAGS.solution_path)
                if str(FLAGS.solution_path).endswith('.pickle'):
                    with open(FLAGS.solution_path, 'rb') as f:
                        masking_variable_value = pickle.load(f)
                elif str(FLAGS.solution_path).endswith('.mat'):
                        assert int(FLAGS.solution_random_rounding) + int(FLAGS.cross_entropy_rounding) + int(FLAGS.add_and_svd_rounding) <= 1, 'only choose one type of rounding'
                        masking_variable_value = scipy.io.loadmat(FLAGS.solution_path)['x']
                        #full solution is the solution of all the variables, including the reduced variables
                        full_solution = mvm.expand_reduced_mask_variables_np(np.squeeze(masking_variable_value), exact_size=True)
                        mat_content = scipy.io.loadmat(FLAGS.solution_path)
                        mat_content['full_x'] = full_solution
                        scipy.io.savemat(FLAGS.solution_path, mat_content, do_compression=True)
                        print('eval_functions_multi: adding full_solution to sqp_solution.mat')

                        if FLAGS.solution_random_rounding:
                            masking_variable_value = DnnUtili.solution_random_rounding(masking_variable_value)
                        elif FLAGS.cross_entropy_rounding:
                            masking_variable_value = mvm.cross_entropy_rounding(masking_variable_value, FLAGS.computation_max, FLAGS.memory_max)
                        elif FLAGS.add_and_svd_rounding:
                            #all the values are calculated in my_slim_layer.py when the network is being constructed
                            masking_variable_value = None
                        else:
                            assert masking_variable_value.shape[1]==1, 'expected a column vector, got %s'%str(masking_variable_value.shape)
                            masking_variable_value = np.reshape(masking_variable_value,(masking_variable_value.shape[0]))
                else:
                    raise ValueError('invalid solution_path: %s'%FLAGS.solution_path)
            elif FLAGS.K_heuristic is not None:
                #use the get_mask_variable_value_using_heuristic() to decide the singular values to use using heuristic
                print('---Using heuristic %d in get_mask_variable_value_using_heuristic()'%(FLAGS.K_heuristic))
                masking_variable_value = mvm.get_mask_variable_value_using_heuristic(FLAGS.K_heuristic, 
                        computation_max=FLAGS.computation_max, memory_max=FLAGS.memory_max, 
                        monotonic=False, timelimit=FLAGS.timelimit)
                 
                #if an sqp_solution exists, and contains a x_full entry, convert the full solution according the reduced index just computed
                if os.path.isfile('/tmp/sqp_solution.mat'):
                    mat_content = scipy.io.loadmat('/tmp/sqp_solution.mat')
                    if 'full_x' in mat_content:
                        reduced_x = mvm.reduce_mask_variables_np(np.squeeze(mat_content['full_x']), exact_size=True)
                        mat_content['reduced_x'] = reduced_x
                        scipy.io.savemat('/tmp/sqp_solution.mat',mat_content, do_compression=True)
                        print('eval_functions_multi: added reduced_x to sqp_solution.mat, based on existing full_x and current reduced_index.')
            elif masking_variable_value is not None:
                print('---Using masking variable solution from argument')
            else:
                print('---!!! No approximation is specified,  all mask are enabled, no approximation to the network')
                masking_variable_value = np.zeros([mvm.get_num_mask_variables()],dtype=np.float32)

            #save the computation and memory cost coefficients to a pickle
            mvm.save_coefficients_to_pickle()

            if FLAGS.only_compute_mask_variable:
                assert FLAGS.call_gurobi or FLAGS.K_heuristic is not None, 'should be computing a mask variable solution'
                mvm.save_variable_index_to_pickle()
                print('---mask_variable solution computed.')
                return

            masking_variable_value_dict = mvm.get_variable_to_value_dict(masking_variable_value)
            if FLAGS.add_and_svd_rounding:
                #all the values are calculated in my_slim_layer.py when the network is being constructed
                masking_variable_value_dict = dict()

            #DEBUG
            #DnnUtili.mvm.print_variable_index()
            #DnnUtili.mvm.print_solution(masking_variable_value)

            #print('-----Total computation cost: %s'%mvm.get_total_computation_cost())
            #print('-----Total memory cost: %s'%mvm.get_total_memory_cost())
            
            if not FLAGS.add_and_svd_rounding:
                computation_percentage, memory_percentage = mvm.calculate_percentage_computation_memory_cost(masking_variable_value)
            else:
                computation_percentage, memory_percentage = -1,-1
            #end MaskingVariableManager

        ##clear the masking variable manager, but not for the last gpu, so we still have a copy of the masking variables
        if idx != num_gpus-1:
            DnnUtili.mvm.__init__()

    with slim.arg_scope([slim.model_variable,slim.variable], device='/cpu:0'):
        loss_op = tf.reduce_sum(tower_loss_list)
        top_1_op = tf.reduce_sum(tower_top_1_op_list)
        top_5_op = tf.reduce_sum(tower_top_5_op_list)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()
    #print('eval_functions_multi: %s'%variables_to_restore)

    #find the absolute path of the checkpoint file and restore weights
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Evaluating %s' % checkpoint_path)

    #compute the number of iterations
    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    elif sample_percentage:
      assert 0<sample_percentage <= 1, 'invalid sample_percentage %s'%sample_percentage
      num_batches = math.ceil(dataset.num_samples*sample_percentage / float(FLAGS.batch_size))
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    assert num_batches > 4, 'only evaluate so few batches? num_batches = %d'%num_batches
    #the number of samples that are actually evaluated
    total_sample_count = num_batches * FLAGS.batch_size

    #duplicate the value of the masking variables to each tower
    if not mvm.is_empty():
        #at this point, masking_variable_value_dict is computed
        duplicated_masking_variable_value_dict = copy.copy(masking_variable_value_dict)
        masking_variable_value_dict_values = list(masking_variable_value_dict.values())
        for i, variables in enumerate(variables_list):
            if i == 0:
                continue
            
            for j, var in enumerate(variables):
                duplicated_masking_variable_value_dict[var] = masking_variable_value_dict_values[j]

        #save a dict mapping from the name of the variable to its values 
        name_value_dict = OrderedDict()
        for var,value in masking_variable_value_dict.items():
            name_value_dict[var.op.name] = value
        with open('/tmp/mask_variable_value_dict.pickle', 'wb') as f:
            pickle.dump(name_value_dict, f, protocol=-1)

        masking_variable_value_dict = duplicated_masking_variable_value_dict
        print_eval_parameters()

    assert eval_op_feed_dict is None, 'because the feed_dict has to be duplicated for each tower for the masking variables, this is not implemented yet'
    if not mvm.is_empty():
        eval_op_feed_dict = masking_variable_value_dict
    ##merge mask variable values with the eval_op_feed_dict argument
    #if eval_op_feed_dict:
    #    eval_op_feed_dict = masking_variable_value_dict
    #else:
    #    raise NotImplementedError('because the feed_dict has to be duplicated for each tower for the masking variables, this is not implemented yet')
    #    eval_op_feed_dict = {**eval_op_feed_dict, **masking_variable_value_dict}

    #start a session
    sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))
    init_op = tf.global_variables_initializer()
    #do not need to run init_op because the weights will be restored using saver?
    #sess.run(init_op)

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)

    loss_sum = np.longdouble(0.0)
    top_1_sum = np.longdouble(0.0)
    top_5_sum = np.longdouble(0.0)

    tf_run_start = time.time()
    #sess.run, the code will halt and produce no result
    with slim.queues.QueueRunners(sess):
        for i in range(num_batches):
            #print('starting iteration %d at %s '%(i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            #iteration_start_time = time.time()
          
            loss_np, top_1_op_np, top_5_op_np = sess.run([loss_op, top_1_op, top_5_op], feed_dict=eval_op_feed_dict)
            loss_sum += np.longdouble(loss_np)
            top_1_sum += np.longdouble(top_1_op_np)
            top_5_sum += np.longdouble(top_5_op_np)

    tf_run_time = time.time() - tf_run_start

    loss = loss_sum/np.longdouble(num_batches)/num_gpus
    top_1 = top_1_sum/np.longdouble(total_sample_count)
    top_5 = top_5_sum/np.longdouble(total_sample_count)

    sess.close()

    #accuracy = slim.evaluation.evaluate_once(
    #    master=FLAGS.master,
    #    checkpoint_path=checkpoint_path,
    #    logdir=FLAGS.eval_dir,
    #    num_evals=num_batches,
    #    #Chong edited
    #    #initial_op=None,
    #    #initial_op_feed_dict=init_op_feed_dict,
    #    eval_op=list(names_to_updates.values()),
    #    eval_op_feed_dict = eval_op_feed_dict,
    #    final_op=final_op,
    #    final_op_feed_dict=None,
    #    session_config = session_config,
    #    variables_to_restore=variables_to_restore)


    #delete the eval directory, so the summary files do not accmulate
    shutil.rmtree(FLAGS.eval_dir, ignore_errors=True)

    results = OrderedDict()

    results['accuracy'] = top_1
    results['accuracy_5'] = top_5
    results['loss'] = loss

    if FLAGS.add_and_svd_rounding:
        computation_percentage, memory_percentage = DnnUtili.calculate_percentage_add_and_svd(FLAGS.computation_max, FLAGS.memory_max)

    try:
        results['computation_cost'] = computation_percentage
        results['memory_cost'] = memory_percentage
    except NameError:
        results['computation_cost'] = -1
        results['memory_cost'] = -1

    results['tf_run_time'] = tf_run_time
    #print('eval_functions_multi: tf.run() time: %.1f'%tf_run_time)

    return results
