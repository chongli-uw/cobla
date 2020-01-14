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

tf.app.flags.DEFINE_boolean('cost_saturation', False, 'if the cost of a layer in the solution is higher than the original cost, then use the un-decomposed layer') 

#tf.app.flags.DEFINE_string('decomposition_scheme', 'tai', 'the decomposition scheme to use in my_slim_layer')
tf.app.flags.DEFINE_string('decomposition_scheme', 'microsoft', 'the decomposition scheme to use in my_slim_layer')


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
    'num_preprocessing_threads', 16,
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
def set_eval_dataset_flags(checkpoint_path=None, use_training_dataset = False, batch_size = None):
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
        FLAGS.batch_size=128
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
    else:
        raise ValueError('Unknown model_name')
#     if model_name.startswith('lenet'):
#         FLAGS.checkpoint_path='/home/chongli/slim_checkpoints/inception_v3.ckpt'

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
    print('batch_size: %d'% FLAGS.batch_size)
    print('-----------------------------------')
      

def eval(eval_op_feed_dict=None, session_config=None, max_num_batches=None, sample_percentage = None, masking_variable_value=None):
  FLAGS.is_training=False
  if not FLAGS.dataset_dir:  
    raise ValueError('You must supply the dataset directory with --dataset_dir')
   
  assert not ((max_num_batches is not None) and (sample_percentage is not None)), 'argument of eval max_num_batches and sample_percentage cannot be both specified'
  #set the number of batches to be evalated, None for all the samples
  if max_num_batches is not None:
      FLAGS.max_num_batches=max_num_batches

  #tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.set_verbosity(tf.logging.WARN)
  with tf.Graph().as_default():

    tf_global_step = slim.get_or_create_global_step()

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
        shuffle=False,
        common_queue_capacity=8 * FLAGS.batch_size,
        common_queue_min=2*FLAGS.batch_size)
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

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    #print(variables_to_restore)

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Specify the loss function
    loss = tf.losses.sparse_softmax_cross_entropy(
            logits = logits, labels = labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_sparse_recall_at_k(logits, tf.reshape(labels, [labels.get_shape().as_list()[0],1]), 5),
        'Loss': tf.contrib.metrics.streaming_mean(loss)
    })

    #final_ops to pass to slim.evaluate
    final_op = list()
    final_op.append(names_to_values['Accuracy'])
    final_op.append(names_to_values['Recall_5'])
    final_op.append(names_to_values['Loss'])
    
    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    elif sample_percentage:
      assert 0<sample_percentage <= 1, 'invalid sample_percentage %s'%sample_percentage
      num_batches = math.ceil(dataset.num_samples*sample_percentage / float(FLAGS.batch_size))
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    assert num_batches > 4, 'only evaluate so few batches ? num_batches = %d'%num_batches

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    #MaskingVariableManager
    mvm = DnnUtili.mvm

    #if the mvm is not empty
    if not mvm.is_empty():
        print_mvm_parameters()

        assert not mvm.is_training, 'cannot be fine-tuning, we are in eval_function!'

        #print the information of the masking variables
        mvm.print_variable_index()

        assert bool(FLAGS.call_gurobi) + bool(FLAGS.load_solution) + bool(FLAGS.K_heuristic is not None) <= 1, 'no more than one of these options can be true, got %s, %s, %s'%(FLAGS.call_gurobi, FLAGS.load_solution, FLAGS.K_heuristic)

        if FLAGS.call_gurobi:
            masking_variable_value = mvm.call_gurobi_miqp(hessian_pickle_path=FLAGS.hessian_pickle_path, 
                    computation_max=FLAGS.computation_max, memory_max=FLAGS.memory_max, 
                    monotonic=False, timelimit=FLAGS.timelimit)
        elif FLAGS.load_solution:
            print('---Loading gurobi solution from %s'%FLAGS.solution_path)
            if str(FLAGS.solution_path).endswith('.pickle'):
                with open(FLAGS.solution_path, 'rb') as f:
                    masking_variable_value = pickle.load(f)
            elif str(FLAGS.solution_path).endswith('.mat'):
                    assert int(FLAGS.solution_random_rounding) + int(FLAGS.cross_entropy_rounding) <= 1, 'only choose one type of rounding'
                    masking_variable_value = scipy.io.loadmat(FLAGS.solution_path)['x']
                    if FLAGS.solution_random_rounding:
                        masking_variable_value = DnnUtili.solution_random_rounding(masking_variable_value)
                    elif FLAGS.cross_entropy_rounding:
                        masking_variable_value = mvm.cross_entropy_rounding(masking_variable_value, FLAGS.computation_max, FLAGS.memory_max)
                    else:
                        assert masking_variable_value.shape[1]==1, 'expected a column vector, got %s'%str(masking_variable_value.shape)
                        masking_variable_value = np.reshape(masking_variable_value,(masking_variable_value.shape[0]))
            else:
                raise ValueError('inavlid solution_path: %s'%solution_path)
        elif FLAGS.K_heuristic is not None:
            #use the get_mask_variable_value_using_heuristic() to decide the singular values to use using heuristic
            print('---Using heuristic %d, percent %s in get_mask_variable_value_using_heuristic()'%(FLAGS.K_heuristic, FLAGS.K_heuristic_percentage))
            masking_variable_value = mvm.get_mask_variable_value_using_heuristic(FLAGS.K_heuristic, FLAGS.K_heuristic_percentage, 
                    computation_max=FLAGS.computation_max, memory_max=FLAGS.memory_max, 
                    monotonic=False, timelimit=FLAGS.timelimit)
        elif masking_variable_value is not None:
            print('---Using masking variable solution from argument')
        else:
            print('---!!! No approximation is specified,  all mask are enabled, no approximation to the network')
            masking_variable_value = np.zeros([mvm.get_num_mask_variables()],dtype=np.float32)

        #save the computation and memory cost coefficients to a pickle
        mvm.save_coefficients_to_pickle()

        if FLAGS.only_compute_mask_variable:
            assert FLAGS.call_gurobi or FLAGS.K_heuristic, 'should be computing a mask variable solution'
            mvm.save_variable_index_to_pickle()
            print('---mask_variable solution computed.')
            return

        #merge mask variable values with the eval_op_feed_dict argument
        masking_variable_value_dict = mvm.get_variable_to_value_dict(masking_variable_value)
        if eval_op_feed_dict is None:
            eval_op_feed_dict = masking_variable_value_dict
        else:
            eval_op_feed_dict = {**eval_op_feed_dict, **masking_variable_value_dict}

        #print('-----Total computation cost: %s'%mvm.get_total_computation_cost())
        #print('-----Total memory cost: %s'%mvm.get_total_memory_cost())

        computation_percentage, memory_percentage = mvm.calculate_percentage_computation_memory_cost(masking_variable_value)
        #end MaskingVariableManager

    #####
    print_eval_parameters()

    accuracy = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        #Chong edited
        #initial_op=None,
        #initial_op_feed_dict=init_op_feed_dict,
        eval_op=list(names_to_updates.values()),
        eval_op_feed_dict = eval_op_feed_dict,
        final_op=final_op,
        final_op_feed_dict=None,
        session_config = session_config,
        variables_to_restore=variables_to_restore)

    #delete the eval directory, so the summary files do not accmulate
    shutil.rmtree(FLAGS.eval_dir, ignore_errors=True)

    results = OrderedDict()

    results['accuracy'] = accuracy[0]
    results['accuracy_5'] = accuracy[1]
    results['loss'] = accuracy[2]
    if mvm.is_empty():
        results['computation_cost'] = None 
        results['memory_cost'] = None 
    else:
        results['computation_cost'] = computation_percentage
        results['memory_cost'] = memory_percentage

    return results
