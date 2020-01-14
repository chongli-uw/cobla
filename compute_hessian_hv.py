'''compute the gradient and the Hv, hessian multipled several orth vectors'''
import time
import pickle
from random import randint

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import numpy as np
import scipy
import scipy.io
import scipy.linalg
import os
import copy

import argparse
parser = argparse.ArgumentParser(description='GPU string')
parser.add_argument('-g','--gpu', help='gpu string', type=str, default='')
parser.add_argument('-s','--save_iter', help='number of iterations to save', type=int, default=10000)
parser.add_argument('-y','--chunk_size', help='compute the mean every check_size samples, so we can evaluate the effect of sampling from the dataset (instead of averaging the whole dataset)', type=int, default=5000)
parser.add_argument('-m','--model_name', help='name of net', type=str, default=None)
parser.add_argument('-p','--pickle_path', help='path to which the result (gradient and/or hessian) is saved', type=str, default='/tmp/hessian_hv.pickle')

parser.add_argument('-k','--K_heuristic', help='K_heuristic', type=int, default=None)
parser.add_argument('-t','--timelimit', help='timelimit for external solver call', type=int, default=None)

parser.add_argument('-c','--computation_max', help='FLAGS.computation_max', type=float, default=None)
parser.add_argument('-z','--memory_max', help='FLAGS.memory_max', type=float, default=None)

parser.add_argument('-M','--solution_path', help='path of pickle file for the masking_variable_value', type=str, default=None)
parser.add_argument('-S','--sample_percentage', help='if choose not to go through every sample in the dataset, can choose the percentage of samples that are randomly used', 
        type=float, default=1.0)
parser.add_argument('-v','--num_hv_vectors', help='number of v vectors in Hv', type=int, default=None)
parser.add_argument('-w','--compute_gradient', help='compute gradient', type=bool, default=False)
parser.add_argument('-W','--compute_hv', help='compute hessian-vector product', type=bool, default=False)
parser.add_argument('-L','--compute_loss', help='compute the loss(and accuracy)', type=bool, default=False)

parser.add_argument('-Z','--hv_gradient_same_dataset', help='use same dataset to compute gradient and hv', type=bool, default=False)
#parser.add_argument('-Y','--lm_path', help='path of where the largarian multiplier is stored, requried to computed hv', type=str, default=None)
parser.add_argument('-V','--orth_v_path', help='path of the v vectors in Hv', type=str, default=None)
parser.add_argument('-N','--magic_number', help='a magic number to be saved with the results', type=int, default=-1)

parser.add_argument('-U','--cost_saturation', help='if the cost of a layer in the solution is higher than the original cost, then use the un-decomposed layer', type=bool, default=False) 

args = vars(parser.parse_args())
cuda_visible_device_string = args['gpu']

gpus = args['gpu'].strip().split(',')
assert len(gpus) >= 1, 'if no gpu is used, use compute_hessian_hv_single.py instead'
num_gpus = len(gpus)

save_iter = args['save_iter']
chunk_size = args['chunk_size']

K_heuristic = args['K_heuristic']

timelimit = args['timelimit']

computation_max = args['computation_max']
memory_max = args['memory_max']

solution_path = args['solution_path']

sample_percentage = args['sample_percentage']
num_hv_vectors = args['num_hv_vectors']

compute_gradient = args['compute_gradient']
compute_hv = args['compute_hv']
compute_loss = args['compute_loss']

hv_gradient_same_dataset=args['hv_gradient_same_dataset']
#lm_path = args['lm_path']
orth_v_path = args['orth_v_path']
assert compute_gradient or compute_hv or compute_loss, 'compute_hessian_hv: neither gradient or hv, or loss is to be computed'
if hv_gradient_same_dataset:
    assert compute_gradient and compute_hv
#if compute_hv:
#    assert lm_path is not None 
#    assert lm_path.endswith('.mat')

magic_number = args['magic_number']

assert int(K_heuristic is not None) + int(solution_path is not None) <= 1, 'K_heuristic and solution_path  cannot both be true'

pickle_path = args['pickle_path']
#pickle_path += '%s.pickle'%cuda_visible_device_string
print('----save Hv hessian to: %s'%pickle_path)

os.environ["CUDA_VISIBLE_DEVICES"]=cuda_visible_device_string

import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import my_slim_layers

slim = tf.contrib.slim
#import train_functions
#from train_functions import *
import eval_functions_multi
from eval_functions_multi import *

import DnnUtili

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from datasets import dataset_utils
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

import math
import numpy as np
import time
from datetime import datetime

#has to defer model_name definnition here, otherwise preprocessing_name cannot be recognized, don't know why TODO
FLAGS.model_name = args['model_name']
FLAGS.cost_saturation = args['cost_saturation']

#always enable reduction
FLAGS.enable_reduction = True
FLAGS.load_reduced_index = True

gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(',')
assert len(gpus) >= 1, 'if no gpu is used, use eval_function.py instead'
num_gpus = len(gpus)

if FLAGS.model_name.startswith('cifarnet'):
    batch_size= 64*num_gpus
elif FLAGS.model_name.startswith('squeezenet'):
    batch_size= 64*num_gpus
elif FLAGS.model_name.startswith('nasnet_mobile'):
    batch_size= 64*num_gpus
elif FLAGS.model_name.startswith('mobilenet'):
    batch_size= 64*num_gpus
elif FLAGS.model_name.startswith('inception'):
    batch_size= 32*num_gpus
elif FLAGS.model_name.startswith('vgg'):
    batch_size= 32*num_gpus
else:
    raise ValueError('unknown model_name %s'%FLAGS.model_name)

eval_functions_multi.set_eval_dataset_flags(use_training_dataset = True, batch_size = batch_size)

with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf_global_step = tf.train.get_or_create_global_step()
    
    ###########################################
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
        #shuffle is true if sample_percentage is smaller than 1
        shuffle= (sample_percentage < 0.999),
        #num_epochs = 1,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=5*FLAGS.batch_size)
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
        capacity=15 * FLAGS.batch_size)

    # Split the batch of images and labels for towers.
    if num_gpus == 0:
        num_splits = 1
    else:
        num_splits = num_gpus

    assert FLAGS.batch_size % num_splits == 0, 'batch_size %d cannot be divided by num_splits %d'%(FLAGS.batch_size, num_splits)

    images_splits = tf.split(axis=0, num_or_size_splits=num_splits, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=num_splits, value=labels)
   
    def _tower_loss(images, labels,reuse_variables=None):
        ####################
        # Define the model #
        ####################
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            logits, _ = network_fn(images)

        #if not compute_hv 
        if compute_hv:
            #convert label to one hot encoding, for softmax_cross_entropy, this works with second order derivitives, 
            onehot_labels = slim.one_hot_encoding(
                  labels, dataset.num_classes - FLAGS.labels_offset)
            # Specify the loss function
            loss = tf.losses.softmax_cross_entropy(
                    logits = logits, onehot_labels = onehot_labels)
        #not computing_hv, then do not need the second deritivite of any operation, use sparse softmax for faster procesing
        else:
            labels = tf.squeeze(labels)
            # Specify the loss function
            loss = tf.losses.sparse_softmax_cross_entropy(
                    logits = logits, labels = labels)

        # Calculate predictions.
        top_1_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(logits, labels, 1)))
        top_5_op = tf.reduce_sum(tf.to_float(tf.nn.in_top_k(logits, labels, 5)))

        return loss, top_1_op, top_5_op

    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_hvs = []
    tower_loss = []
    tower_top_1 = []
    tower_top_5 = []
    #each element is the concatenation of all the masking variables in one tower
    variables_list = []

    for idx, gpu_id in enumerate(gpus):
        assert gpu_id.isdigit(), 'gpu id %s is not an integer'%gpu_id
        #with slim.arg_scope([slim.model_variable,slim.variable], device='/gpu:%s' % gpu_id):
        with tf.device('/gpu:%d' % idx):

            #DEBUG, placing loss on cpu or GPU does not seem to make a difference
            # Calculate the loss for one tower of the ImageNet model. This
            # function constructs the entire ImageNet model but shares the
            # variables across all towers.
            loss, top_1_op, top_5_op = _tower_loss(images_splits[idx], labels_splits[idx], reuse_variables=True if idx>0 else None)

            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.model_variable,slim.variable], device='/cpu:0'):
            #with tf.device('/cpu:0'):

                #test get variables
                #variables = slim.get_model_variables()
                variables = DnnUtili.mvm.get_variables()
                variables_list.append(variables)

                num_mask_variables = DnnUtili.mvm.get_num_reduced_mask_variables()

                variables_name = [str(v) for v in variables]
                #DEBUG
                #print(variables_name)

                #slim.get_unique_variable does not work for unknown reason
                #v = slim.get_unique_variable(variables_name[0])

                #only run for the first tower
                if idx == 0:
                    #prepare v vectors
                    if compute_hv:
                        if orth_v_path is None:
                            assert num_hv_vectors is not None, 'has to specifiy either orth_v_path or num_hv_vectors'
                            assert num_hv_vectors >= 1
                            #generate orthnormal vectors, essentially computing the left singular vectors of a random matrix
                            orth_v = scipy.linalg.orth(np.random.normal(0,5,[num_mask_variables, num_mask_variables]))
                            orth_v = orth_v.astype(np.float32)

                            assert num_hv_vectors <= 0.5*num_mask_variables, 'does not make sense to have number of v larger than the dimension of the hessian'
                            assert np.dot(orth_v[:,0], orth_v[:,1]) < 1e-5
                            assert orth_v.shape[0] == num_mask_variables
                            assert orth_v.shape[1] >= num_hv_vectors
                            orth_v = orth_v[:,0:num_hv_vectors]
                            #rescale orth_v to make it in a reasonable range
                            orth_v = orth_v*(num_hv_vectors**2)
                        else:
                            orth_v = scipy.io.loadmat(orth_v_path)['orth_v']
                            orth_v = orth_v.astype(np.float32)
                            assert orth_v.shape[0] == num_mask_variables

                        v_op_list = list()
                        for i in range(orth_v.shape[1]):
                            v_op_list.append(tf.Variable(orth_v[:,i], dtype=tf.float32, name='v_%d'%i))
                   
                    #generate the feed_dict of the value of masking variables
                    if K_heuristic is not None:
                        print('----in compute_hessian_hv, mask variable values are set by K_heuristic %d'%K_heuristic)
                        masking_variable_value = DnnUtili.mvm.get_mask_variable_value_using_heuristic(K_heuristic,
                            computation_max=computation_max, memory_max=memory_max, timelimit=timelimit)
                    #if choose to load mask_variable_value from a pickle file    
                    elif solution_path is not None:
                        print('----in compute_hessian_hv, mask variable values are set by loading %s'%solution_path)
                        if str(solution_path).endswith('.pickle'):
                            with open(solution_path,'rb') as f:
                                masking_variable_value = pickle.load(f)
                        elif str(solution_path).endswith('.mat'):
                                masking_variable_value = scipy.io.loadmat('/tmp/current_x.mat')['x']
                                assert masking_variable_value.shape[1]==1, 'expected a column vector, got %s'%str(masking_variable_value.shape)
                                masking_variable_value = np.reshape(masking_variable_value,(masking_variable_value.shape[0]))
                        else:    
                            raise ValueError('inavlid solution_path: %s'%solution_path)
                    else:
                        print('----in compute_hessian_hv, mask variable values are set to all on')
                        masking_variable_value = np.zeros([DnnUtili.mvm.get_num_mask_variables()],dtype=np.float32)

                    #DEBUG
                    DnnUtili.mvm.print_variable_index()
                    #DnnUtili.mvm.print_solution(masking_variable_value)

                    masking_variable_value_dict = DnnUtili.mvm.get_variable_to_value_dict(masking_variable_value)
                    #save the computation and memory cost coefficients to a pickle
                    #DnnUtili.mvm.save_coefficients_to_pickle()

                    computation_percentage, memory_percentage = DnnUtili.mvm.calculate_percentage_computation_memory_cost(masking_variable_value)

                    #start a session
                    sess = tf.Session(config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=False))
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)    

            #compute gradient and hv, on gpu
            if compute_hv:
                #compute the hessian vector product
                with DnnUtili.Timer('Build hessian_vector op'):    
                    hessian_vector_op_list = list()
                    for v in v_op_list:
                        hessian_vector_op_list.append(DnnUtili.my_hessian_vector_product(loss, variables, v))
                    tower_hvs.append(hessian_vector_op_list)

            if compute_gradient:
                with DnnUtili.Timer('Build gradient op'):    
                    gradient_op = DnnUtili.myGradients(loss, variables)
                    tower_grads.append(gradient_op)
                #print('size of hessian_vector_op: %s'%str(hessian_vector_op))

            if compute_loss:
                tower_loss.append(loss)
                tower_top_1.append(top_1_op)
                tower_top_5.append(top_5_op)

            ##clear the masking variable manager, but not for the last gpu, so we still have a copy of the masking variables
            if idx != num_gpus-1:
                DnnUtili.mvm.__init__()
    
    #average the tower
    gradient_op = tf.reduce_mean(tower_grads, axis=0)
    hessian_vector_op_list = tf.reduce_mean(tower_hvs, axis=0)
    loss_op = tf.reduce_sum(tower_loss,axis=0)
    top_1_op = tf.reduce_sum(tower_top_1,axis=0)
    top_5_op = tf.reduce_sum(tower_top_5,axis=0)
    eval_ops = [loss_op, top_1_op, top_5_op]

    #duplicate the value of the masking variables to each tower
    #at this point, masking_variable_value_dict is computed
    duplicated_masking_variable_value_dict = copy.copy(masking_variable_value_dict)
    masking_variable_value_dict_values = list(masking_variable_value_dict.values())
    for i, variables in enumerate(variables_list):
        if i == 0:
            continue
        
        for j, var in enumerate(variables):
            duplicated_masking_variable_value_dict[var] = masking_variable_value_dict_values[j]
    masking_variable_value_dict = dict(duplicated_masking_variable_value_dict)

    #number of iterations
    num_batches = math.floor(float(dataset.num_samples*sample_percentage)/float(FLAGS.batch_size))
    assert num_batches >= 1, 'at least run one iteration'
    total_sample_count = num_batches * FLAGS.batch_size

    #every n_save iterations, the mean of n_save hessians are computed and stored
    #offline compute of hessian_mean initial hessian_op
    #https://math.stackexchange.com/questions/106700/incremental-averageing
    if compute_hv:
        hessian_mean_list = list()
        hessian_mean = np.longdouble(0.0)
    else:
        hessian_mean_list = np.float64(None)
        hessian_mean = np.float64(None)

    if compute_gradient:
        gradient_mean_list = list()
        gradient_mean = np.longdouble(0.0)
    else:
        gradient_mean_list = np.float64(None)
        gradient_mean = np.float64(None)

    if compute_loss:
        loss_sum = np.longdouble(0.0)
        top_1_sum = np.longdouble(0.0)
        top_5_sum = np.longdouble(0.0)
    else:
        loss_sum = np.longdouble(None)
        top_1_sum = np.longdouble(None)
        top_5_sum = np.longdouble(None)


    #track the time actually spent in sess.run
    tf_run_time = 0.0

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

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)

    #with tf.Session() as sess:    
    #the QueueRunners is key for the code to work, otherwise, if just call
    #sess.run, the code will halt and produce no result
    with slim.queues.QueueRunners(sess):
        for i in range(num_batches):
            tf_run_start = time.time()
            #print('starting iteration %d at %s '%(i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            #iteration_start_time = time.time()
          
            #use the same batch to compute gradient and hv, useful for testing
            if hv_gradient_same_dataset:
                assert not compute_loss, 'compute loss with gradient and hv not implemented'
                run_np = sess.run([gradient_op] + hessian_vector_op_list, feed_dict=masking_variable_value_dict)
                gradient_iter = run_np[0]
                gradient_mean = gradient_mean + (np.longdouble(gradient_iter) - gradient_mean)/np.longdouble(i+1)
                hessian_iter = np.transpose(np.array(run_np[1:]))
                hessian_mean = hessian_mean + (np.longdouble(hessian_iter) - hessian_mean)/np.longdouble(i+1)
            else:
                if compute_hv:
                    if not compute_gradient:
                        assert not compute_loss, 'compute loss without computing gradient not implemented'
                    hessian_iter = sess.run(hessian_vector_op_list, feed_dict=masking_variable_value_dict)
                    hessian_iter = np.transpose(np.array(hessian_iter))
                    hessian_mean = hessian_mean + (np.longdouble(hessian_iter) - hessian_mean)/np.longdouble(i+1)
                if compute_gradient:
                    if compute_loss:
                        gradient_iter, loss_np, top_1_np, top_5_np = sess.run([gradient_op]+eval_ops, feed_dict=masking_variable_value_dict)
                        loss_sum += loss_np
                        top_1_sum += top_1_np
                        top_5_sum += top_5_np
                    else:
                        gradient_iter = sess.run(gradient_op, feed_dict=masking_variable_value_dict)
                    gradient_mean = gradient_mean + (np.longdouble(gradient_iter) - gradient_mean)/np.longdouble(i+1)

            tf_run_time += time.time() - tf_run_start

            #compute the average across chunk_size samples
            j = i % chunk_size 
            if j == 0:
                if compute_hv:
                    hessian_chunk_mean = np.longdouble(0.0)
                if compute_gradient:
                    gradient_chunk_mean = np.longdouble(0.0)
                        
            if compute_hv:
                hessian_chunk_mean = hessian_chunk_mean + (np.longdouble(hessian_iter) - hessian_chunk_mean)/np.longdouble(j+1)
            if compute_gradient:
                gradient_chunk_mean = gradient_chunk_mean + (np.longdouble(gradient_iter) - gradient_chunk_mean)/np.longdouble(j+1)

            if j == chunk_size - 1:
                if compute_hv:
                    hessian_mean_list.append(np.float64(hessian_chunk_mean))
                if compute_gradient:
                    gradient_mean_list.append(np.float64(gradient_chunk_mean))

            #print('Iteration took %.6f'%( time.time()-iteration_start_time))
            #if (i > 1 and i % 1000 == 0) or (i == num_batches -1) or i < 5:
                #print('Iteration %d at %s '%(i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            #save mean every save_iter iterations
            #if (i > 1 and i % save_iter == 0) or (i == num_batches-1):
            if (i == num_batches-1):
                print('saving iteration %d at %s '%(i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                with open(pickle_path, 'wb') as f:  
                    if compute_hv:
                        content = (np.float64(hessian_mean), np.float64(gradient_mean), np.float64(hessian_mean_list), np.float64(gradient_mean_list), orth_v, magic_number, tf_run_time)
                    else:
                        content = (np.float64(hessian_mean), np.float64(gradient_mean), np.float64(hessian_mean_list), np.float64(gradient_mean_list), magic_number, tf_run_time)
                    pickle.dump(content, f, protocol=-1)

                #save a mat version as well
                mat_path = pickle_path.replace('.pickle','.mat')
                mat_dict = {'Hv':hessian_mean, 'gradient':gradient_mean, 'Hv_mean_list':np.float64(hessian_mean_list), 'gradient_mean_list':np.float64(gradient_mean_list), 'magic_number':magic_number, 'tf_run_time':tf_run_time}
                if compute_hv:
                    mat_dict['v'] = np.float64(orth_v)
                scipy.io.savemat(mat_path, mat_dict , do_compression=True)

sess.close()
#DEBUG
print('compute_hessian_hv: tf_run_time is %.2f'%tf_run_time)

if compute_loss:
    loss = loss_sum/np.longdouble(num_batches)/num_gpus
    top_1 = top_1_sum/np.longdouble(total_sample_count)
    top_5 = top_5_sum/np.longdouble(total_sample_count)

    results = OrderedDict()

    results['accuracy'] = top_1
    results['accuracy_5'] = top_5
    results['loss'] = loss
    results['computation_cost'] = computation_percentage
    results['memory_cost'] = memory_percentage
    results['tf_run_time'] = tf_run_time

    accuracy = results
    print(accuracy)

    with open('/tmp/mask_wrapper_results.pickle','wb') as f:
        pickle.dump((accuracy, magic_number), f, protocol=-1)

    #format content of the mat file, accuracy is a ordered dict
    mat_content = dict(accuracy)
    if magic_number is not None:
        mat_content['magic_number'] = magic_number
    scipy.io.savemat('/tmp/mask_wrapper_results.mat',{'mask_wrapper_results':mat_content}, do_compression=True)

print(0)            
