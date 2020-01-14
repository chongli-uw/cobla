import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import os
import TFInclude
import DnnUtili

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

import pickle
import numpy as np
import scipy
import scipy.io
import eval_functions_multi
from eval_functions_multi import *

import time
import numpy as np
from collections import OrderedDict

######
#TODO argparse cannot handle boolean type, for example, passing script.py --enable_reduction 0, will make enable_reduce to be 1, has to manually convert to boolean type
#for now, keep all booelan type to be default false, and only specify it when want it to be true
import argparse
parser = argparse.ArgumentParser(description='GPU string')
parser.add_argument('-g','--gpu', help='gpu string', type=str, default='')
#parser.add_argument('-n','--num_gpu', help='total number of GPU to use', type=int, default=0)
parser.add_argument('-K','--K_heuristic', help='which FLAGS.K_heuristic to use', type=int, default=None)
parser.add_argument('-c','--computation_max', help='FLAGS.computation_max', type=float, default=None)
parser.add_argument('-m','--memory_max', help='FLAGS.memory_max', type=float, default=None)
parser.add_argument('-G','--call_gurobi', help='FLAGS.call_gurobi', type=bool, default=False)
parser.add_argument('-l','--load_solution', help='FLAGS.load_solution', type=bool, default=False)
parser.add_argument('-P','--solution_path', help='FLAGS.call_gurobi', type=str, default=None)
parser.add_argument('-V','--checkpoint_path', help='FLAGS.checkpoint_path', type=str, default=None)

parser.add_argument('-M','--model_name', help='model_name', type=str, required=True)
parser.add_argument('-H','--hessian_pickle_path', help='', type=str, default=None)
parser.add_argument('-z','--use_training_dataset', help='', type=bool, default=False)
parser.add_argument('-y','--sample_percentage', help='if choose not to go through every sample in the dataset, can choose the percentage of samples that are randomly used', 
        type=float, default=1.0)

parser.add_argument('-N','--magic_number', help='a magic number to be saved with the results', type=int, default=None)
parser.add_argument('-t','--timelimit', help='timelimit for gurobi call', type=int, default=None)

parser.add_argument('-e','--enable_reduction', help='FLAGS.enable_reduction', type=bool, default=False)
parser.add_argument('-S','--only_compute_mask_variable', help='in evaluation function, only compute(and return) the mask variable, without doing the actually evaluation', type=bool, default=False)
parser.add_argument('-L','--load_reduced_index', help='whether to load reduced_index from /tmp/reduced_index.pickle in compute_reduced_index()', type=bool, default=False) 
parser.add_argument('-q','--solution_random_rounding', help='convert a solution of continuous variable in [0,1] to a binary solution using random rounding', type=bool, default=False) 
parser.add_argument('-Q','--cross_entropy_rounding', help='convert a solution of continuous variable in [0,1] to a binary solution by minimizing cross entropy', type=bool, default=False) 
parser.add_argument('-x','--add_and_svd_rounding', help='convert a solution of continuous to integer using add_and_svd', type=bool, default=False) 
parser.add_argument('-r','--load_add_and_svd_K', help='whether to load from /tmp/add_and_svd_K.pickle to decide the add_and_svd_K for each layer', type=bool, default=False) 
parser.add_argument('-X','--eval_fine_tuned_decomposition', help='evaluating a fine-tuned decomposition', type=bool, default=False) 
parser.add_argument('-Y','--train_feed_dict_path', help='path to pickle file containing the value of mask variables, for fine tuning', type=str, default=None) 

parser.add_argument('-u','--compute_delta_cost_per_layer_solution', help='path of solution for compute_delta_cost_per_layer in MVM', type=str, default=None) 
parser.add_argument('-v','--compute_delta_cost_per_layer_solution2', help='path of solution2 for compute_delta_cost_per_layer', type=str, default=None) 

parser.add_argument('-U','--cost_saturation', help='if the cost of a layer in the solution is higher than the original cost, then use the un-decomposed layer', type=bool, default=False) 


args = vars(parser.parse_args())

cuda_visible_device_string = args['gpu']
#num_gpu = args['num_gpu']
FLAGS.K_heuristic = args['K_heuristic']
FLAGS.computation_max = args['computation_max']
FLAGS.memory_max = args['memory_max']

FLAGS.call_gurobi = args['call_gurobi']
FLAGS.timelimit = args['timelimit']
FLAGS.enable_reduction = args['enable_reduction']
FLAGS.only_compute_mask_variable = args['only_compute_mask_variable']
FLAGS.load_reduced_index = args['load_reduced_index']
FLAGS.solution_random_rounding = args['solution_random_rounding']
FLAGS.cross_entropy_rounding = args['cross_entropy_rounding']
FLAGS.add_and_svd_rounding = args['add_and_svd_rounding']
FLAGS.load_add_and_svd_K = args['load_add_and_svd_K']
FLAGS.eval_fine_tuned_decomposition = args['eval_fine_tuned_decomposition']
FLAGS.train_feed_dict_path = args['train_feed_dict_path']
FLAGS.cost_saturation = args['cost_saturation']

assert int(FLAGS.solution_random_rounding) + int(FLAGS.cross_entropy_rounding) + int(FLAGS.add_and_svd_rounding) <= 1, 'only choose one type of rounding'

FLAGS.load_solution=args['load_solution']
FLAGS.solution_path=args['solution_path']

FLAGS.model_name = args['model_name']
FLAGS.hessian_pickle_path = args['hessian_pickle_path']

magic_number = args['magic_number']

os.environ["CUDA_VISIBLE_DEVICES"]=cuda_visible_device_string
#print('---in mask_wrapper CUDA_VISIBLE_DEVICES is %s' % os.environ["CUDA_VISIBLE_DEVICES"])
#print('Total GPU to use is %d' % num_gpu)
######
mvm = DnnUtili.mvm
if args['add_and_svd_rounding']:
    print('mask_wrapper: setting add_and_svd')

eval_functions_multi.set_eval_dataset_flags(checkpoint_path=args['checkpoint_path'],use_training_dataset=args['use_training_dataset'])

accuracy=eval_functions_multi.eval(sample_percentage=args['sample_percentage'],
        compute_delta_cost_per_layer_solution=args['compute_delta_cost_per_layer_solution'], 
        compute_delta_cost_per_layer_solution2=args['compute_delta_cost_per_layer_solution2'])

#DEBUG
#print('conv2d computation:%.5E'%mvm.get_total_computation_cost())
#print('conv2d memory:%.5E'%mvm.get_total_memory_cost())

if not (FLAGS.only_compute_mask_variable or args['compute_delta_cost_per_layer_solution']):
    print(accuracy)

    with open('/tmp/mask_wrapper_results.pickle','wb') as f:
        pickle.dump((accuracy, magic_number), f, protocol=-1)

    #format content of the mat file, accuracy is a ordered dict
    mat_content = dict(accuracy)
    if magic_number is not None:
        mat_content['magic_number'] = magic_number
    scipy.io.savemat('/tmp/mask_wrapper_results.mat',{'mask_wrapper_results':mat_content}, do_compression=True)
