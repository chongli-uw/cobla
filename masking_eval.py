#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:14:31 2017

@author: chongli
"""

#%%
import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import TFInclude
import DnnUtili

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

#use slim in tensorflow contrib
slim = tf.contrib.slim

import pickle
import numpy as np
import eval_functions
from eval_functions import *

import time
import numpy as np
t = time.time()
#FLAGS.model_name='vgg_16_rank'
#FLAGS.model_name='vgg_19'
#FLAGS.model_name='inception_v3'
FLAGS.model_name='cifarnet'
#FLAGS.model_name='cifarnet_rank'
#FLAGS.model_name='alexnet_v2_rank'
#FLAGS.model_name='alexnet_v2'

mvm = DnnUtili.mvm

#print('is_training %s'%mvm.is_training)
#mvm.is_training = True

eval_functions.set_eval_dataset_flags()


#DEBUG
#this pickle file only contains 751 variables
#path='/tmp/solution_0.3_memory.pickle'
#with open(path, 'rb') as f:
#        data = pickle.load(f)
#
#masking_variable_value=np.zeros([1913], np.float32)
#masking_variable_value[0:751] = data

#FLAGS.call_gurobi=True

#FLAGS.load_gurobi_solution=True
#FLAGS.gurobi_solution_path = '/tmp/solution.pickle'
#FLAGS.gurobi_solution_path = '/tmp/solution_CifarNet_mem_0.10.pickle'
FLAGS.computation_max = 0.4
FLAGS.memory_max = 0.4

#FLAGS.hessian_pickle_path='/tmp/hessian_cifar_751.pickle.correct'
#FLAGS.hessian_pickle_path='/tmp/hessian_cifar_751_gn.pickle'
#FLAGS.hessian_pickle_path='/tmp/hessian_gn_untested.pickle'
#FLAGS.hessian_pickle_path='/tmp/hessian_gn_no_gradient.pickle'
#FLAGS.hessian_pickle_path='/tmp/hessian.pickle'
FLAGS.hessian_pickle_path='/tmp/hessian_gn_inner.pickle'

#FLAGS.timelimit = 100

FLAGS.enable_reduction = False

FLAGS.K_heuristic = 3
#FLAGS.K_heuristic_percentage = 0.33


accuracy=eval_functions.eval()



print(accuracy)
