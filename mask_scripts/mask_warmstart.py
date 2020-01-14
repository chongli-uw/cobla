import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import pickle

import time
import numpy as np
from collections import OrderedDict

import subprocess
from random import randint

results = OrderedDict()


#path to evaluation function wrapper
wrapper_path = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py '
model_name = 'cifarnet'

#gradually lower the computation/memory max
computation_max = 0.15
memory_max = 0.15

timelimit = 1000

#first find an initial solution using heuristic, the bound is increase by 50%
cmd = wrapper_path + '--model_name %s '%(model_name)
cmd += '--K_heuristic 2 '
cmd += '--computation_max %.3f '%(computation_max*2)
cmd += '--memory_max %.3f '%(memory_max*2)
cmd += '--timelimit %d '%timelimit

#attach a magic number to veirfy the validaty of the subprocess call
magic_number = randint(0,999)
cmd += '--magic_number %d'%magic_number
print(cmd)

#external call
subprocess.call(cmd, shell=True)

#load the result
with open('/tmp/mask_wrapper_results.pickle','rb') as f:
    accuracy, magic_number_read = pickle.load(f)
assert magic_number_read == magic_number, 'magic_number verification failed'
print('initial value using heuristic')
print(accuracy)

for i in range(10):
    print('----------------------------------------------------------------Iteration %d-------------------'%i)

    print('recompute the hessian, given the current solution')
    hessian_pickle_path = '/tmp/hessian_iter.pickle'
    hessian_cmd = 'python3.5 compute_hessian_gn.py --model_name %s --gpu 0 --mask_variable_pickle_path /tmp/solution.pickle --pickle_path %s --sample_percentage 1.0'%(model_name,hessian_pickle_path)

    print(hessian_cmd)
    subprocess.call(hessian_cmd, shell=True)

    #call gurobi using bound that is gradually close to the desired bound
    computation_max = (2-(i+1)*0.1)*0.15
    memory_max = (2-(i+1)*0.1)*0.15

    cmd = wrapper_path + '--model_name %s --hessian_pickle_path %s --gpu 2 '%(model_name, hessian_pickle_path)
    cmd += '--call_gurobi %d '%1
    cmd += '--computation_max %.3f '%computation_max
    cmd += '--memory_max %.3f '%memory_max
    cmd += '--timelimit %d '%timelimit

    #attach a magic number to veirfy the validaty of the subprocess call
    magic_number = randint(0,999)
    cmd += '--magic_number %d'%magic_number
    print(cmd)

    #external call
    subprocess.call(cmd, shell=True)

    #load the result
    with open('/tmp/mask_wrapper_results.pickle','rb') as f:
        accuracy, magic_number_read = pickle.load(f)
    assert magic_number_read == magic_number, 'magic_number verification failed'
    print(accuracy)
