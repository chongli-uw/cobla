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
#hessian_pickle_path = '/tmp/hessian_gn_full_inner0.pickle'
#hessian_pickle_path = '/tmp/hessian_gn_inner0.pickle'
#hessian_pickle_path = '/tmp/hessian_cifar_751.pickle'
hessian_pickle_path = '/tmp/hessian_cifar_751_gn.pickle'

#gradually lower the computation/memory max
computation_max = 0.2
memory_max = 0.2
call_gurobi=True

timelimit = 900


for i in range(3):
    print('----------------------------------------------------------------Iteration %d-------------------'%i)

    if i == 0:
        computation_max = 0.7
        memory_max = 0.7
    elif i == 1:
        computation_max = 0.45
        memory_max = 0.45
    elif i == 2:
        computation_max = 0.2
        memory_max = 0.2

    cmd = wrapper_path + '--model_name %s --hessian_pickle_path %s '%(model_name, hessian_pickle_path)
    cmd += '--call_gurobi %d '%call_gurobi
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

    print('!!!!!!!!!!!!!!!!!!!!')
    print(accuracy)
    print('!!!!!!!!!!!!!!!!!!!!')

    #don't need to compute hessian again in last iteration
    if i == 2:
        break
    print('recompute the hessian, given the current solution')
    hessian_pickle_path = '/tmp/hessian_iter.pickle'
    hessian_cmd = 'python3.5 compute_hessian_gn.py --gpu 0 --load_mask_variable 1 --mask_variable_pickle_path /tmp/solution.pickle --pickle_path %s'%hessian_pickle_path

    print(hessian_cmd)
    subprocess.call(hessian_cmd, shell=True)

    #call solver again, using the newly computed hessian

