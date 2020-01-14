import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')

import os
import subprocess
import pickle
import numpy as np

import time
import numpy as np
from collections import OrderedDict
from datetime import datetime

######
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-c','--computation_max', help='FLAGS.computation_max', type=float, default=None)
parser.add_argument('-m','--memory_max', help='FLAGS.memory_max', type=float, default=None)
parser.add_argument('-M','--model_name', help='model_name', type=str, required=True)
parser.add_argument('-N','--magic_number', help='a magic number to be saved with the results', type=int, default=None)

args = parser.parse_args()

#########################################
def _call_heuristics(model_name, computation_max, memory_max):

    assert 0<computation_max <= 1, 'got %f'%computation_max
    assert 0<memory_max <= 1, 'got %f'%memory_max

    wrapper_path = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --only_compute_mask_variable 1 --enable_reduction 0 '

    solution_list = list()
    heuristics = [2,3]

    #clear the solution file variable index file
    try:
        os.remove('/tmp/solution.pickle')
        os.remove('/tmp/variable_index.pickle')
    except OSError:
        pass

    #compute a mask variable solution using heuristics
    for heu in heuristics:
        cmd = wrapper_path + '--model_name %s --K_heuristic %d --computation_max %.3f --memory_max %.3f'%(model_name, heu, computation_max, memory_max )
        print(cmd)

        subprocess.call(cmd, shell=True)

        #load the result
        with open('/tmp/solution.pickle','rb') as f:
            solution_list.append(pickle.load(f))

    for sol in solution_list:
        assert sol.ndim == 1
        assert sol.shape[0] == solution_list[0].shape[0], 'mask varible solution is not of the same length'

    #convert to np array, each row is one solution
    solutions = np.array(solution_list) 

    #load the index of he variables
    with open('/tmp/variable_index.pickle','rb') as f:
        var_name_index = pickle.load(f)

    var_name_reduced_index = OrderedDict()

    for var_name,index in var_name_index.items():
        #the solution of the variable
        sv = solutions[:,index[0]:index[1]]
        num_binary_mask = index[1] - index[0]

        top_index = 0
        #0 for include the singular value
        while np.all(sv[:,top_index]==0) and top_index < num_binary_mask-1:
            top_index += 1

        bottom_index = num_binary_mask-1
        while np.all(sv[:,bottom_index]==1) and bottom_index > 0 :
            bottom_index -= 1

        #store the index, +1 because the last index is one past 
        var_name_reduced_index[var_name] = (top_index, bottom_index+1)

        reduced_index = var_name_reduced_index[var_name]
        assert reduced_index[0] <= reduced_index[1]
        assert reduced_index[1] > 0,' if the reduced_index is [0,0], all the binary variables are discarded, which should never happen'

        #if the reduced_index is [num_binary_mask, num_binary_mask], meaning all binary variables are kept
        #DEBUG
        if reduced_index[0] == reduced_index[1]:
            print('reduced_index equal, var_name: %s'%var_name)
            print(sv)

    return var_name_reduced_index, solutions
#########################################

#keep the singular values that are unanimously kept by all heuristic even when the constraint is stricter
#discard the singular values that are unanimously discarded by all heuristic even when the constraint is looser
stricter_constraint_factor = 1.3
looser_constraint_factor = 0.7

stricter_computation_max = 1-(1-args.computation_max)*stricter_constraint_factor
stricter_memory_max = 1-(1-args.memory_max)*stricter_constraint_factor

looser_computation_max = 1-(1-args.computation_max)*looser_constraint_factor
looser_memory_max = 1-(1-args.memory_max)*looser_constraint_factor

var_name_reduced_index_stricter, solutions_stricter = _call_heuristics(args.model_name, stricter_computation_max, stricter_memory_max)
var_name_reduced_index_looser, solutions_looser = _call_heuristics(args.model_name, looser_computation_max, looser_memory_max)

assert set(var_name_reduced_index_stricter.keys()) == set(var_name_reduced_index_looser.keys())

var_reduced_index = OrderedDict()
for var_name in var_name_reduced_index_looser:
    var_reduced_index[var_name] = (var_name_reduced_index_stricter[var_name][0], var_name_reduced_index_looser[var_name][1])

with open('/tmp/reduced_index.pickle','wb') as f:
    pickle.dump(var_reduced_index, f, protocol=-1)

with open('/tmp/test.pickle','wb') as f:
    pickle.dump((var_name_reduced_index_stricter, solutions_stricter, var_name_reduced_index_looser, solutions_looser  ), f, protocol=-1)
