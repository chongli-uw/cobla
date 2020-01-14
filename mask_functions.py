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

#########################################
def _call_heuristics(model_name, computation_max, memory_max, heuristics, control_side):
    '''compute the solution of the mask variables given constraints using the heuristics specified in heuristics list,
    find the index up to which all the heuristics agree to keep or discard singular values'''

    assert 0<computation_max, 'got %f'%computation_max
    assert 0<memory_max, 'got %f'%memory_max

    if computation_max > 1:
        print('mask_functions: computation_max is %.3f'%computation_max)
    if memory_max > 1:
        print('mask_functions: memory_max is %.3f'%memory_max)


    wrapper_path = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --only_compute_mask_variable 1  --gpu 0  '

    solution_list = list()
    assert type(heuristics) == str
    heuristics = heuristics.split(',')
    for h in heuristics:
        assert h.isdigit()
    heuristics = [int(h) for h in heuristics]

    #clear the solution file variable index file
    try:
        os.remove('/tmp/solution.pickle')
        os.remove('/tmp/variable_index.pickle')
        os.remove('/tmp/reduced_variables.pickle')
    except OSError:
        pass

    #compute a mask variable solution using heuristics
    for heu in heuristics:
        cmd = wrapper_path + '--model_name %s --K_heuristic %d --computation_max %.3f --memory_max %.3f'%(model_name, heu, computation_max, memory_max )
        print('~calling: %s'%cmd)

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
        #the number of binary mask variables in this layer
        num_binary_mask = index[1] - index[0]

        top_index = 0
        #top index increment by one if all heuristics vote to keep it
        for i in range(0, num_binary_mask):
            #0 for include the singular value
            if np.all(sv[:,i]==0):
                top_index = i
            else:
                break

        bottom_index = num_binary_mask-1
        #bottom index decrease by one if all heuristics vote to keep it
        for i in reversed(range(0, num_binary_mask)):
            #1 for exclude the singular value
            if np.all(sv[:,i]==1):
                bottom_index = i
            else:
                break

        assert control_side in ('top','bottom','both','neither')
        if control_side == 'top':
            control_top = 1
            control_bottom = 0
        elif control_side == 'bottom':
            control_top = 0
            control_bottom = 1
        elif control_side == 'both':
            control_top = 1
            control_bottom = 1
        elif control_side == 'neither':
            control_top = 0
            control_bottom = 0

        if not control_top:
            #don't control the included left side
            top_index = 0
        if not control_bottom:
            #don't control the discarded right side
            bottom_index = num_binary_mask-1

        #if the top_index has moved to the end of the array, then +1 to indicate this layer should not be reduced at all
        #for example, with a layer of 10 variables, reduced_index of [9,9] means no truncation, 
        if top_index == num_binary_mask - 1:
            #DEBUG
            assert bottom_index == num_binary_mask - 1, 'num_binary_mask: %d, top_index: %d, bottom_index: %d'%(num_binary_mask, top_index, bottom_index)
            top_index += 1

        #store the index, +1 because the last index is one past 
        var_name_reduced_index[var_name] = (top_index, bottom_index+1)

        reduced_index = var_name_reduced_index[var_name]
        assert reduced_index[0] <= reduced_index[1]
        assert reduced_index[1] > 0,' if the reduced_index is [0,0], all the binary variables are discarded, which should never happen'

        #if the reduced_index is [num_binary_mask, num_binary_mask], meaning all binary variables are kept
        #DEBUG
        #if reduced_index[0] == reduced_index[1]:
        #    print('reduced_index equal, var_name: %s'%var_name)
        #    print(sv)


    return var_name_reduced_index, solutions
#########################################
def compute_reduced_index(model_name, computation_max, memory_max, s_factor=1.8, l_factor=0.2, heuristics='0,2,3',control_side='both',scheme=None):
    '''
    keep the singular values that are unanimously kept by all heuristic even when the constraint is stricter
    discard the singular values that are unanimously discarded by all heuristic even when the constraint is looser

    return: None, results are saved in /tmp/reduced_index.pickle
    '''

    if heuristics == 'manual':
        manual_reduced_index(model_name, scheme)
        return

    stricter_constraint_factor = s_factor
    looser_constraint_factor = l_factor

    print('mask_function:compute_reduced_index: s_factor: %.1f, l_factor: %.1f, heuristics: %s, control_side: %s'%(s_factor,l_factor,heuristics,control_side))

    #stricter_computation_max = max(1-(1-computation_max)*stricter_constraint_factor, computation_max/2)
    stricter_computation_max = max(1-(1-computation_max)*stricter_constraint_factor, 0.03)
    stricter_memory_max = max(1-(1-memory_max)*stricter_constraint_factor, 0.03)

    looser_computation_max = min(1-(1-computation_max)*looser_constraint_factor, 1.0)
    looser_memory_max = min(1-(1-memory_max)*looser_constraint_factor, 1.0)

    var_name_reduced_index_stricter, solutions_stricter = _call_heuristics(model_name, stricter_computation_max, stricter_memory_max, heuristics,control_side)
    var_name_reduced_index_looser, solutions_looser = _call_heuristics(model_name, looser_computation_max, looser_memory_max, heuristics,control_side)

    assert set(var_name_reduced_index_stricter.keys()) == set(var_name_reduced_index_looser.keys())

    var_reduced_index = OrderedDict()
    for var_name in var_name_reduced_index_looser:
        var_reduced_index[var_name] = (var_name_reduced_index_stricter[var_name][0], var_name_reduced_index_looser[var_name][1])
        assert var_reduced_index[var_name][0] <= var_reduced_index[var_name][1], 'not a valid index, %s, %s'%(var_name, var_reduced_index[var_name])

    with open('/tmp/reduced_index.pickle','wb') as f:
        pickle.dump(var_reduced_index, f, protocol=-1)
    print('---compute_reduced_index saved reduced_index in %s at %s'%('/tmp/reduced_index.pickle', datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))


def manual_reduced_index(model_name, scheme):

    assert scheme in ('tai','microsoft')
    #clear the solution file variable index file
    try:
        os.remove('/tmp/solution.pickle')
        os.remove('/tmp/variable_index.pickle')
        os.remove('/tmp/reduced_variables.pickle')
    except OSError:
        pass

    wrapper_path = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --only_compute_mask_variable 1  --gpu 0 --K_heuristic 3 --computation_max 1.0 --memory_max 1.0  '
    cmd = wrapper_path + '--model_name %s '%(model_name)
    print('~calling: %s'%cmd)

    subprocess.call(cmd, shell=True)

    #load the index of the variables
    with open('/tmp/variable_index.pickle','rb') as f:
        var_name_index = pickle.load(f)

    var_name_reduced_index = OrderedDict()

    if model_name == 'vgg_16':
        if scheme=='tai':
            #var_name_reduced_index['vgg_16/conv1/conv1_1/S_masks'] = (0, 9) #5
            #var_name_reduced_index['vgg_16/conv1/conv1_2/S_masks'] = (0, 192) #24
            #var_name_reduced_index['vgg_16/conv2/conv2_1/S_masks'] = (0, 192) #48
            #var_name_reduced_index['vgg_16/conv2/conv2_2/S_masks'] = (0, 384) #48
            #var_name_reduced_index['vgg_16/conv3/conv3_1/S_masks'] = (0, 384) #64
            #var_name_reduced_index['vgg_16/conv3/conv3_2/S_masks'] = (0, 768) #128
            #var_name_reduced_index['vgg_16/conv3/conv3_3/S_masks'] = (0, 768) #160
            #var_name_reduced_index['vgg_16/conv4/conv4_1/S_masks'] = (0, 768) #192
            #var_name_reduced_index['vgg_16/conv4/conv4_2/S_masks'] = (0, 1536) #192
            #var_name_reduced_index['vgg_16/conv4/conv4_3/S_masks'] = (0, 1536) #256
            #var_name_reduced_index['vgg_16/conv5/conv5_1/S_masks'] = (0, 1536) #320
            #var_name_reduced_index['vgg_16/conv5/conv5_2/S_masks'] = (0, 1536) #320
            #var_name_reduced_index['vgg_16/conv5/conv5_3/S_masks'] = (0, 1536) #320

            var_name_reduced_index['vgg_16/conv1/conv1_1/S_masks'] = (3, 9) #5
            var_name_reduced_index['vgg_16/conv1/conv1_2/S_masks'] = (12, 64) #24
            var_name_reduced_index['vgg_16/conv2/conv2_1/S_masks'] = (24, 96) #48
            var_name_reduced_index['vgg_16/conv2/conv2_2/S_masks'] = (24, 96) #48
            var_name_reduced_index['vgg_16/conv3/conv3_1/S_masks'] = (32, 100) #64
            var_name_reduced_index['vgg_16/conv3/conv3_2/S_masks'] = (64, 200) #128
            var_name_reduced_index['vgg_16/conv3/conv3_3/S_masks'] = (80, 240) #160
            var_name_reduced_index['vgg_16/conv4/conv4_1/S_masks'] = (100, 300) #192
            var_name_reduced_index['vgg_16/conv4/conv4_2/S_masks'] = (100, 300) #192
            var_name_reduced_index['vgg_16/conv4/conv4_3/S_masks'] = (200, 500) #256
            var_name_reduced_index['vgg_16/conv5/conv5_1/S_masks'] = (200, 500) #320
            var_name_reduced_index['vgg_16/conv5/conv5_2/S_masks'] = (200, 500) #320
            var_name_reduced_index['vgg_16/conv5/conv5_3/S_masks'] = (200, 500) #320
        elif scheme=='microsoft':
            #var_name_reduced_index['vgg_16/conv1/conv1_1/S_masks'] = (0, 27)
            #var_name_reduced_index['vgg_16/conv1/conv1_2/S_masks'] = (0, 64)
            #var_name_reduced_index['vgg_16/conv2/conv2_1/S_masks'] = (0, 128)
            #var_name_reduced_index['vgg_16/conv2/conv2_2/S_masks'] = (0, 128)
            #var_name_reduced_index['vgg_16/conv3/conv3_1/S_masks'] = (0, 256)
            #var_name_reduced_index['vgg_16/conv3/conv3_2/S_masks'] = (0, 256)
            #var_name_reduced_index['vgg_16/conv3/conv3_3/S_masks'] = (0, 256)
            #var_name_reduced_index['vgg_16/conv4/conv4_1/S_masks'] = (0, 512)
            #var_name_reduced_index['vgg_16/conv4/conv4_2/S_masks'] = (0, 512)
            #var_name_reduced_index['vgg_16/conv4/conv4_3/S_masks'] = (0, 512)
            #var_name_reduced_index['vgg_16/conv5/conv5_1/S_masks'] = (0, 512)
            #var_name_reduced_index['vgg_16/conv5/conv5_2/S_masks'] = (0, 512)
            #var_name_reduced_index['vgg_16/conv5/conv5_3/S_masks'] = (0, 512)

            var_name_reduced_index['vgg_16/conv1/conv1_1/S_masks'] = (3, 25)
            var_name_reduced_index['vgg_16/conv1/conv1_2/S_masks'] = (8, 50)
            var_name_reduced_index['vgg_16/conv2/conv2_1/S_masks'] = (8, 100)
            var_name_reduced_index['vgg_16/conv2/conv2_2/S_masks'] = (8, 100)
            var_name_reduced_index['vgg_16/conv3/conv3_1/S_masks'] = (8, 128)
            var_name_reduced_index['vgg_16/conv3/conv3_2/S_masks'] = (16, 128)
            var_name_reduced_index['vgg_16/conv3/conv3_3/S_masks'] = (16, 128)
            var_name_reduced_index['vgg_16/conv4/conv4_1/S_masks'] = (16, 256)
            var_name_reduced_index['vgg_16/conv4/conv4_2/S_masks'] = (16, 256)
            var_name_reduced_index['vgg_16/conv4/conv4_3/S_masks'] = (16, 256)
            var_name_reduced_index['vgg_16/conv5/conv5_1/S_masks'] = (16, 300)
            var_name_reduced_index['vgg_16/conv5/conv5_2/S_masks'] = (16, 300)
            var_name_reduced_index['vgg_16/conv5/conv5_3/S_masks'] = (16, 300)
        else:
            raise ValueError()
    else:
        raise ValueError()

    #double check that manually reduced index is within range
    for var_name,index in var_name_index.items():
        assert var_name in var_name_reduced_index, '%s'%var_name
        assert index[1] - index[0] >= var_name_reduced_index[var_name][1], '%s'%var_name
        assert var_name_reduced_index[var_name][0] <= var_name_reduced_index[var_name][1],'%s'%var_name

    with open('/tmp/reduced_index.pickle','wb') as f:
        pickle.dump(var_name_reduced_index, f, protocol=-1)
    print('---manual_reduced_index saved reduced_index in %s at %s'%('/tmp/reduced_index.pickle', datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))
