import sys
sys.path.append('/home/chongli/research/sparse')
sys.path.append('/home/chongli/research/sparse/slim_utili')
import DnnUtili
import os

import subprocess
from random import randint
import pickle
import time
import numpy as np
from collections import OrderedDict
import matlab
import matlab.engine
import shutil

import scipy
import scipy.io

import mask_functions
############################
import os
import shutil
def copy_rename(new_file_name, to_folder, src_dir='/tmp/',old_file_name='sqp_solution.mat'):
    dst_dir= to_folder
    src_file = os.path.join(src_dir, old_file_name)
    shutil.copy(src_file,dst_dir)

    dst_file = os.path.join(dst_dir, old_file_name)
    new_dst_file_name = os.path.join(dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)


def mask_main(model_name, computation_max=None, memory_max=None, gradient_only=1, 
        provide_orth_v=0, hv_interval=5, num_hv_vectors=1, sample_percentage=0.1, hv_sample_percentage = 0.1, max_iteration=80,
        clear_all=1, compute_reduced_index=1, call_sqplab=1, sqplab_warmstart=0, monotonic=0, results_path=None, clear_results_path=False,
        s_factor=1.8, l_factor=0.2, heuristics='10,11',control_side='both', dxmin=1e-8, scheme='tai'):
############################
############################

    print('#################')
    print('#################')
    print('##mask_main.py parameters')
    print('#model_name: %s'%model_name)
    print('#computation_max/memory_max: %.2f/%.2f'%(computation_max,memory_max))
    print('#gradient_only: %d'%gradient_only)
    print('#sqplab_warmstart: %d'%sqplab_warmstart)
    print('#monotonic: %d'%monotonic)
    print('#results_path: %s'%results_path)
    print('#################')
    print('#################')


    start_time= time.time()
    if clear_all:
        #clear all the existing solutions in the tmp folder
        subprocess.call(r'rm -f /tmp/*.pickle',shell=True)
        subprocess.call(r'rm -f /tmp/*.mat',shell=True)
        subprocess.call(r'rm -rf /tmp/tensorflow_train',shell=True)

    if compute_reduced_index:
        #Step 1: compute the reduced index
        mask_functions.compute_reduced_index(model_name,computation_max, memory_max, s_factor, l_factor, heuristics,control_side,scheme)
        #output: /tmp/reduced_index.pickle, mapping from variable name to the reduced_index

    ####
    #Step 0: evaluate the solution using the computed reduced_index and a heuristic, currently using heuristic 0
    cmd = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --K_heuristic 0 '
    cmd += ' --model_name %s --gpu 0,1,2 --computation_max %.3f --memory_max %.3f --sample_percentage 0.1 '%(model_name, computation_max, memory_max);

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
    heuristic_accuracy_0 = accuracy

    #####
    ##Step 2: evaluate the solution using a heuristic, currently using heuristic 2
    #cmd = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --K_heuristic 2 '
    #cmd += ' --model_name %s --gpu 0,1,2 --computation_max %.3f --memory_max %.3f --sample_percentage 0.1'%(model_name, computation_max, memory_max);

    ##attach a magic number to veirfy the validaty of the subprocess call
    #magic_number = randint(0,999)
    #cmd += '--magic_number %d'%magic_number
    #print(cmd)

    ##external call
    #subprocess.call(cmd, shell=True)

    ##load the result
    #with open('/tmp/mask_wrapper_results.pickle','rb') as f:
    #    accuracy, magic_number_read = pickle.load(f)
    #assert magic_number_read == magic_number, 'magic_number verification failed'
    heuristic_accuracy_2 = accuracy

    ####
    #Step 3: evaluate the solution using a heuristic, currently using heuristic 3
    #cmd = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --enable_reduction 1 --load_reduced_index 1 --K_heuristic 3 '
    cmd = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py  --K_heuristic 3 '
    cmd += ' --model_name %s --gpu 0,1,2 --computation_max %.3f --memory_max %.3f --sample_percentage 0.1 '%(model_name, computation_max, memory_max);

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
    heuristic_accuracy_3 = accuracy

    ####
    #Step 4: evaluate the solution using the computed reduced_index and a heuristic, currently using heuristic 3
    #this will also dump a solution.pickle
    cmd = 'python3.5 /home/chongli/research/sparse/mask_wrapper.py --enable_reduction 1 --load_reduced_index 1 --K_heuristic 3 '
    cmd += ' --model_name %s --gpu 0,1,2 --computation_max %.3f --memory_max %.3f --sample_percentage 0.1 '%(model_name, computation_max, memory_max);

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
    heuristic_accuracy_3_reduced_index = accuracy

    if call_sqplab:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('K_heuristic 0 accuracy:        %s'%heuristic_accuracy_0)
        print('K_heuristic 2 accuracy:        %s'%heuristic_accuracy_2)
        print('K_heuristic 3 accuracy:        %s'%heuristic_accuracy_3)
        print('K_heuristic 3 accuracy reduced:        %s'%heuristic_accuracy_3_reduced_index)
        #Step 4: sequential quadratic programming, with the binary variables relaxed to continuous between 0-1
        eng = matlab.engine.start_matlab()
        #add path
        eng.addpath(r'/home/chongli/research/sparse',nargout=0)

        sqp_start_time= time.time()

        sqplab_num_call = 1
        assert sqplab_num_call == 1
        for i in range(sqplab_num_call):
            local_sample_percentage = sample_percentage * pow(2,i)
            local_hv_sample_percentage = hv_sample_percentage * pow(2,i)
            local_max_iteration = int(max_iteration /pow(2,i))
            #sqplab_warmstart in first iteration is set by the argument input
            if i == 0:
                local_sqplab_warmstart = sqplab_warmstart
                cost_saturation = 0
            else:
                #the following iterationn sqplab_warmstart is forced to 2
                local_sqplab_warmstart = 2
                cost_saturation = 1

            print('mask_main: sample_percentage and max_iteration scale factor is %d'%pow(2,i))
            print('mask_main:  dnn_sqplab_main(\'%s\',%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%d, %.2e, %d)'%(model_name, computation_max, memory_max, gradient_only, provide_orth_v, hv_interval, num_hv_vectors, local_sqplab_warmstart, monotonic, local_sample_percentage, local_hv_sample_percentage, local_max_iteration, dxmin, cost_saturation))
            eng.dnn_sqplab_main(model_name, computation_max, memory_max, gradient_only, provide_orth_v, hv_interval, num_hv_vectors, local_sqplab_warmstart, monotonic, local_sample_percentage, local_hv_sample_percentage, local_max_iteration,dxmin,cost_saturation,nargout=0)

            if i == 0 and (model_name not in ('vgg_16')):
                new_name = '%s_%.3f_%.3f_no_cost_saturation_sqp_solution.mat'%(model_name, computation_max, memory_max)
                copy_rename(new_name, results_path)

                #re-run with cost_saturation set to 1 and sqplab_warmstart set to 2
                print('mask_main:  dnn_sqplab_main(\'%s\',%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%d, %.2e, %d)'%(model_name, computation_max, memory_max, gradient_only, provide_orth_v, hv_interval, num_hv_vectors, 2, monotonic, local_sample_percentage, local_hv_sample_percentage, local_max_iteration, dxmin, 1))
                eng.dnn_sqplab_main(model_name, computation_max, memory_max, gradient_only, provide_orth_v, hv_interval, num_hv_vectors, 2, monotonic, local_sample_percentage, local_hv_sample_percentage, local_max_iteration,dxmin,1,nargout=0)


        sqp_time = time.time()-sqp_start_time

        eng.quit()

        #save a copy of the sqp solution
        if results_path is not None:
            new_name = '%s_%.3f_%.3f_sqp_solution.mat'%(model_name, computation_max, memory_max)
            copy_rename(new_name, results_path)
            #copy the reduced_index pickle file to result path
            new_name = '%s_%.3f_%.3f_reduced_index.pickle'%(model_name, computation_max, memory_max)
            copy_rename(new_name,results_path, '/tmp/','reduced_index.pickle')
    else:
        return

    #Step 5: evaluate the solution from sqp directly, mask variable is not integer
    cmd = 'python3.5 mask_wrapper.py --model_name %s --gpu 0,1,2 --load_solution 1 --solution_path /tmp/sqp_solution.mat --enable_reduction 1 --load_reduced_index 1 '%model_name
    if model_name not in ('vgg_16'):
        cmd += 'cost_saturation 1 '
    print(cmd)
    subprocess.call(cmd, shell=True)

    with open('/tmp/mask_wrapper_results.pickle','rb') as f:
        accuracy,_ = pickle.load(f)

    #load the time spent in tensorflow sess.run()
    #TODO not sure why the scalar is saved as array in array
    accuracy['tf_gradient_run_time'] = scipy.io.loadmat('/tmp/sqp_solution.mat')['tf_gradient_run_time'][0][0]
    accuracy['tf_eval_run_time'] = scipy.io.loadmat('/tmp/sqp_solution.mat')['tf_eval_run_time'][0][0]
    accuracy['sqp_niter'] = scipy.io.loadmat('/tmp/sqp_solution.mat')['niter'][0][0]
    accuracy.pop('tf_run_time', None)

    sqp_non_integer_accuracy = accuracy

    ##Step 6: convert the solution from sqp to integer using random rounding
    #cmd = 'python3.5 mask_wrapper.py --model_name %s --gpu 0,1,2 --load_solution 1 --solution_path /tmp/sqp_solution.mat --enable_reduction 1 --load_reduced_index 1 --solution_random_rounding 1'%model_name
    #print(cmd)
    #subprocess.call(cmd, shell=True)

    #with open('/tmp/mask_wrapper_results.pickle','rb') as f:
    #    accuracy,_ = pickle.load(f)
    #sqp_random_rounding_accuracy = accuracy

    ##Step 7: convert the solution from sqp to integer using cross entropy rounding
    #cmd = 'python3.5 mask_wrapper.py --model_name %s --gpu 0,1,2 --load_solution 1 --solution_path /tmp/sqp_solution.mat --enable_reduction 1 --load_reduced_index 1 --computation_max %.3f --memory_max %.3f --cross_entropy_rounding 1'%(model_name, computation_max, memory_max);
    #print(cmd)
    #subprocess.call(cmd, shell=True)

    #with open('/tmp/mask_wrapper_results.pickle','rb') as f:
    #    accuracy,_ = pickle.load(f)
    #sqp_cross_entropy_rounding_accuracy = accuracy

    #Step 8: convert the solution from sqp to integer using add_and_svd
    #run add_and_svd a couple of times and fine the best result
    num_add_and_svd = 10

    attribute_to_check = 'accuracy' if model_name == 'cifarnet' else 'accuracy_5'
    best_result = np.float32('-inf')
    best_accuracy = None

    for add_and_svd_i in range(num_add_and_svd):
        try:
            cmd = 'python3.5 mask_wrapper.py --model_name %s --gpu 0,1,2 --load_solution 1 --solution_path /tmp/sqp_solution.mat --enable_reduction 1 --load_reduced_index 1 --computation_max %.3f --memory_max %.3f --add_and_svd_rounding 1 '%(model_name, computation_max, memory_max);
            if model_name not in ('vgg_16'):
                cmd += 'cost_saturation 1 '
            print(cmd)
            subprocess.call(cmd, shell=True)

            _, _, constraint_satisfied = DnnUtili.calculate_percentage_add_and_svd(computation_max, memory_max, check_satisfied=True, cost_saturation=(model_name not in ('vgg_16')))
            if not constraint_satisfied:
                print('mask_main: add_and_svd computation and memory constraints not satisfied')
                continue

            with open('/tmp/mask_wrapper_results.pickle','rb') as f:
                accuracy,_ = pickle.load(f)

            if accuracy[attribute_to_check] > best_result:
                best_result = accuracy[attribute_to_check] 
                best_accuracy = accuracy
                shutil.copyfile('/tmp/add_and_svd_K.pickle','/tmp/best_add_and_svd_K.pickle')
        except KeyboardInterrupt:
            break

    assert best_accuracy is not None, 'no add_and_svd result is valid, maybe increase num_add_and_svd'

    #save a copy of add_and_svd_K.pickle
    if results_path is not None:
        new_name = '%s_%.3f_%.3f_add_and_svd_K.pickle'%(model_name, computation_max, memory_max)
        copy_rename(new_name,results_path, '/tmp/','best_add_and_svd_K.pickle')

    sqp_add_and_svd_rounding_accuracy = best_accuracy

    print('mask_main took %.1f seconds'%(time.time()-start_time))
    #compare results
    print('mask_main: compare results')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('K_heuristic 0 accuracy:        %s'%heuristic_accuracy_0)
    #print('K_heuristic 2 accuracy:        %s'%heuristic_accuracy_2)
    print('K_heuristic 3 accuracy:        %s'%heuristic_accuracy_3)
    print('K_heuristic 3 accuracy reduced:        %s'%heuristic_accuracy_3_reduced_index)
    print('SQP accuracy (non-integer  ): %s'%sqp_non_integer_accuracy)
    #print('SQP accuracy (rand rounding): %s'%sqp_random_rounding_accuracy)
    #print('SQP accuracy (cross entropy): %s'%sqp_cross_entropy_rounding_accuracy)
    print('SQP accuracy (add and svd):   %s'%sqp_add_and_svd_rounding_accuracy)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    results = OrderedDict()
    results['heuristic_0'] = heuristic_accuracy_0
    #results['heuristic_2'] = heuristic_accuracy_2
    results['heuristic_3'] = heuristic_accuracy_3
    results['sqp_non_integer'] = sqp_non_integer_accuracy
    #results['sqp_rand_rounding'] = sqp_random_rounding_accuracy
    #results['sqp_cross_entropy'] = sqp_cross_entropy_rounding_accuracy
    results['sqp_add_and_svd_rounding'] = sqp_add_and_svd_rounding_accuracy

    for ac in results.values():
        ac.pop('tf_run_time',None)

    #save the time spent in sqp in non_integer_accuracy
    results['sqp_time'] = sqp_time

    #save the accuracy stats to a pickle
    results_pickle_name = '%s_%.3f_%.3f_results.pickle'%(model_name, computation_max, memory_max)
    results_pickle_name = os.path.join(results_path, results_pickle_name)
    with open(results_pickle_name, 'wb') as f:
        pickle.dump(results, f, protocol=-1)

    return results

if __name__=='__main__':
    args = {'model_name':'squeezenet',
            'computation_max':0.8,
            'memory_max':0.8,
        #below are the settings for mask_main
        'call_sqplab':1,
        'clear_all':1,
        'compute_reduced_index':1,
        #only use gradient information to update hessian, hv computation is disabled
        'gradient_only':0,
        'provide_orth_v':0,
        'hv_interval':5,
        'num_hv_vectors':1,
        'sample_percentage':0.025,
        'hv_sample_percentage':0.1,
        'max_iteration': 60,
        #use the result from a heuristic as the initial point
        'sqplab_warmstart':1,
        'monotonic':0,
    }
    mask_main(**args)
