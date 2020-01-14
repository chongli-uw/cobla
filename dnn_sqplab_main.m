%%
function dnn_sqplab_main(model_name_in, computation_max_in, memory_max_in, ...
gradient_only_in, provide_orth_v, hv_interval, num_hv_vectors, warmstart, ...
monotonic_in, sample_percentage_in, hv_sample_percentage, max_iteration, ...
dxmin_input, cost_saturation_in)
addpath('/home/chongli/Dropbox/research/sparse/sqplab/src');
addpath('/home/cad/mosek/8/toolbox/r2014a');


%%%%%%%%%%%%
global model_name;
global computation_max;
global memory_max;
global sample_percentage;
sample_percentage = sample_percentage_in;

model_name = string(model_name_in);
computation_max = double(computation_max_in);
memory_max = double(memory_max_in);

global monotonic;
monotonic = monotonic_in;
%the constraints for monotonic
global mc;

%tiem spent in tensorflow tf.run() for computing gradient and evaluation
global tf_gradient_run_time;
global tf_eval_run_time;
tf_gradient_run_time = 0.0;
tf_eval_run_time = 0.0;

%number of dxmin violation
global num_dxmin_violation;
num_dxmin_violation = 0;

%for a layer, if the cost of decomposed layer is higher than original
%un-decomposed layer, than use the un-decomposed layer
global cost_saturation;
cost_saturation = cost_saturation_in;

%%%%%%%%%%%%
options.algo_method        = 'quasi-Newton';
%options.algo_method        = 'Newton';
options.algo_globalization = 'line-search';
%options.algo_globalization = 'unit step-size';
%options.algo_descent = 'Wolfe';
options.algo_descent = 'Powell';
options.inf = 5;
%options.verbose = 3;
%options.miter   = 1000;       % max iterations
options.miter   = max_iteration;       % max iterations
options.msimul  = 10000;      % max simulations
options.dxmin = dxmin_input; %smallest change in x in infinity norm as stop criteria, default is 1e-8
assert(options.dxmin <= 1e-3 && options.dxmin>=1e-11);

options.gradient_only = gradient_only_in; 
%in computing hessian vector product, whether to use random orthnormal basis or provide in sqplab_bfgs.m
options.provide_orth_v = provide_orth_v;
%interval to compute Hv to correct the estimation of hessian (compute Hv every x iterations), could be stochastic (compute Hv with 1/x probabillity)
options.hv_interval = hv_interval;
options.hv_sample_percentage = hv_sample_percentage;
%number of (v, Hv) pairs to compute if choose to do Hv correction
options.num_hv_vectors = num_hv_vectors;
%with the same amount of computation, we can experiment to use a larger hv_num_v but smaller hv_sample_percentage, or small hv_num_v but large hv_sample_percentage

%debug
%options.verbose = 4;
%%%%%%%%%%%%

%load initial solution (reduced_index) from a heuristic
solution_mat = load('/tmp/solution.mat');
solution = solution_mat.solution;

if warmstart == 2
    if exist('/tmp/sqp_solution.mat','file')==2
        sqp_solution_mat = load('/tmp/sqp_solution.mat');
        if numel(solution) == numel(sqp_solution_mat.x)
            fprintf('dnn_sqplab_main: using sqp_solution.mat as initial solution.\n')
            solution = transpose(sqp_solution_mat.x);
            if isfield(sqp_solution_mat,'info')
                options.initial_M = sqp_solution_mat.info.M;
            end
            if isfield(sqp_solution_mat,'tf_eval_run_time')
                tf_eval_run_time = sqp_solution_mat.tf_eval_run_time;
            end
        elseif isfield(sqp_solution_mat, 'reduced_x') 
            assert(numel(solution) == numel(sqp_solution_mat.reduced_x));
            fprintf('dnn_sqplab_main: using reduced_x in sqp_solution.mat as initial solution.\n')
            solution = double(sqp_solution_mat.reduced_x);
            tf_eval_run_time = sqp_solution_mat.tf_eval_run_time;
        else
            fprintf('dnn_sqplab_main: using all on due to size mismatch.\n')
            solution = zeros(size(solution));
        end
    else
        warning(sprintf('dnn_sqplab_main: warmstart is set to 2 but sqp_solution.mat is not found. Using all on as initial  \n'));
        solution = zeros(size(solution));
    end
elseif warmstart ==1
    %use solution from heuristic
    fprintf('dnn_sqplab_main: using solution.pickle as initial solution.\n')
    solution = solution_mat.solution;
else
    %all mask on
    fprintf('dnn_sqplab_main: using all on as initial solution.\n')
    solution = zeros(size(solution));
end
%initial solution has to be a column vector
if ~iscolumn(solution)
    solution = transpose(solution);
end
assert(iscolumn(solution));
solution=double(solution);
fprintf('dnn_sqplab_main: size of problem is %d\n',numel(solution));

num_variables = numel(solution);
%[x,lm,info] = sqplab (@simul, x, lm, lb, ub, options)

num_inequality_constraints = 2;

%initialize row vector
lm = [];
lb = zeros(1, (num_variables+num_inequality_constraints));
ub = ones(1, (num_variables+num_inequality_constraints));

%compute the bound, the solution.*coeff is the cost that is discarded
%total computation cost is the total cost of un-decomposed layers
load('/tmp/cost_coeff.mat','total_computation_cost');
load('/tmp/cost_coeff.mat','total_memory_cost');
load('/tmp/cost_coeff.mat','computation_coeff');
load('/tmp/cost_coeff.mat','memory_coeff');
load('/tmp/cost_coeff.mat','unaccounted_computation_top');
load('/tmp/cost_coeff.mat','unaccounted_memory_top');
load('/tmp/cost_coeff.mat','original_computation_cost_per_layer');
load('/tmp/cost_coeff.mat','original_memory_cost_per_layer');
load('/tmp/cost_coeff.mat','num_var_per_layer');
load('/tmp/cost_coeff.mat','unaccounted_computation_top_per_layer');
load('/tmp/cost_coeff.mat','unaccounted_memory_top_per_layer');
assert(numel(unaccounted_computation_top_per_layer)==numel(unaccounted_memory_top_per_layer));
assert(numel(unaccounted_computation_top_per_layer)==numel(original_computation_cost_per_layer));
assert(numel(unaccounted_computation_top_per_layer)==numel(num_var_per_layer));
assert(size(computation_coeff,1) == 1);
assert(size(memory_coeff,1) == 1);
computation_coeff = double(computation_coeff);
memory_coeff = double(memory_coeff);

%find out the reduction of cost
%all_on cost is different from total cost, total cost is the original cost
%of the undecomposed network, all_on cost is the cost that all the singular
%values are kept using the selected decomposition
all_on_comp = sum(computation_coeff);
all_on_mem = sum(memory_coeff);

lb(num_variables+1) = double(all_on_comp - computation_max*total_computation_cost + unaccounted_computation_top);
lb(num_variables+2) = double(all_on_mem - memory_max*total_memory_cost + unaccounted_memory_top);
ub(num_variables+1:end) = [inf,inf];
    

%for the layers whose cost is higher than the un-decomposed, set the mask to be on all, and set lb and ub to both 0
if cost_saturation
    x = double(solution);
    assert(~all(x==0), 'should not use cost_saturation using an all on initial solution\n')

    start_idx = 1;
    for i=1:numel(num_var_per_layer)
        assert(num_var_per_layer(i)>=0);
        end_idx = start_idx + num_var_per_layer(i) - 1;
        
        %the computation cost coefficients
        layer_computation_coeff = computation_coeff(start_idx:end_idx);
        layer_memory_coeff = memory_coeff(start_idx:end_idx);
        
        %this is different from original cost, since decomposition may increase cost
        layer_all_on_comp = sum(layer_computation_coeff);
        layer_all_on_mem = sum(layer_memory_coeff);
        
        layer_computation_cost = layer_all_on_comp + unaccounted_computation_top_per_layer(i) - dot(x(start_idx:end_idx),layer_computation_coeff);
        layer_memory_cost = layer_all_on_mem + unaccounted_memory_top_per_layer(i) - dot(x(start_idx:end_idx),layer_memory_coeff);
        
        %if decomposed cost is higher than using un-decomposed layer
        if layer_computation_cost>original_computation_cost_per_layer(i) || layer_memory_cost>original_memory_cost_per_layer(i)
            fprintf('dnn_sqplab_main: cost saturation, layer index: %d, %d\n', start_idx, end_idx)
            x(start_idx:end_idx) = 0;
            ub(start_idx:end_idx) = 0;
        end
        
        %increment start_idx
        start_idx = start_idx + num_var_per_layer(i);
    end
    solution = x;
end

if monotonic
    load('/tmp/cost_coeff.mat','num_var_per_layer');
    assert(exist('num_var_per_layer', 'var') == 1, 'failed to load num_var_per_layer from mat file');
    n = sum(num_var_per_layer);
    assert(isvector(num_var_per_layer));
    mc = [];
    
    start_idx = 1;
    for i=1:numel(num_var_per_layer)
        if num_var_per_layer(i) == 0
            continue
        end
        end_idx = start_idx + num_var_per_layer(i) - 1;
        assert(end_idx-start_idx+1 == num_var_per_layer(i));

        if i == numel(num_var_per_layer)
            assert(end_idx == n);
        end

        %to get the m mask variables for each layer in sorted order, need m-1 inequality constraints
        for j = 1:num_var_per_layer(i)-1
            %a Row in the Constraint matrix
            cr = zeros(1,n);
            cr(start_idx + j - 1) = 1;
            cr(start_idx + j) = -1;
            
            %DEBUG
            %only randomly add a subset of the constraints
            if rand(1) < 0.2
                continue;
            end
            %DEBUG end
            
            mc = cat(1,mc,cr);
        end

        %increment start_idx
        start_idx = start_idx + num_var_per_layer(i);
    end
    %number of constraints
    %assert(size(mc,1) == n - numel(find(num_var_per_layer)));
    assert(size(mc,2) == n);
    
    lb = cat(2,lb, zeros(1, size(mc,1)));
    ub = cat(2,ub, inf*ones(1, size(mc,1)));
end

t = cputime;
[x,lm,info] = sqplab(@dnn_sim,solution,lm,lb,ub,options);
time_used = cputime-t;

fprintf('\n---------------------\n')
fprintf('\n---sqplab returned %.3f in %d iterations, %.1f seconds\n', info.f, info.niter, time_used)
fprintf('\n---------------------\n')

