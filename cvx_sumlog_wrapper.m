function m = cvx_sumlog_wrapper(num_var_per_layer, obj_coeff, unaccounted_sv, comp_coeff, comp_rhs, mem_coeff, mem_rhs, timelimit)
%solve Equation 13 and 14 in microsoft cvpr paper. 

%num_var_per_layer: the number of reduced (free) variable in each layer
%obj_coeff: the coefficients for masking variables in the objective 
%unaccounted_sv: the singular values that are included by
    %default due to reduced_index, need to be added
%enable reduction: if false, unaccounted_sv is not used
%comp_coeff: constraint in computation cost, only for the reduced_index
%mem_coeff: constraint in memory cost, only for the reduced_index
%comp_rhs: right hand side of the constraint, if negative, ignore this
%constraint, only for the reduced_index

%return the value of the mask variables

cvx_startup;
%number of reduced (free) variables
n = sum(num_var_per_layer);
fprintf('cvx_sumlog_wrapper: number of reduced variables %d\n',n)

assert(size(num_var_per_layer,2) > 1, 'only 1 layer?');
assert(numel(unaccounted_sv) == numel(num_var_per_layer));
assert(n == numel(obj_coeff), 'number of masking variables does not match obj_coeff');
assert(n == numel(comp_coeff), 'number of masking variables does not match comp_coeff');
assert(n == numel(mem_coeff), 'number of masking variables does not match mem_coeff');

assert(size(num_var_per_layer,1) == 1, 'num_var_per_layer should be a vector');
assert(size(obj_coeff, 1) == 1);
assert(size(comp_coeff, 1) == 1);
assert(size(mem_coeff, 1) == 1);
assert(size(unaccounted_sv, 1) == 1);

%0 for gurobi, 1 for mosek
%gurobi does not seem to work
solver = 1;

cvx_begin quiet
%cvx_begin
    if solver == 0
        cvx_solver Gurobi_2
    else
        cvx_solver Mosek_2
    end
   
    if solver == 0
        if timelimit > 0
            cvx_solver_settings('TimeLimit', timelimit)
        end
        %cvx_solver_settings('MIPGap', 0.02)
        cvx_solver_settings('OutputFlag', 0)
    else
        %set solver settings
        cvx_solver_settings('MSK_DPAR_OPTIMIZER_MAX_TIME', timelimit )
        cvx_solver_settings('MSK_IPAR_NUM_THREADS', 32)
        %cvx_solver_settings('MSK_DPAR_INTPNT_TOL_REL_GAP',0.02)
        %cvx_solver_settings('MSK_DPAR_INTPNT_NL_TOL_REL_GAP', 0.02)
        %cvx_solver_settings('MSK_DPAR_MIO_NEAR_TOL_REL_GAP', 0.02)
        
        %terminate at x% of optimality gap
        %cvx_solver_settings('MSK_DPAR_MIO_TOL_REL_GAP', 0.01)
        %cvx_solver_settings('MSK_IPAR_LOG', 1)
    end
    
    %cvx_precision high
    cvx_precision medium
    %cvx_precision low
  
    variable m(1,n) binary
    %to use geo_mean, the result has to have more than one column and one
    %row
    all_cols = obj_coeff.*m;
    
    %compute the sum of each layer, and put them in a vector
    layer_sums = [];
    start_idx = 1;

    %the matrix for the constraint: in each layer, a singular value should not be discarded until all the values smaller than it are discarded
    mc = [];

    for i=1:size(num_var_per_layer,2)
        end_idx = start_idx + num_var_per_layer(i) - 1;
        assert(end_idx-start_idx+1 == num_var_per_layer(i));

        if i == size(num_var_per_layer,2)
            assert(end_idx == n);
        end

        %make sure the obj_coeff (singular values) are in descending order
        assert(issorted(fliplr(obj_coeff(start_idx:end_idx))) );
        layer_sums = cat(1, layer_sums, sum(all_cols(start_idx: end_idx)) + unaccounted_sv(i) );

        %to get the m mask variables for each layer in sorted order, need m-1 inequality constraints
        %this seems to make the solver works faster
        for j = 1:num_var_per_layer(i)-1
            %a Row in the Constraint matrix
            cr = zeros(1,n);
            cr(start_idx + j - 1) = 1;
            cr(start_idx + j) = -1;
            mc = cat(1,mc,cr);
        end

        %increment start_idx
        start_idx = start_idx + num_var_per_layer(i);
    end

    assert(size(mc,1) == n - size(num_var_per_layer, 2));
    assert(size(mc,2) == n);

    maximize(geo_mean(layer_sums));
  
    subject to
        mc*transpose(m) >= zeros(size(mc,1),1);
        if comp_rhs > 0
            dot(comp_coeff,m) <= comp_rhs;
        end
        if mem_rhs > 0
            dot(mem_coeff,m) <= mem_rhs;
        end
cvx_end

fprintf('~~~~~~~RESULT:product sum objective value (in log, to be compared with greedy): %.6E \n', size(num_var_per_layer,2)*log(cvx_optval))

%in MaskingVariableManager, 0 means the singular value is included, 1 means
%the singular value is discarded, so we have to revert the m values
for i = 1:length(m)
    if abs(m(i)-1) < 1e-5
        m(i) = 0;
    elseif abs(m(i)) < 1e-5
        m(i) = 1;
    else
        assert(1==0, sprintf('non binary value returned from solver: %f', m(i)));
    end
end

return
