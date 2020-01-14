function [outdic,out2,out3,out4,out5,out6,out7,out8] = dnn_sim(indic,x,lm)
% [outdic,f,ci,ce,cs,g,ai,ae] = simulopt (indic,xy)

%DEBUG
%fprintf('indic is %d\n', indic)
%%%%%%%%%%%%
%the loss value and gradient of the network given x
persistent computed_x;
persistent computed_f;
persistent computed_gradient;
persistent computed_ci;
%persistent computed_ce;
persistent computed_ai;
%persistent computed_ae;
%%%%%%%%%%%%
global model_name;
global computation_max;
global memory_max;
global sample_percentage;

global monotonic;
global mc;
global tf_gradient_run_time;
global tf_eval_run_time;

global cost_saturation;
%%%%%%%%%%%%
% On the output arguments
outdic = [];
out2   = [];
out3   = [];
out4   = [];
out5   = [];
out6   = [];
out7   = [];
out8   = [];

x = double(x);

if (indic >= 2) && (indic <= 4)
    %if result is pre-computed, just return cached result
    if ~isempty(computed_x)
        if is_close(computed_x,x)
            out2 = computed_f;
            out6 = computed_gradient;
            out3 = computed_ci;
            %out4 = computed_ce;
            out7 = computed_ai;
            %out8 = computed_ae;
            outdic = 0;
            
            %DEBUG
            %fprintf('dnn_sim: used pre-computed result, indic: %d, time: %s\n', indic, datestr(now))
            return;
        end
    end
    
    %load the coefficients from the mat
    load('/tmp/cost_coeff.mat','computation_coeff');
    load('/tmp/cost_coeff.mat','memory_coeff');
    load('/tmp/cost_coeff.mat','unaccounted_computation_top');
    load('/tmp/cost_coeff.mat','unaccounted_memory_top');
    load('/tmp/cost_coeff.mat','original_computation_cost_per_layer');
    load('/tmp/cost_coeff.mat','original_memory_cost_per_layer');
    load('/tmp/cost_coeff.mat','num_var_per_layer');
    load('/tmp/cost_coeff.mat','unaccounted_computation_top_per_layer');
    load('/tmp/cost_coeff.mat','unaccounted_memory_top_per_layer');
    computation_coeff = double(computation_coeff);
    memory_coeff = double(memory_coeff);
    assert(size(computation_coeff,1) == 1);
    assert(size(memory_coeff,1) == 1);
    assert(numel(unaccounted_computation_top_per_layer)==numel(unaccounted_memory_top_per_layer));
    assert(numel(unaccounted_computation_top_per_layer)==numel(original_computation_cost_per_layer));
    assert(numel(unaccounted_computation_top_per_layer)==numel(num_var_per_layer));
    
    if cost_saturation
        assert(~all(x==0), 'should not use cost_saturation using an all on initial solution\n')
        %inequality constraint
        ci = zeros(2,1);
        %jacobian of ineuality constraint
        ai = [];
        
        x = double(x);
        
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
            if layer_computation_cost>=original_computation_cost_per_layer(i) || layer_memory_cost>=original_memory_cost_per_layer(i)
                %fprintf('dnn_sim: cost saturation, layer index: %d, %d\n', start_idx, end_idx)
                ci = ci + [layer_all_on_comp + unaccounted_computation_top_per_layer(i) - original_computation_cost_per_layer(i);
                        layer_all_on_mem + unaccounted_memory_top_per_layer(i) - original_memory_cost_per_layer(i)];
                ai = cat(2,ai,[zeros(size(layer_computation_coeff));zeros(size(layer_memory_coeff))]);
                
                %if the cost is higher than the original cost, then set all masks to be on, which is equivlent to using the original un-decomposed layer
                %x(start_idx:end_idx) = 0.0;
            else
                ci = ci + [dot(x(start_idx:end_idx),layer_computation_coeff); dot(x(start_idx:end_idx),layer_memory_coeff)];
                ai = cat(2,ai,[computation_coeff(start_idx:end_idx);memory_coeff(start_idx:end_idx)]);
            end
            
            %increment start_idx
            start_idx = start_idx + num_var_per_layer(i);
        end
        
        %ai = [computation_coeff;memory_coeff];

        %DEBUG
        %fprintf('dnn_sim:ci\n')
        %disp((sum(computation_coeff) - ci(1) + unaccounted_computation_top)/sum(original_computation_cost_per_layer))
        %disp((sum(memory_coeff) - ci(2) + unaccounted_memory_top)/sum(original_memory_cost_per_layer))
        
    else
        %inequality constraint
        ci = [dot(x,computation_coeff); dot(x,memory_coeff)];
        %jacobian of ineuality constraint
        ai = [computation_coeff;memory_coeff];
    end

    
    %fprintf('dnn_sim: computing f,ci, indic: %d, time: %s\n', indic, datestr(now))
    %save current x into a mat file
    save('/tmp/current_x.mat','x');
    
    %do a evaluation, given the current x
    magic_number_input = randi([0,999],1);
    
    %compute the loss and gradients
    cmd = 'python3.5 /home/chongli/research/sparse/compute_hessian_hv.py --solution_path /tmp/current_x.mat --pickle_path /tmp/hessian_hv.pickle ';
    cmd = cat(2,cmd, sprintf(' --model_name %s --gpu 0,1,2 --computation_max %.3f --memory_max %.3f --magic_number %d --compute_gradient 1 --compute_loss 1 --sample_percentage %.3f',model_name, computation_max, memory_max, magic_number_input, sample_percentage ));
    if cost_saturation
        cmd = cat(2,cmd,' --cost_saturation 1 ');
    end
    %DEBUG
    %fprintf('%s\n',cmd);
    
    [status,cmdout] = system(cmd);
    assert(status==0, sprintf('command failed: %s \n %s',cmd,cmdout));
    
    %load the mask_wrapper_results
    load('/tmp/mask_wrapper_results.mat');
    assert(magic_number_input == mask_wrapper_results.magic_number);
    f = double(mask_wrapper_results.loss);
    accuracy = double(mask_wrapper_results.accuracy);
    accuracy_5 = double(mask_wrapper_results.accuracy_5);
    comp_cost = double(mask_wrapper_results.computation_cost);
    mem_cost = double(mask_wrapper_results.memory_cost);
    %DEBUG
    fprintf('dnn_sim: loss: %.3f, accuracy: %.3f, accuracy_5: %.3f, comp_cost: %.3f, mem_cost: %.3f, indic: %d, time(s): %.0f \n',f, accuracy, accuracy_5, comp_cost, mem_cost, indic, mask_wrapper_results.tf_run_time);
    
    out2 = f;
    
    assert(isscalar(f),'loss value is not scalar');
    
    %record the time spent in tensorflow sess.run()
    tf_eval_run_time = tf_eval_run_time + mask_wrapper_results.tf_run_time;
    
    %load the result
    load('/tmp/hessian_hv.mat','magic_number');
    assert(magic_number_input == magic_number, 'magic_number does not match, maybe last function call failed')
    load('/tmp/hessian_hv.mat','gradient');
    
    gradient = double(transpose(gradient));
    %make sure gradient and jacobian is of right size
    assert(iscolumn(gradient), 'size of gradient is %s, not column', mat2str(size(gradient)));
    assert(size(gradient,1) == size(x,1), 'size of gradient: %s, != size of x: %s', mat2str(size(gradient,1)), mat2str(size(x,1)));
    
    %%set the gradients of the variables in the un-decomposed layer to 0
    %if cost_saturation
    %    assert(~all(x==0), 'should not use cost_saturation using an all on initial solution\n')
    %    start_idx = 1;
    %    for i=1:numel(num_var_per_layer)
    %        assert(num_var_per_layer(i)>=0);
    %        end_idx = start_idx + num_var_per_layer(i) - 1;
    %        
    %        %the computation cost coefficients
    %        layer_computation_coeff = computation_coeff(start_idx:end_idx);
    %        layer_memory_coeff = memory_coeff(start_idx:end_idx);
    %        
    %        %this is different from original cost, since decomposition may increase cost
    %        layer_all_on_comp = sum(layer_computation_coeff);
    %        layer_all_on_mem = sum(layer_memory_coeff);
    %        
    %        layer_computation_cost = layer_all_on_comp + unaccounted_computation_top - dot(x(start_idx:end_idx),layer_computation_coeff);
    %        layer_memory_cost = layer_all_on_mem + unaccounted_memory_top - dot(x(start_idx:end_idx),layer_memory_coeff);
    %        
    %        %if decomposed cost is higher than using un-decomposed layer
    %        if layer_computation_cost>original_computation_cost_per_layer(i) || layer_memory_cost>original_memory_cost_per_layer(i)
    %            gradient(start_idx:end_idx) = 0;
    %        end
    %    end
    %end
    
    out6 = gradient;
    
    
    if monotonic
        assert(size(mc, 2) == size(computation_coeff, 2));
        assert(size(mc, 2) == size(memory_coeff, 2));
        ci = cat(1, ci, sum(x'.*mc, 2));
        ai = cat(1, ai, mc);
    end
    
    out3 = ci;
    out7 = ai;
    
    outdic = 0;
    
    %save the computed result
    computed_x = x;
    computed_f = f;
    computed_gradient = gradient;
    computed_ci = ci;
    computed_ai = ai;
    
    return
elseif indic == 0 || indic == 1
    %do nothing
    outdic = 0;
    return
else
    fprintf('\n dnnsim:unexpected value of indic (=%i)\n\n',indic);
    outdic = -2;
    return
end


end



function result = is_close(x,y)
%test if x, y are close up to a tolerance
tolerance = 5*1.2e-7;

if numel(size(x)) ~= numel(size(y))
    result = 0;
    return
else
    for i = 1:numel(size(x))
        if size(x,i) ~= size(y,i)
            result = 0;
            return
        end
    end
end

%up to this point x and y are of same size
result = all(abs(x-y) < tolerance);
return

end
