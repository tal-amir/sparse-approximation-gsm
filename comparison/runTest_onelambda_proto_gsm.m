function result = runTest_onelambda_proto_gsm(A, x0, y, k, lambda, lambda_base, lambda_rel, eta, residual_stop_thresh, optimization_success_thresh, recovery_success_thresh, noiseStrength, noiseSigma)
[m,n] = size(A);

result = struct();

calc_obj = @(x) 0.5*norm(A*x-y)^2 + lambda*trimmedLasso(x,k);

fprintf('\nObjective of x0: %g\n\n', calc_obj(x0));


%% Solve with Basis Pursuit
if true
    methodName = 'bp';    
        
    startTimes = getStartTimes();
    
    x_bp = solve_bp(A,y,k,0); 

    timeStats = getTimeStats(startTimes);    
    
    perf = evaluatePerformance_onelambda(x_bp, A, y, k, lambda, lambda_base, lambda_rel, x0, timeStats, 1);
    
    result = structMerge(result, perf, '', ['_', methodName]);

    fprintf('Objective: %g\n', calc_obj(x_bp));

    clear methodName startTimes x_sol timeStats perf
end


if false
    x_init = solve_bp(A,y,k,0);
    fprintf('Taking BP as initialization for GSM\n');
else
    x_init = [];
    fprintf('Using no init for GSM\n');
end


%% Solve with gsm-sense, residual power 2
if true
    methodName = 'gsm';    
    if lambda_base < lambda
        lambdaVals = [lambda_base, lambda];
    elseif lambda_base == lambda
        lambdaVals = lambda;
    else
        error('lambda_base must be lower or equal to lambda');
    end
    
    startTimes = getStartTimes();
    
    params = {'residualPower', 2, 'x_init', x_init, ...
        'profile', 'normal', ...
        'P0_objective_stop_threshold', residual_stop_thresh, ...
        'lambdaVals_manual', lambdaVals, ...
        'propagate_solution_through_lambdas', 'on'};
        %'gamma_test_every_n_iters', 100000030, ...
        %'gamma_test_growth_factors', [10, 2.5, 1.5, 1.25, 1.05, 1.01]  };

    display(params);
    
    [~, sol] = solve_P0_gsm_v4_7(A, y, k, params);
   
    x_sol_untrunc1 = sol.db(end).x_lambda;
    
    %% Remove this
    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'dcp', 'lambdaVals', lambda, 'eta', eta, ...
        'propagate_solutions_through_lambdas', true, 'x_init', x_sol_untrunc1);
    
    x_sol_untrunc1 = dbinfo.x_all{end};
    %% This remove

    nIter1 = sol.db(end).num_ws;
    
    fprintf('\nObjective (take 1): %g\n\n', calc_obj(x_sol_untrunc1));

    params = {'residualPower', 2, 'x_init', x_init, ...
        'profile', 'normal', ...
        'P0_objective_stop_threshold', residual_stop_thresh, ...
        'lambdaVals_manual', lambda};
        %'gamma_test_every_n_iters', 100000030, ...
        %'gamma_test_growth_factors', [10, 2.5, 1.5, 1.25, 1.05, 1.01]  };
        

    display(params);
    
    [~, sol] = solve_P0_gsm_v4_7(A, y, k, params);
       
    x_sol_untrunc2 = sol.db(end).x_lambda;
    nIter2 = sol.db(end).num_ws;
   
    %% Remove this
    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'dcp', 'lambdaVals', lambda, 'eta', eta, ...
        'propagate_solutions_through_lambdas', true, 'x_init', x_sol_untrunc2);
    
    x_sol_untrunc2 = dbinfo.x_all{end};
    %% This remove
    
    fprintf('\nObjective (take 2): %g\n\n', calc_obj(x_sol_untrunc2));
    
    if calc_obj(x_sol_untrunc1) <= calc_obj(x_sol_untrunc2)
        x_sol_untrunc = x_sol_untrunc1;
        nIter = nIter1;
        fprintf('Choosing take 1 solution\n');
    else
        x_sol_untrunc = x_sol_untrunc2;
        nIter = nIter2;
        fprintf('Choosing take 2 solution\n');
    end

    timeStats = getTimeStats(startTimes);    
    
    perf = evaluatePerformance_onelambda(x_sol_untrunc,A, y, k, lambda, lambda_base, lambda_rel, x0, timeStats, nIter);
    
    result = structMerge(result, perf, '', ['_', methodName]);

    clear methodName startTimes x_sol timeStats perf
end


end
