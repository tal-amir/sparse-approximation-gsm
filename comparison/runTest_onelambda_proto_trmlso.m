function result = runTest_onelambda_proto_trmlso(A, x0, y, k, lambda, lambda_base, lambda_rel, residual_stop_thresh, optimization_success_thresh, recovery_success_thresh, noiseStrength, noiseSigma)
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


%% Solve with Trimmed Lasso, DC-Programming
if true
    methodName = 'dcp';    
    
    if lambda_base < lambda
        lambdaVals = [lambda_base, lambda];
    elseif lambda_base == lambda
        lambdaVals = lambda;
    else
        error('lambda_base must be lower or equal to lambda');
    end
    
    %lambdaVals = lambda;
    
    startTimes = getStartTimes();
    
    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'dcp', 'lambdaVals', lambdaVals, 'eta', eta, ...
        'propagate_solutions_through_lambdas', true, 'x_init', []);    
    
    x_sol_untrunc1 = dbinfo.x_all{end};
    nIter1 = dbinfo.nIter_all(end);
    fprintf('Objective: %g\n', calc_obj(x_sol_untrunc1));

    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'dcp', 'lambdaVals', lambda, 'eta', eta, 'x_init', []);    
    
    x_sol_untrunc2 = dbinfo.x_all{end};
    nIter2 = dbinfo.nIter_all(end);
    fprintf('Objective: %g\n', calc_obj(x_sol_untrunc2));
    
    if calc_obj(x_sol_untrunc1) <= calc_obj(x_sol_untrunc2)
        x_sol_untrunc = x_sol_untrunc1;
        nIter = nIter1;
    else
        x_sol_untrunc = x_sol_untrunc2;
        nIter = nIter2;
    end

    timeStats = getTimeStats(startTimes);    
    
    perf = evaluatePerformance_onelambda(x_sol_untrunc,A, y, k, lambda, lambda_base, lambda_rel, x0, timeStats, nIter);
    
    result = structMerge(result, perf, '', ['_', methodName]);

    clear methodName startTimes x_sol timeStats perf
end


%% Solve with Trimmed Lasso, ADMM
if true
    methodName = 'admm';    
    
    if lambda_base < lambda
        lambdaVals = [lambda_base, lambda];
    elseif lambda_base == lambda
        lambdaVals = lambda;
    else
        error('lambda_base must be lower or equal to lambda');
    end
    
    %lambdaVals = lambda;
    
    startTimes = getStartTimes();
    
    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'admm', 'lambdaVals', lambdaVals, 'eta', eta, ...
        'propagate_solutions_through_lambdas', true, 'x_init', [], 'use_alternative_code_admm', false);          
    
    x_sol_untrunc1 = dbinfo.x_all{end};
    nIter1 = dbinfo.nIter_all(end);
    fprintf('Objective: %g\n', calc_obj(x_sol_untrunc1));

    [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, 'method', 'admm', 'lambdaVals', lambda, 'eta', eta, 'x_init', [], 'use_alternative_code_admm', false);   
    
    x_sol_untrunc2 = dbinfo.x_all{end};
    nIter2 = dbinfo.nIter_all(end);
    fprintf('Objective: %g\n', calc_obj(x_sol_untrunc1));
    
    if calc_obj(x_sol_untrunc1) <= calc_obj(x_sol_untrunc2)
        x_sol_untrunc = x_sol_untrunc1;
        nIter = nIter1;
    else
        x_sol_untrunc = x_sol_untrunc2;
        nIter = nIter2;
    end
    
    timeStats = getTimeStats(startTimes);    
    
    perf = evaluatePerformance_onelambda(x_sol_untrunc,A, y, k, lambda, lambda_base, lambda_rel, x0, timeStats, nIter);
    
    result = structMerge(result, perf, '', ['_', methodName]);

    clear methodName startTimes x_sol timeStats perf
end



end
