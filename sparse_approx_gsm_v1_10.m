function [x_sol, sol] = sparse_approx_gsm(A,y,k,varargin)
%function [x_sol, sol] = sparse_approx_gsm(A,y,k,varargin)
% Sparse Approximation by the Generalized SoftMin Penalty
%
% Tal Amir, Ronen Basri, Boaz Nadler
% Weizmann Institute of Science
% tal.amir@weizmann.ac.il
%
% Given an m x n dictionary A, a vector y in R^m and sparsity level k <= m,
% estimates a solution to the sparse approximation problem
%
% (P0)             min ||A*x-y||_2 s.t. ||x||_0 <= k                 
%
% by estimating a solution of the trimmed-lasso penalized problem
%
% (P^p_lambda)     min (1/p)*||A*x-y||^p + lambda * tau_k(x)    
%
% for several values of lambda. Each such problem is approached by the
% generalized soft-min penalty.
%
% Input arguments: 
% A - Matrix of size m x n
% y - Column vector in R^m
% k - Target sparsity level, 1 < k < m
% varargin - name/value pairs of parameters. See Parameters.txt.
% 
% Output arguments:
% x_sol - Estimated solution vector to the sparse apprixmation problem
% sol   - A struct containing information about the solution. See external
%         documentation.
%
% This program requires the Mosek optimization solver.
% https://www.mosek.com/downloads/

version = '1.10';  version_date = '23-Mar-2020';

% =========================================================================

% Detect Mosek
if isempty(which('mosekopt'))
    error('This function requires the Mosek optimization solver. https://www.mosek.com/downloads/');
end

% Generates a parameter struct from the default parameters and the user
% overrides.
params = processParams(varargin);
vl = params.verbosity; % Verbosity level

qprintln(vl,1,'\nSparse Approximation by the Generalized SoftMin Penalty');
qprintln(vl,1,'Tal Amir, Ronen Basri, Boaz Nadler');
qprintln(vl,1,'Weizmann Institute of Science');
qprintln(vl,1,'Version %s, %s', version, version_date);

% Analysis of the dictionary A
A_data = get_A_data(A,k);
optData = makeOptData(A, y, k, A_data, params);

% These are the values of lambda that will be used to solve P0
lambda_vals = get_lambda_vals(params, A_data, norm(y));

% Solve
[x_sol, sol] = solve_P0_main(A, y, k, lambda_vals, optData, params);

sol.version = version;
sol.version_date = version_date;
end


function [x_sol, sol] = solve_P0_main(A, y, k, lambda_vals, optData, params)
vl = params.verbosity; % Verbosity level
db = struct();
tStart = tic;

% Display settings
% ----------------
% Pad with spaces up to this length when displaying numbers
num_disp_length = 11; 

% This character designates that for a certain lambda, a new solution was
% found that is better than all solutions for all previous lambdas.
newMinChar = '*';
normalIterChar = ' ';

% Prefixes to be used when reporting violations and iteration results.
violationStr = '>>>>';
reportStr = '====';

% Constants for later use
[n,d] = size(A);

sol = struct();
sol.params = params;

% Initial reporting
qprintln(vl,1,'');
qprintln(vl,1,'Profile: %s', lower(params.profile));
qprintln(vl,1,'Residual power: %d', params.residualPower);
qprintln(vl,1,'Optimization solver: %s', upper(params.solver));
qprintln(vl,1,'');

qprintln(vl,2,'Running with %d values of lambda: %g --> %g', numel(lambda_vals), min(lambda_vals), max(lambda_vals));
qprintln(vl,2,'');

nn = 8214;
s2 = 8322;
s1 = 8321;
sk = char(8342);

uTruncx = [char(928), char(8342), '(x)'];
uProjx = 'Proj(x)';
uLambda = char(955);
uTau = char(964);

qprintln(vl, 2, 'Legend:');
qprintln(vl, 2, '            %srel = %s / %s_bar', uLambda, uLambda, uLambda);
%qprintln(vl, 2, ' truncResNormRel = %cA*%s - y%c%c / %cy%c%c', nn, uProjx, nn, s2, nn, nn, s2);
qprintln(vl, 2, '  projResNormRel = %cA*%s - y%c%c / %cy%c%c', nn, uProjx, nn, s2, nn, nn, s2);
qprintln(vl, 2, '      resNormRel = %cA*x - y%c%c / %cy%c%c', nn, nn, s2, nn, nn, s2);   
qprintln(vl, 2, '            %srel = k/(n-k) * %cx - %s%c%c / %c%s%c%c', uTau, nn, uTruncx, nn, s1, nn, uTruncx, nn, s1);
qprintln(vl,2,'');
qprintln(vl,2,'%s_bar \t- The threshold on %s that guarantees a k-sparse solution.', uLambda, uLambda);
qprintln(vl,2,'%s \t- k-sparse projection of x.', uTruncx);
qprintln(vl,2,'%s\t- minimizer of %cA*u - y%c%c over the support of %s.', uProjx, nn, nn, s2, uTruncx);

qprintln(vl,1);


%% Initialize violation monitors

violationMax_lambdaSol_solution_inferior_to_projSol = 0;
violationMax_lambdaSol_nonZero_residual_for_small_lambda = 0;

violationMax_gammaSol_solution_inferior_to_init = 0;

violationMax_wSol_solution_inferior_to_init = 0;
violationMax_wSol_solution_inferior_to_projSol = 0;
violationMax_gammaSol_energy_increase_majorizer_decrease = 0;
violationMax_abs_gammaSol_gsm_larger_than_majorizer = 0;

violationMax_wSol_mosek_inferior_to_yalmip = nan;
violationMax_wSol_gurobi_inferior_to_yalmip = nan;
violationMax_wSol_quadprog_inferior_to_yalmip = nan;
violationMax_wSol_gd_inferior_to_yalmip = nan;

if params.debug_mosek
    violationMax_wSol_mosek_inferior_to_yalmip = 0;
end

if params.debug_gurobi
    violationMax_wSol_gurobi_inferior_to_yalmip = 0;
end

if params.debug_quadprog
    violationMax_wSol_quadprog_inferior_to_yalmip = 0;
end

if params.debug_gd
    violationMax_wSol_gd_inferior_to_yalmip = 0;
end

violationMax_wSol_energy_increase_majorizer_decrease = 0;
violationMax_wSol_majorizer_increase = 0;

violationMax_wSol_nonZero_residual_for_small_lambda = 0;
violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;


%% Initialize loop variables

% A function that tells if x is sparse according to the threshold parameter
isSparse = @(x) nonSparsityAbs(x, k) <= params.sparsityThreshAbs_x;

% Any lambda above this values guarantees that the solution is sparse
lambdaSparse   = optData.lambdaSparseGuarantee;

% Any lambda below this value guarantees that the solution x satisfies
% Ax = y
lambdaEquality = optData.lambdaEqualityGuarantee;

% This data structure keeps measurements per lambda value
db = struct();
db(1:numel(lambda_vals)) = db;

tmp = num2cell(lambda_vals);
[db.lambda] = tmp{:};

% Lambda values relative to the lambda threshold that guarantees sparsity
tmp = num2cell([db.lambda] / lambdaSparse);
[db.lambdaRel] = tmp{:};

% Keeps track of the best objective of a projected solution
% || A*proj_k(xSol)-y ||_2 encountered so far.
energy_P0_best = inf;

% We report when lambda crosses the threshold that guarantees sparsity.
% This boolean makes sure we only report once.
already_reported_crossed_large_lambda = false;

% Tells if the (P0) objective has reached below the threshold for stopping,
% specified in the parameters.
reached_objective = false;

% Tells if at least one of the solutions to (P_lambda) obtained so far is
% sparse.
reached_sparsity = false;

% Tells the lambda index of the next iteration
i_next = 1;


%% Main loop: Start solving (P0) with increasing values of lambda
while ~isempty(i_next) && (i_next <= numel(db))
    i = i_next;    
    
    % By default, once we finish with the current lambda, we move on
    % to the next one.
    i_next = i + 1;
       
    lambda_curr = db(i).lambda;    
    
    isEqualityConstrained = (lambda_curr == 0);
    
    % Prepare initialization for solving with current lambda
    if (i == 1)
        x_init_curr = params.x_init;
    elseif ~params.init_x_from_previous_lambda
        x_init_curr = params.x_init;
    elseif ~isempty(db(i-1).x)
        % If the problem was solved for a previous lambda and we need to
        % propagate it as an initialization to the current problem, we do
        % so.
        % Here i > 1 and params.init_x_from_previous_lambda is true, so we
        % ignore params.x_init
        x_init_curr = db(i-1).x;
    else
        x_init_curr = [];
    end
       
    %% Solve problem P_lambda with the current lambda
    tElapsed_curr = tic;
    [xUnproj_curr, dbinfo_lambda] = solve_P_lambda(A, y, k, lambda_curr, x_init_curr, optData, params);
    tElapsed_curr = toc(tElapsed_curr);

    xProj_curr = projectVec(xUnproj_curr, A, y, k);
    
    
    %% Analyze current solution   
    % Energy of current solution in (P_lambda)
    energy_P_lambda_curr = calcEnergy(A, xUnproj_curr, y, k, lambda_curr, params.residualPower);
    
    % Energy of projected solution in (P_lambda).
    % If we're solving the equality-constrained problem, the projected
    % solution can be seen as infeasible.
    if isEqualityConstrained
        energyProj_P_lambda_curr = inf;
    else
        energyProj_P_lambda_curr = calcEnergy(A, xProj_curr, y, k, lambda_curr, params.residualPower);
    end
    
    % Energy of projected solution in (P0)
    energy_P0_curr   = norm(A*xProj_curr-y);
    
    % Relative tail and residual norm of solution
    tailRel_curr     = nonSparsityRel(xUnproj_curr, k);
    resNormRel_curr  = norm(A*xUnproj_curr-y) / norm(y);

    
    %% Update the best iterate if needed
    % In case of two solutions with equal energy, which is the best so far,
    % we prefer the one obtained using the smallest lambda. It is
    % particularly useful to record the smallest lambda which obtained a
    % good solution.
    if (energy_P0_curr < energy_P0_best) || ...
            ((energy_P0_curr <= energy_P0_best) && (lambda_curr < lambda_best))
        
        energy_P0_best = energy_P0_curr;
        xProj_best = xProj_curr;
        lambda_best = lambda_curr;
        
        % This character is for reporting
        iterChar = newMinChar;
    else
        iterChar = normalIterChar;
    end
    
    
    %% Record info about current solution
    db(i).x     = xUnproj_curr;
    db(i).xProj = xProj_curr;
    
    db(i).F_k_lambda       = energy_P_lambda_curr;

    db(i).tau_k      = trimmedLasso(xUnproj_curr, k);
    db(i).tau_k_rel  = (d-k)/k * trimmedLasso(xUnproj_curr, k) / norm(truncVec(xUnproj_curr,k),1);

    xs = sort(abs(xUnproj_curr),'descend');
    db(i).x_abs_ratio = xs(k)/xs(1);
    
    db(i).resNorm      = norm(A*xUnproj_curr-y);
    db(i).resNormRel   = norm(A*xUnproj_curr-y)/norm(y);

    db(i).projResNorm      = norm(A*xProj_curr-y);
    db(i).projResNormRel   = norm(A*xProj_curr-y)/norm(y);
        
    db(i).num_gammas    = dbinfo_lambda.num_gammas;
    db(i).num_ws        = dbinfo_lambda.num_ws;
    
    % Number of iterations it took the optimization method to solve each
    % reweighted subproblem, summed over all such problems.
    db(i).nIter_w_solver = dbinfo_lambda.nIter_w_solver;
    
    db(i).tElapsed = tElapsed_curr;
    
    % Total time spent on solving reweighted subproblems
    db(i).tElapsed_w_solver = dbinfo_lambda.tElapsed_w_solver;
    
    % Total time spent on improving solutions to reweighted subproblems by
    % golden search.
    %db(i).tElapsed_w_goldenSearch = dbinfo_lambda.tElapsed_w_goldenSearch;
    
    
    %% Loop control    
    % Check if the current solution is sparse
    if isSparse(xUnproj_curr)
        reached_sparsity = true;
    end
        
    % Check if we got a sparse solution for enough consecutive values of
    % lambda
    if (i >= params.nLambdas_sparse_x_to_stop) && ...
            all(cellfun(isSparse, {db(i - params.nLambdas_sparse_x_to_stop + 1 : i).x}))

        reached_consecutive_sparsity = true;
    else
        reached_consecutive_sparsity = false;
    end
    
    % If the objective of the truncated x is smaller than the stopping
    % threshold, break.
    if energy_P0_curr <= params.P0_objective_stop_threshold
        reached_objective = true;
    end
    
    if reached_consecutive_sparsity || reached_objective
        % No need to solve for further lambdas
        i_next = [];
    end
    
    if reached_objective
        sol.stop_reason = 'reached_objective_threshold';
        sol.stop_message = '(P0) objective reached below user stopping threshold.';
    elseif reached_consecutive_sparsity
        sol.stop_reason = 'reached_sparsity';
        sol.stop_message = sprintf('Solution to (P_lambda) was sparse for %d consecutive times.', params.nLambdas_sparse_x_to_stop);
    elseif (i_next > numel(db)) && reached_sparsity
        sol.stop_reason = 'finished';
        sol.stop_message = 'Finished solving for all values of lambda. Got a sparse solution.';
    elseif (i_next > numel(db)) && ~reached_sparsity
        sol.stop_reason = 'finished_no_sparse';
        sol.stop_message = 'Finished solving for all values of lambda, but failed to get a sparse solution.';
    end

    
    %% Record violations
    violationCurr_lambdaSol_solution_inferior_to_projSol = (energy_P_lambda_curr-energyProj_P_lambda_curr)/energyProj_P_lambda_curr;
    
    if lambda_curr < lambdaEquality
        violationCurr_lambdaSol_nonZero_residual_for_small_lambda = norm(A*xUnproj_curr-y)/norm(y);
    else
        violationCurr_lambdaSol_nonZero_residual_for_small_lambda = 0;
    end
    
    violationMax_lambdaSol_solution_inferior_to_projSol = max(violationMax_lambdaSol_solution_inferior_to_projSol, violationCurr_lambdaSol_solution_inferior_to_projSol);
    violationMax_lambdaSol_nonZero_residual_for_small_lambda = max(violationMax_lambdaSol_nonZero_residual_for_small_lambda, violationCurr_lambdaSol_nonZero_residual_for_small_lambda);
    
    violationMax_gammaSol_solution_inferior_to_init = max( ...
        violationMax_gammaSol_solution_inferior_to_init, ...
        dbinfo_lambda.violationMax_gammaSol_solution_inferior_to_init);
    
    violationMax_wSol_solution_inferior_to_init = max( ...
        violationMax_wSol_solution_inferior_to_init, ...
        dbinfo_lambda.violationMax_wSol_solution_inferior_to_init);
    
    violationMax_wSol_solution_inferior_to_projSol = max( ...
        violationMax_wSol_solution_inferior_to_projSol, ...
        dbinfo_lambda.violationMax_wSol_solution_inferior_to_projSol);
    
    violationMax_gammaSol_energy_increase_majorizer_decrease = max( ...
        violationMax_gammaSol_energy_increase_majorizer_decrease, ...
        dbinfo_lambda.violationMax_gammaSol_energy_increase_majorizer_decrease);
    
    violationMax_abs_gammaSol_gsm_larger_than_majorizer = max( ...
        violationMax_abs_gammaSol_gsm_larger_than_majorizer,...
        dbinfo_lambda.violationMax_abs_gammaSol_gsm_larger_than_majorizer);
    
    violationMax_wSol_mosek_inferior_to_yalmip = max( ...
        violationMax_wSol_mosek_inferior_to_yalmip, ...
        dbinfo_lambda.violationMax_wSol_mosek_inferior_to_yalmip);
    
    violationMax_wSol_gurobi_inferior_to_yalmip = max( ...
        violationMax_wSol_gurobi_inferior_to_yalmip, ...
        dbinfo_lambda.violationMax_wSol_gurobi_inferior_to_yalmip);
    
    violationMax_wSol_quadprog_inferior_to_yalmip = max( ...
        violationMax_wSol_quadprog_inferior_to_yalmip, ...
        dbinfo_lambda.violationMax_wSol_quadprog_inferior_to_yalmip);

    violationMax_wSol_gd_inferior_to_yalmip = max( ...
        violationMax_wSol_gd_inferior_to_yalmip, ...
        dbinfo_lambda.violationMax_wSol_gd_inferior_to_yalmip);
    
    violationMax_wSol_energy_increase_majorizer_decrease = max( ...
        violationMax_wSol_energy_increase_majorizer_decrease, ...
        dbinfo_lambda.violationMax_wSol_energy_increase_majorizer_decrease);
    
    violationMax_wSol_majorizer_increase = max( ...
        violationMax_wSol_majorizer_increase, ...
        dbinfo_lambda.violationMax_wSol_majorizer_increase);

    violationMax_wSol_nonZero_residual_for_small_lambda = max( ...
        violationMax_wSol_nonZero_residual_for_small_lambda, ...
        dbinfo_lambda.violationMax_wSol_nonZero_residual_for_small_lambda);

    violationMax_abs_wSol_nonSparse_solution_for_large_lambda = max( ...
        violationMax_abs_wSol_nonSparse_solution_for_large_lambda, ...
        dbinfo_lambda.violationMax_abs_wSol_nonSparse_solution_for_large_lambda);

    %% Report current iteration
    % Round the time for reporting
    if tElapsed_curr < 100
        tLambdaRound = round(100*tElapsed_curr)/100;
    elseif tElapsed_curr < 1000
        tLambdaRound = round(10*tElapsed_curr)/10;
    else
        tLambdaRound = round(tElapsed_curr);
    end    
        
    % Report if we crossed the large-lambda threshold
    if (lambda_curr > lambdaSparse) && ~already_reported_crossed_large_lambda
        qprintln(vl,2,'%s%s Lambda has crossed the threshold for sparsity guarantee (lambda = %g)\n', normalIterChar, reportStr, lambdaSparse);
        already_reported_crossed_large_lambda = true;
    end
    
    % Report iteration number
    iterStr = sprintf('%s%.3d', iterChar, i);
    
    % Report current lambda
    if isEqualityConstrained
        lambdaStr = ['Constrained LP', repmat(' ', 1, num_disp_length + numel('lambda = ') - numel(lambdaStr))];
    else
        lambdaStr = sprintf('%srel = %s', uLambda, rpad_num(lambda_curr / lambdaSparse, num_disp_length));
    end
    
    % Report the current iteration's performance
    qprintln(vl, 2, '%s) %s | time = %s | nGammas = %s | %srel = %s | projResNormRel = %s | resNormRel = %s', ...
        iterStr, lambdaStr, rpad_num(tLambdaRound,5), rpad_num(dbinfo_lambda.num_gammas,4), uTau, rpad_num(tailRel_curr,num_disp_length), rpad_num(energy_P0_curr/norm(y),num_disp_length), rpad_num(resNormRel_curr,num_disp_length)); 
 
    
    %% Report violations
    % This funciton reports violations that are measured in relative terms
    reportViolation = @(viol_curr, viol_thresh, viol_name) ...
        qprintln((vl>=3) && (viol_curr > viol_thresh), ...
        '%s%s Violation: %s  [ %g rel ]', normalIterChar, violationStr, viol_name, viol_curr);
    
    % This funciton reports violations that are measured in absolute terms
    reportViolation_abs = @(viol_curr, viol_thresh, viol_name) ...
        qprintln((vl>=3) && (viol_curr > viol_thresh), ...
        '%s%s Violation: %s  [ %g abs ]', normalIterChar, violationStr, viol_name, viol_curr);
    
    nReports = 0;
    
    viol_curr = violationCurr_lambdaSol_solution_inferior_to_projSol;
    viol_thresh =  params.reportThresh_lambdaSol_solution_inferior_to_projSol;
    viol_name = 'lambdaSol => Solution is inferior to truncated solution';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = violationCurr_lambdaSol_nonZero_residual_for_small_lambda;
    viol_thresh =  params.reportThresh_lambdaSol_nonZero_residual_for_small_lambda;
    viol_name = 'lambdaSol => Non-zero range error for small lambda';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_gammaSol_solution_inferior_to_init;
    viol_thresh =  params.reportThresh_gammaSol_solution_inferior_to_init;
    viol_name = ' gammaSol => Solution is inferior to initialization';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_gammaSol_energy_increase_majorizer_decrease;
    viol_thresh =  params.reportThresh_gammaSol_energy_increase_majorizer_decrease;
    viol_name = ' gammaSol => Energy increased while majorizer decreased';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_abs_gammaSol_gsm_larger_than_majorizer;
    viol_thresh =  params.reportThresh_abs_gammaSol_gsm_larger_than_majorizer;
    viol_name = ' gammaSol => GSM is larger than its majorizer';
    nReports = nReports + reportViolation_abs(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_majorizer_increase;
    viol_thresh =  params.reportThresh_wSol_majorizer_increase;
    viol_name = '     wSol => Energy majorizer increase';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_energy_increase_majorizer_decrease;
    viol_thresh =  params.reportThresh_wSol_energy_increase_majorizer_decrease;
    viol_name = '     wSol => Energy increased while majorizer decreased';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_solution_inferior_to_init;
    viol_thresh =  params.reportThresh_wSol_solution_inferior_to_init;
    viol_name = '     wSol => Solution is inferior to initialization';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_solution_inferior_to_projSol;
    viol_thresh =  params.reportThresh_wSol_solution_inferior_to_projSol;
    viol_name = '     wSol => Solution is inferior to truncated solution';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_mosek_inferior_to_yalmip;
    viol_thresh =  params.reportThresh_wSol_mosek_inferior_to_yalmip;
    viol_name = '     wSol => MOSEK solution is inferior to YALMIP';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);

    viol_curr = dbinfo_lambda.violationMax_wSol_gurobi_inferior_to_yalmip;
    viol_thresh =  params.reportThresh_wSol_gurobi_inferior_to_yalmip;
    viol_name = '     wSol => GUROBI solution is inferior to YALMIP';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_quadprog_inferior_to_yalmip;
    viol_thresh =  params.reportThresh_wSol_quadprog_inferior_to_yalmip;
    viol_name = '     wSol => QUADPROG solution is inferior to YALMIP';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);

    viol_curr = dbinfo_lambda.violationMax_wSol_gd_inferior_to_yalmip;
    viol_thresh =  params.reportThresh_wSol_gd_inferior_to_yalmip;
    viol_name = '     wSol => Gradient Descent solution is inferior to YALMIP';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_wSol_nonZero_residual_for_small_lambda;
    viol_thresh =  params.reportThresh_wSol_nonZero_residual_for_small_lambda;
    viol_name = '     wSol => Non-zero range error for small lambda';
    nReports = nReports + reportViolation(viol_curr, viol_thresh, viol_name);
    
    viol_curr = dbinfo_lambda.violationMax_abs_wSol_nonSparse_solution_for_large_lambda;
    viol_thresh =  params.reportThresh_abs_wSol_nonSparse_solution_for_large_lambda;
    viol_name = '     wSol => Non-sparse solution for large lambda';
    nReports = nReports + reportViolation_abs(viol_curr, viol_thresh, viol_name);
    
    % If we reported violations, insert a blank line before the next
    % iteration report
    qprintln(nReports > 0);    
end

tElapsed = toc(tStart);

% Return final solution
x_sol = xProj_best;


%% Final report
qprintln(vl,2);

qprintln(vl,1, '%s Breaking.', sol.stop_message);
qprintln(vl,1,'Total time elapsed: %s', getTimeStr(tElapsed));
qprintln(vl,1);


%% Return solution info
% Prepare and return the debug info 

% Return debug info only for lambda values that we actually ran with
db = db(~cellfun(@isempty, {db.x}));

sol.db = db;

sol.tElapsed = tElapsed;
sol.lambda_sparse_guarantee   = lambdaSparse;
sol.lambda_equality_guarantee = lambdaEquality;

% Report minimal and maximal lambdas that yield the best solution
I_best = find([db.projResNorm] == min([db.projResNorm]));

sol.lambdaBestIdx_min = min(I_best);
sol.lambdaBestIdx_max = max(I_best);

sol.lambdaBest_min = min([db(I_best).lambda]);
sol.lambdaBest_max = max([db(I_best).lambda]);

sol.lambdaRelBest_min = min([db(I_best).lambdaRel]);
sol.lambdaRelBest_max = max([db(I_best).lambdaRel]);


%% Return violations
viol = struct();

viol.violationMax_lambdaSol_solution_inferior_to_projSol  =  violationMax_lambdaSol_solution_inferior_to_projSol;
viol.violationMax_lambdaSol_nonZero_residual_for_small_lambda     =  violationMax_lambdaSol_nonZero_residual_for_small_lambda;

viol.violationMax_gammaSol_solution_inferior_to_init       =  violationMax_gammaSol_solution_inferior_to_init;
viol.violationMax_gammaSol_energy_increase_majorizer_decrease     = violationMax_gammaSol_energy_increase_majorizer_decrease;
viol.violationMax_abs_gammaSol_gsm_larger_than_majorizer          = violationMax_abs_gammaSol_gsm_larger_than_majorizer;

viol.violationMax_wSol_energy_increase_majorizer_decrease = violationMax_wSol_energy_increase_majorizer_decrease;
viol.violationMax_wSol_majorizer_increase = violationMax_wSol_majorizer_increase;
viol.violationMax_wSol_solution_inferior_to_init = violationMax_wSol_solution_inferior_to_init;
viol.violationMax_wSol_solution_inferior_to_projSol = violationMax_wSol_solution_inferior_to_projSol;

viol.violationMax_wSol_nonZero_residual_for_small_lambda = violationMax_wSol_nonZero_residual_for_small_lambda;
viol.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = violationMax_abs_wSol_nonSparse_solution_for_large_lambda;

if strcmp(params.solver, 'debug')
    viol.violationMax_wSol_mosek_inferior_to_yalmip = violationMax_wSol_mosek_inferior_to_yalmip;
    viol.violationMax_wSol_gurobi_inferior_to_yalmip = violationMax_wSol_gurobi_inferior_to_yalmip;
    viol.violationMax_wSol_quadprog_inferior_to_yalmip = violationMax_wSol_quadprog_inferior_to_yalmip;
    viol.violationMax_wSol_gd_inferior_to_yalmip = violationMax_wSol_gd_inferior_to_yalmip;
else
    viol.violationMax_wSol_mosek_inferior_to_yalmip = nan;
    viol.violationMax_wSol_gurobi_inferior_to_yalmip = nan;
    viol.violationMax_wSol_quadprog_inferior_to_yalmip = nan;
    viol.violationMax_wSol_gd_inferior_to_yalmip = nan;
end

sol.viol = viol;
end



function displayLambdaPlot(lambdaVals, A_properties)
%function displayLambdaPlot(lambdaVals, A_properties)
%
% Plots a chart of all the lambda values and the thresholds.

figure;  hold on
plot(1:numel(lambdaVals), lambdaVals,'.b');
plot(1:numel(lambdaVals), A_properties.lambdaSmall * ones(1,numel(lambdaVals)), 'r');
plot(1:numel(lambdaVals), A_properties.maxColNorm * ones(1,numel(lambdaVals)), 'g');
plot(1:numel(lambdaVals), A_properties.lambdaLarge * ones(1,numel(lambdaVals)), 'm');
legend('Lambda values', 'Small-lambda thresh', 'b2k','Large-lambda thresh', 'location','northwest');
title('Lambda values');
drawnow
end


function [x_sol, dbinfo_lambda] = solve_P_lambda(A, y, k, lambda, x_init, optData, params)
%function [x_sol, dbinfo_lambda] = solve_P_lambda(A, y, k, lambda, x_init, optData, params)
%
% This function solves the optimization problem (P_lambda) for a single lambda value.
%
% Input arguments:
%
% A, y, k - Dictionary matrix, signal to represent sparsely and sparsity
%           level respectively.
%
% lambda - A nonnegative penalty parameter. When equals zero, solves the
%          equality-constrained problem min tau_k(x) s.t. A*x=y.
%
% x_init - Initialization. Ignored when set to [].
%
% optData, params - Structs containing the data structures for optimization
%                   and parameters respectively.

% This boolean tells if we seek a solution to the equality-constrained
% problem
%                       min tau_k(x) s.t. A*x==y.
isEqualityConstrained = (lambda == 0);

[n,d] = size(A);
dbinfo_lambda = struct();

%TODO: Parametrize this
% Used for debugging purposes. TODO: Control this by a parameter and return
% in output.
% Tells whether to keepwdp track of several measures throughout iterations,
% including the individual iterates and gamma values.
keep_track = false;

if keep_track
    track = struct;
    track.x = nan(size(A,2),0);
    track.w = nan(size(A,2),0);
    track.gamma = nan(1,0);
    track.num_ws = nan(1,0);
    track.used_init = false(1,0);
end


%% Handle initialization
if ~isempty(x_init) && isEqualityConstrained
    % When we are solving the equality-constrained problem,
    % any initialization must satisfy A*x=y, so we take the closest
    % vector to x which satisfies it.
    x_init = projectToZeroRangeError(x_init, optData);
end

if isempty(x_init) || (maxDiffOfSums(x_init, k) == 0) || ~params.full_homotopy_with_init
    x_curr = [];
    gamma_curr = 0;

    gsm_curr = [];
    energy_gamma_curr = inf;
    w_next = [];
else
    x_curr = x_init;
    gamma_curr = params.gamma_first_max_difference_of_sums / maxDiffOfSums(x_curr,k);
    
    [energy_gamma_curr, gsm_curr, w_next] = calcEnergy_gamma(A, x_curr, y, k, lambda, gamma_curr, params.residualPower);
end


%% Initialize violation monitors

violationMax_gammaSol_solution_inferior_to_init = 0;

violationMax_wSol_solution_inferior_to_init = 0;
violationMax_wSol_solution_inferior_to_projSol = 0;
violationMax_gammaSol_energy_increase_majorizer_decrease = 0;
violationMax_abs_gammaSol_gsm_larger_than_majorizer = 0;

violationMax_wSol_mosek_inferior_to_yalmip = 0;
violationMax_wSol_gurobi_inferior_to_yalmip = 0;
violationMax_wSol_quadprog_inferior_to_yalmip = 0;
violationMax_wSol_gd_inferior_to_yalmip = 0;

violationMax_wSol_energy_increase_majorizer_decrease = 0;
violationMax_wSol_majorizer_increase = 0;

violationMax_wSol_nonZero_residual_for_small_lambda = 0;
violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;

sparse_w_counter = 0;
sparse_x_counter = 0;

num_gammas   = 0;
num_ws       = 0;
nIter_w_solver = 0;
tElapsed_w_solver = 0;
tElapsed_w_goldenSearch = 0;

report_gammas = false;

x_best = [];
energy_lambda_best = inf;

% ksupp_curr keeps track of the k largest-magnitude entries of x.
ksupp_curr = [];

%TODO: Remove this
% ssupp_curr tries to keep track of the "real" support of x at each iteration in
% order to give special treatment to cases where x is s-sparse, where s<k.
% Such cases are prone to numerical instability, and therefore are given
% special treatment by trying to complete the support with the LS-OMP
% algorithm instead of breaking ties by numerical inaccuracies.
ssupp_curr = [];
s_curr = d;

i = 0;
count_to_test = params.gamma_test_counter_init - 1;

while true
    i = i+1;        
    
    %% Get the next iterate    
    x_init_gamma = x_curr;

    w_curr = w_next;
    x_prev = x_curr;
    
    gsm_prev = gsm_curr;
    energy_gamma_prev = energy_gamma_curr;
    
    %% Test if we can safely increase gamma by a larger growth factor
    %% with no effect on the next iterate.
    count_to_test = count_to_test + 1;
    
    test_success = false;
    
    % If it's about time to test a large growth factor
    if (gamma_curr > 0) && (gamma_curr < inf) && (~isempty(x_curr)) && (count_to_test >= params.gamma_test_every_n_iters)
        % Test the candidate growth factors in decreasing magnitudes
        % (we assume that the vector <gamma_test_growth_factors> is decreasing) 
        for i_test = 1:numel(params.gamma_test_growth_factors)
            gamma_test = gamma_curr * params.gamma_test_growth_factors(i_test);

            [x_test, energy_gamma_test, gsm_test, w_next_test, dbinfo_gamma_test] = solve_P_lambda_gamma(A, y, k, lambda, gamma_test, x_init_gamma, optData, params);

            num_gammas = num_gammas + 1;
            num_ws = num_ws + dbinfo_gamma_test.num_ws;
            
            % If the current test gamma passed the test, break
            if (norm(x_test-x_curr, inf) <= params.gamma_test_maximal_x_distance_abs) || ...
                    (norm(x_test-x_curr, inf) <= norm(x_curr,inf)*params.gamma_test_maximal_x_distance_rel)
                test_success = true;
                break
            end
        end
        
        % Regardless whether the test was passed or not, we reset the
        % counter. A test will be performed again in
        % <gamma_test_every_n_iters> iterations.
        %TODO: Decide on whether to reset counter when test succeeded
        if ~test_success
            count_to_test = 0;
        end
    end
    
    if test_success
        % The test has succeeded, so we take the corresponding test gamma
        % to be the current gamma and the resulting solution to be the current iterate.
        gamma_curr = gamma_test;
        x_curr = x_test;
        
        gsm_curr = gsm_test;
        energy_gamma_curr = energy_gamma_test;
        w_next = w_next_test;
        dbinfo_gamma = dbinfo_gamma_test;
    else
        % The test did not succeed, or no test was run, so solve
        % (P_lambda,gamma) as usual with the current gamma.
        [x_curr, energy_gamma_curr, gsm_curr, w_next, dbinfo_gamma] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, x_init_gamma, optData, params);
    end
    
    if gamma_curr == 0
        x_gammazero = x_curr;
    end
    
    % Compare new solution to initialization. If initialization yields a
    % better objective, replace it with current iterate.    
    used_x_init_now = false;
    
    if (gamma_curr > 0) && ~isempty(x_init)
        [energy_gamma_init, gsm_init, w_next_init] = calcEnergy_gamma(A, x_init, y, k, lambda, gamma_curr, params.residualPower);

        if energy_gamma_init < energy_gamma_curr
            used_x_init_now = true;

            % Replace current iterate with x_init
            x_curr = x_init;                        
            
            energy_gamma_curr = energy_gamma_init;
            gsm_curr = gsm_init;
            w_next = w_next_init;
            
            x_init_gamma = x_curr;
            
            % Optimize P_{lambda,gamma} with current gamma, starting from
            % x_init
            [x_curr, energy_gamma_curr, gsm_curr, w_next, dbinfo_gamma] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, x_init_gamma, optData, params);
            num_gammas = num_gammas + 1;
            num_ws = num_ws + dbinfo_gamma.num_ws;
        end
    end
    
    % Check ideal energy of the new x and keep it if it is the best so far
    energy_lambda_curr = calcEnergy(A, x_curr, y, k, lambda, params.residualPower);
    
    xProj_curr = projectVec(x_curr, A, y, k);
    energyProj_lambda_curr = calcEnergy(A, xProj_curr, y, k, lambda, params.residualPower);
    
    % Update best solution to (P_lambda) seen so far, using both x_curr and
    % its k-sparse projection with respect to A,y.
    [x_best, energy_lambda_best] = update_min(x_best, energy_lambda_best, x_curr, energy_lambda_curr);
    [x_best, energy_lambda_best] = update_min(x_best, energy_lambda_best, xProj_curr, energyProj_lambda_curr);

    % Run one more optimization of P_lambda, this time initialized by
    % x_best, which might have been obtained at an earlier gamma.
    if gamma_curr == inf
        [x_temp, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, x_best, optData, params);
        
        % This comparison is ok since here gamma=inf 
        if energy_gamma_temp < energy_lambda_best
            x_curr = x_temp;
            energy_gamma_curr = energy_gamma_temp;
            gsm_curr = gsm_temp;
            w_next = w_temp;
            dbinfo_gamma = dbinfo_gamma_temp;
            
            x_best = x_curr;
            energy_lambda_best = energy_gamma_curr;
        end
    end
    
    % Run one more optimization of P_lambda, this time initialized by
    % x_init, or with a zero vector if no init is suppplied.
    if (gamma_curr == inf) && ~isempty(x_init)
        [x_temp, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, x_init, optData, params);
        
        % This comparison is ok since here gamma=inf 
        if energy_gamma_temp < energy_lambda_best
            x_curr = x_temp;
            energy_gamma_curr = energy_gamma_temp;
            gsm_curr = gsm_temp;
            w_next = w_temp;
            dbinfo_gamma = dbinfo_gamma_temp;
            
            x_best = x_curr;
            energy_lambda_best = energy_gamma_curr;
        end
    elseif (gamma_curr == inf)
        [x_temp, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, zeros(size(x_curr)), optData, params);
        
        % This comparison is ok since here gamma=inf 
        if energy_gamma_temp < energy_lambda_best
            x_curr = x_temp;
            energy_gamma_curr = energy_gamma_temp;
            gsm_curr = gsm_temp;
            w_next = w_temp;
            dbinfo_gamma = dbinfo_gamma_temp;
            
            x_best = x_curr;
            energy_lambda_best = energy_gamma_curr;
        end
    end
    
    
    % Here we try to replace part of the support greedily by
    % Least Squares OMP
    if (gamma_curr == inf) %&& (s_curr < k)
            %((calc_numerical_l0norm(x_curr, params.looseSparsityThreshRel_x) < k) || ...
            %(nnz(abs(x_curr) > params.looseSparsityThreshAbs_x) < k))
        
        if params.use_omp
            t_omp = tic;
            
            for i_omp = 0:(k-1) %s_curr:(k-1)
                x_omp = omp_complete_support(A, y, k, ksupp_curr(1:i_omp));
                
                energy_omp = calcEnergy(A, x_omp, y, k, lambda, params.residualPower);
                
                % Optimizing only when OMP improves the objective slightly
                % degrades performance and yields a negligible speedup.
                if true %energy_omp < energy_gamma_curr
                    [x_temp, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = solve_P_lambda_gamma(A, y, k, lambda, gamma_curr, x_omp, optData, params);
                    
                    % This comparison is ok since here gamma=inf
                    if energy_gamma_temp < energy_lambda_best
                        x_curr = x_temp;
                        energy_gamma_curr = energy_gamma_temp;
                        gsm_curr = gsm_temp;
                        w_next = w_temp;
                        dbinfo_gamma = dbinfo_gamma_temp;
                        
                        x_best = x_curr;
                        energy_lambda_best = energy_gamma_curr;
                    end                    
                end
            end
            
            t_omp = toc(t_omp);
            %fprintf('OMP time: %g\n', t_omp);
        end 
    end
    
    
    %% Collect debug info   
    num_gammas = num_gammas + 1;
    num_ws = num_ws + dbinfo_gamma.num_ws;
    nIter_w_solver = nIter_w_solver + dbinfo_gamma.nIter_w_solver;
    
    tElapsed_w_solver = tElapsed_w_solver + dbinfo_gamma.tElapsed_w_solver;
    tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_gamma.tElapsed_w_goldenSearch;    
    
    if keep_track
        track.x = [track.x, x_curr];
        track.w = [track.w, w_next];
        track.gamma = [track.gamma, gamma_curr];
        track.num_ws = [track.num_ws, dbinfo_gamma.num_ws];
        track.used_init = [track.used_init, used_x_init_now];
    end
    
    
    %% Monitor violations
    violationCurr_gammaSol_solution_inferior_to_init = ...
        (dbinfo_gamma.energy_sol - dbinfo_gamma.energy_init) / dbinfo_gamma.energy_init;
        
    violationMax_gammaSol_solution_inferior_to_init = max( ...
        violationCurr_gammaSol_solution_inferior_to_init, ...
        violationMax_gammaSol_solution_inferior_to_init);
    
    violationMax_wSol_solution_inferior_to_init = max( ...
        violationMax_wSol_solution_inferior_to_init, ...
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_init);
    
    violationMax_wSol_solution_inferior_to_projSol = max( ...
        violationMax_wSol_solution_inferior_to_projSol, ...
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol);
    
    violationMax_gammaSol_energy_increase_majorizer_decrease = max( ...
        violationMax_gammaSol_energy_increase_majorizer_decrease, ...
        dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease);
    
    violationMax_abs_gammaSol_gsm_larger_than_majorizer = max( ...
        violationMax_abs_gammaSol_gsm_larger_than_majorizer,...
        dbinfo_gamma.violationMax_abs_gammaSol_gsm_larger_than_majorizer);
    
    violationMax_wSol_mosek_inferior_to_yalmip = max( ...
        violationMax_wSol_mosek_inferior_to_yalmip, ...
        dbinfo_gamma.violationMax_wSol_mosek_inferior_to_yalmip);
    
    violationMax_wSol_gurobi_inferior_to_yalmip = max( ...
        violationMax_wSol_gurobi_inferior_to_yalmip, ...
        dbinfo_gamma.violationMax_wSol_gurobi_inferior_to_yalmip);
    
    violationMax_wSol_quadprog_inferior_to_yalmip = max( ...
        violationMax_wSol_quadprog_inferior_to_yalmip, ...
        dbinfo_gamma.violationMax_wSol_quadprog_inferior_to_yalmip);

    violationMax_wSol_gd_inferior_to_yalmip = max( ...
        violationMax_wSol_gd_inferior_to_yalmip, ...
        dbinfo_gamma.violationMax_wSol_gd_inferior_to_yalmip);

    violationMax_wSol_energy_increase_majorizer_decrease = max( ...
        violationMax_wSol_energy_increase_majorizer_decrease, ...
        dbinfo_gamma.violationMax_wSol_energy_increase_majorizer_decrease);
    
    violationMax_wSol_majorizer_increase = max( ...
        violationMax_wSol_majorizer_increase, ...
        dbinfo_gamma.violationMax_wSol_majorizer_increase);
    
    violationMax_wSol_nonZero_residual_for_small_lambda = max( ...
        violationMax_wSol_nonZero_residual_for_small_lambda, ...
        dbinfo_gamma.violationMax_wSol_nonZero_residual_for_small_lambda);
    
    violationMax_abs_wSol_nonSparse_solution_for_large_lambda = max( ...
        violationMax_abs_wSol_nonSparse_solution_for_large_lambda, ...
        dbinfo_gamma.violationMax_abs_wSol_nonSparse_solution_for_large_lambda);
    
    
    %% Loop control & gamma update
       
    maxDiff_curr = maxDiffOfSums(x_curr, k);
    
    % If we have just used the final gamma at this iteration, break.
    if (gamma_curr == inf) 
        break
        
    elseif maxDiff_curr == 0
        % If all the entries of x have equal magnitude even after the first
        % iteration, increasing gamma would not make a difference anyway
        % (it would result in the same weight vector w), so we can break.
        % Note: This is an extremely rare and unlikely case.
        gamma_curr = inf;
        
    elseif (gamma_curr == 0) && ~isempty(x_init) && ~params.full_homotopy_with_init
        % Here we need to find the smallest gamma for which x_init is
        % better than x_gammazero
        gamma_min = params.gamma_first_max_difference_of_sums / maxDiff_curr;
        if calcEnergy_gamma(A, x_gammazero, y, k, lambda, gamma_min, params.residualPower) <= ...
                calcEnergy_gamma(A, x_init, y, k, lambda, gamma_min, params.residualPower)
            if calcEnergy_gamma(A, x_gammazero, y, k, lambda, inf, params.residualPower) >= ...
                calcEnergy_gamma(A, x_init, y, k, lambda, inf, params.residualPower)
                
                gamma_max = gamma_min;
                
                while (calcEnergy_gamma(A, x_gammazero, y, k, lambda, gamma_max, params.residualPower) <= ...
                         calcEnergy_gamma(A, x_init, y, k, lambda, gamma_max, params.residualPower)) ...
                         && (gamma_max < 10^30)
                     gamma_max = 10*gamma_max;
                end
                
                while true
                    gamma_test = 0.5*(gamma_min+gamma_max);
                    if (calcEnergy_gamma(A, x_gammazero, y, k, lambda, gamma_test, params.residualPower) <= ...
                         calcEnergy_gamma(A, x_init, y, k, lambda, gamma_test, params.residualPower))
                        gamma_min = gamma_test;
                    else
                        gamma_max = gamma_test;
                    end
                    
                    if gamma_max / gamma_min <= 1.01
                        break;
                    end
                end
                
                gamma_curr = gamma_max;
            else
                gamma_curr = inf;
            end
        else
            gamma_curr = gamma_min;
        end    
        
    elseif gamma_curr == 0
        % If we have just solved with gamma=0, this is the first gamma, so
        % we always keep the solution and continue to the next gamma.                       
        gamma_curr = params.gamma_first_max_difference_of_sums / maxDiff_curr;
        
    else
        %% Here we increase gamma by a set of update rules
        
        % Exponential growth by a constant multiplicative factor.
        % At each iteration, gamma grows by at least this much.        
        gamma_curr = params.gamma_growth_factor * gamma_curr;
        
        % Below is an obsolete feature that makes sure that the maximal
        % difference of k-sums of |x| increases by at least 1.03.
        
        %weightRatioGrowthFactor = chooseByKeyStr(params.profile, 'fast', 0.03, 'normal', 0.02, 'thorough', 0.02, 'crazy', 0.02);
        %gamma_next2 = gamma_curr + log1p(weightRatioGrowthFactor)/maxDiff_curr;
        
        %gamma_curr = max(gamma_next1, gamma_next2);
        %if gamma_next2 > gamma_next1
            %gamma_next2/gamma_next1
        %end
                
        [energy_gamma_curr, gsm_curr, w_next] = calcEnergy_gamma(A, x_curr, y, k, lambda, gamma_curr, params.residualPower);

        w_next_backup = w_next;

        % We try increasing gamma again by the normal growth factor as long
        % as we don't move the weight vector by more than
        % (d-k)/d * <w_diff_thresh_to_keep_increasing_gamma> in l-infinity
        % norm.       

        gamma_more = gamma_curr;
        w_more = w_next;
        energy_gamma_more = energy_gamma_curr;
        gsm_more = gsm_curr;
        
        if true %nonSparsityRel(x_curr,k-1) >= 1e-2 %TODO: Magic number
            t_temp = tic;
            c_temp = 0;
            while true
                c_temp = c_temp + 1;
                
                gamma_more = params.gamma_growth_factor_when_w_doesnt_move * gamma_more;
                
                if gamma_more >= 1e30
                    break;
                end
                
                [energy_gamma_more, gsm_more, w_more] = calcEnergy_gamma(A, x_curr, y, k, lambda, gamma_more, params.residualPower);
                
                if (max(abs(w_more-w_next_backup)) >= (d-k)/d * params.w_diff_thresh_to_keep_increasing_gamma)
                    break
                end
                
                gamma_curr = gamma_more;
                energy_gamma_curr = energy_gamma_more;
                w_next = w_more;
                gsm_curr = gsm_more;
                
                if nonSparsityAbs(w_next, d-k) <= params.sparsityThreshAbs_w
                    % If the weight vector is already (d-k)-sparse, there
                    % is no need to further increase gamma.
                    break
                end
            end
            t_temp = toc(t_temp);
            %fprintf('%g: gamma=%g\ttime=%g\n', c_temp, gamma_more, t_temp); % TODO: Remove this
        end
        
        
        %% Loop control
        
        % If the weight vector is accutely biased towards the k winning
        % entries, there is no reason to run more iterations with larger
        % values of gamma, which would only make it more accute.
        if nonSparsityAbs(w_next, d-k) > params.sparsityThreshAbs_w
            sparse_w_counter = 0;
        else
            sparse_w_counter = sparse_w_counter + 1;

            % If w turned out to be sparse a number of times in a row, jump
            % to the final gamma.
            if sparse_w_counter >= params.nGammas_sparse_w_to_stop
                gamma_curr = inf;
                %fprintf('Setting gamma to inf. Reason: Sparse w\n'); % TODO: Remove
            end
        end
        
        % If the iterate x_curr is k-sparse for a given number of
        % consecutive iterations, and its sparsity pattern remains fixed,
        % we assume that the pattern will remain fixed for all the
        % remaining gamma values, and therefore skip to gamma=inf.
        ksupp_prev = ksupp_curr;
        [~, ksupp_curr] = sort(abs(x_curr), 'descend');
        ksupp_curr = ksupp_curr(1:k);
        
        if (nonSparsityAbs(x_curr, k) <= params.sparsityThreshAbs_x) && ...
            (~isempty(ksupp_prev) && (numel(intersect(ksupp_curr, ksupp_prev)) == numel(ksupp_curr)))
            sparse_x_counter = sparse_x_counter + 1;

            % If w turned out to be sparse a number of times in a row, jump
            % to the final gamma.
            if sparse_x_counter >= params.nGammas_sparse_x_to_stop
                gamma_curr = inf;
                %fprintf('Setting gamma to inf. Reason: Sparse x\n'); % TODO: Remove
            end
        else
            sparse_x_counter = 0;
        end 
        
        % Detect if x_curr is s-sparse for some 0 <= s < k. 
        % TODO: Add explanation 
                         
        s_temp = calc_numerical_l0norm(x_curr, params.looseSparsityThreshRel_x);
        s_temp = min(s_temp, k);
         
        % Backup
        ssupp_prev = ssupp_curr;
        s_prev = s_curr;
         
        % Update
        if s_temp <= s_prev
            s_curr = s_temp;
            ssupp_curr =  ksupp_curr(1:s_curr);
        else
            ssupp_curr =  ksupp_curr(1:s_curr);
            
            if numel(intersect(ssupp_curr,ssupp_prev)) < numel(ssupp_curr)
                s_curr = s_temp;
                ssupp_curr =  ksupp_curr(1:s_curr);
            end
        end
    end
    
    % If we reached the iteration number limit, run a last iteration with
    % gamma = inf.
    if num_gammas >= params.nGammaVals_max - 1
        gamma_curr = inf;        
    end
   
    if report_gammas
        w_sort = sort(w_curr);
        wRatio = w_sort(k)/w_sort(k+1);
        relP0Obj = norm(A*projectVec(x_curr,A,y,k)-y) / norm(y);
        nonSparsity_x_curr = nonSparsityRel(x_curr,k);
        nIterw_curr = dbinfo_gamma.num_ws;
        
        maxWDist = norm(w_next-w_curr,inf);
        x_dist_curr = norm(x_curr-x_prev,1)/norm(x_prev,1);
        
        fprintf('%d: Gamma: %g  nIter w: %d  w-dist: %g  x-dist: %g   x-tail: %g  relP0Obj = %g\n', num_gammas, gamma_curr, nIterw_curr, maxWDist, x_dist_curr, nonSparsity_x_curr, relP0Obj);
        %fprintf('%d: Gamma: %g  nIter w: %d  w-dist: %g  x-dist: %g  w-tail: %g  x-tail: %g  relP0Obj = %g\n', num_gammas, gamma_curr, nIterw_curr, maxWDist, x_dist_curr, nonSparsity_curr, nonSparsity_x_curr, relP0Obj);
    end
end


if report_gammas   
    fprintf('Number of gammas: %d\n', num_gammas);
    %pause
end

if keep_track
    fprintf('Kept track\n');
end

% Return solution
x_sol = x_best;


%% Return violations

dbinfo_lambda.violationMax_gammaSol_solution_inferior_to_init = ...
    violationMax_gammaSol_solution_inferior_to_init;

dbinfo_lambda.violationMax_gammaSol_energy_increase_majorizer_decrease = violationMax_gammaSol_energy_increase_majorizer_decrease;
dbinfo_lambda.violationMax_abs_gammaSol_gsm_larger_than_majorizer = violationMax_abs_gammaSol_gsm_larger_than_majorizer;

dbinfo_lambda.violationMax_wSol_solution_inferior_to_init = violationMax_wSol_solution_inferior_to_init;
dbinfo_lambda.violationMax_wSol_solution_inferior_to_projSol = violationMax_wSol_solution_inferior_to_projSol;

dbinfo_lambda.violationMax_wSol_mosek_inferior_to_yalmip = violationMax_wSol_mosek_inferior_to_yalmip;
dbinfo_lambda.violationMax_wSol_gurobi_inferior_to_yalmip = violationMax_wSol_gurobi_inferior_to_yalmip;
dbinfo_lambda.violationMax_wSol_quadprog_inferior_to_yalmip = violationMax_wSol_quadprog_inferior_to_yalmip;
dbinfo_lambda.violationMax_wSol_gd_inferior_to_yalmip = violationMax_wSol_gd_inferior_to_yalmip;

dbinfo_lambda.violationMax_wSol_energy_increase_majorizer_decrease = violationMax_wSol_energy_increase_majorizer_decrease;
dbinfo_lambda.violationMax_wSol_majorizer_increase = violationMax_wSol_majorizer_increase;

dbinfo_lambda.violationMax_wSol_nonZero_residual_for_small_lambda = violationMax_wSol_nonZero_residual_for_small_lambda;
dbinfo_lambda.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = violationMax_abs_wSol_nonSparse_solution_for_large_lambda;

dbinfo_lambda.num_gammas        = num_gammas;
dbinfo_lambda.num_ws            = num_ws;
dbinfo_lambda.nIter_w_solver     = nIter_w_solver;
dbinfo_lambda.tElapsed_w_solver = tElapsed_w_solver;
dbinfo_lambda.tElapsed_w_goldenSearch = tElapsed_w_goldenSearch;
end


function [x_out, dbinfo_w] = solve_P_lambda_w(A, y, k, lambda, w, x_init, optData, params)
%function [x_out, dbinfo_w] = solve_P_lambda_w(A, y, k, lambda, w, x_init, optData, params)

isEqualityConstrained = lambda == 0;

dbinfo_w = struct();

%% Used in debug mode
energy_w_yalmip = nan;
energy_w_mosek  = nan;
energy_w_gurobi = nan;
energy_w_quadprog    = nan;
energy_w_gd   = nan;

if ~isempty(x_init)
    energy_w_init = calcEnergy_w(A, x_init, y, lambda, params.residualPower, w);
else
    energy_w_init = inf;
end

% Time measurement for Golden Search is NaN when we don't use it
t_gs = nan;

if ~strcmp(params.solver,'mosek')
    error('Only the MOSEK solver is currently supported.');
end

switch(params.solver)
    case 'yalmip'
        tic;
        [x_out, dbinfo_w_method] = solve_P_lambda_w_yalmip(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'mosek'
        tic;
        [x_out, dbinfo_w_method] = solve_P_lambda_w_mosek(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'gurobi'
        tic;
        [x_out, dbinfo_w_method] = solve_P_lambda_w_gurobi(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'quadprog'
        tic;
        [x_out, dbinfo_w_method] = solve_P_lambda_w_quadprog(A, y, lambda, w, x_init, optData, params);
        t = toc;
        
    case 'gd'
        tic;
        [x_out, dbinfo_w_method] = solve_P_lambda_w_gd(A, y, lambda, w, x_init, optData, params);
        t = toc;
        
    case 'debug'        
        tic;
        [x_debug_yalmip, dbinfo_w_yalmip] = solve_P_lambda_w_yalmip(A, y, lambda, w, optData, params);
        t = toc;
        
        if params.use_golden_search
            tic;
            x_debug_yalmip = goldenSearch(A, y, lambda, w, x_init, x_debug_yalmip, params.residualPower);
            t_gs = toc;
        end
        
        % Take the YALMIP solution as the output
        x_out = x_debug_yalmip;
        dbinfo_w_method = dbinfo_w_yalmip;
        
        energy_w_yalmip = calcEnergy_w(A, x_debug_yalmip, y, lambda, params.residualPower, w);
        
        % Compare other methods to YALMIP.
        % Results of methods that are not included in this debug session
        % remain empty vectors.
        
        % Mosek
        if params.debug_mosek
            [x_debug_mosek, dbinfo_w_mosek] = solve_P_lambda_w_mosek(A, y, lambda, w, optData, params);
            
            if params.use_golden_search
                x_debug_mosek = goldenSearch(A, y, lambda, w, x_init, x_debug_mosek, params.residualPower);
            end
            
            energy_w_mosek = calcEnergy_w(A, x_debug_mosek, y, lambda, params.residualPower, w);
        end
        
        % Gurobi
        if params.debug_gurobi
            [x_debug_gurobi, dbinfo_w_gurobi] = solve_P_lambda_w_gurobi(A, y, lambda, w, optData, params);
            
            if params.use_golden_search
                x_debug_gurobi = goldenSearch(A, y, lambda, w, x_init, x_debug_gurobi, params.residualPower);
            end
            
            energy_w_gurobi = calcEnergy_w(A, x_debug_gurobi, y, lambda, params.residualPower, w);
        end
        
        % Quadprog
        if params.debug_quadprog
            [x_debug_quadprog, dbinfo_w_quadprog] = solve_P_lambda_w_quadprog(A, y, lambda, w, x_out, optData, params);
            
            if params.use_golden_search
                x_debug_quadprog = goldenSearch(A, y, lambda, w, x_init, x_debug_quadprog, params.residualPower);
            end
            
            energy_w_quadprog = calcEnergy_w(A, x_debug_quadprog, y, lambda, params.residualPower, w);
        end
        
        % Gradient Descent
        if params.debug_gd
            [x_debug_gd, dbinfo_w_gd] = solve_P_lambda_w_gd(A, y, lambda, w_curr, x_out, optData, params);
            
            if params.use_golden_search
                x_debug_gd = goldenSearch(A, y, lambda, w, x_init, x_debug_gd, params.residualPower);
            end
            
            energy_w_gd = calcEnergy_w(A, x_debug_gd, y, lambda, params.residualPower, w);
        end
                
    otherwise
        error('Invalid optimization method ''%s''', params.method);
end


% Improve solution by golden search
if ~strcmp(params.solver, 'debug') && params.use_golden_search
    tic;
    x_out = goldenSearch(A, y, lambda, w, x_init, x_out, params.residualPower);
    t_gs = toc;
end

dbinfo_w.dbinfo_w_method = dbinfo_w_method;
dbinfo_w.tElapsed_w_solver = t;
dbinfo_w.tElapsed_w_goldenSearch = t_gs;

%% Monitor violations
% Compare solution to initialization

% Compare all methods to yalmip
if strcmp(params.solver, 'debug')
    dbinfo_w.violation_wSol_mosek_inferior_to_yalmip = max(0, (energy_w_mosek - energy_w_yalmip) / energy_w_yalmip);
    dbinfo_w.violation_wSol_gurobi_inferior_to_yalmip = max(0, (energy_w_gurobi - energy_w_yalmip) / energy_w_yalmip);
    dbinfo_w.violation_wSol_quadprog_inferior_to_yalmip = max(0, (energy_w_quadprog - energy_w_yalmip) / energy_w_yalmip);
    dbinfo_w.violation_wSol_gd_inferior_to_yalmip = max(0, (energy_w_gd - energy_w_yalmip) / energy_w_yalmip);
else
    dbinfo_w.violation_wSol_mosek_inferior_to_yalmip = nan;
    dbinfo_w.violation_wSol_gurobi_inferior_to_yalmip = nan;
    dbinfo_w.violation_wSol_quadprog_inferior_to_yalmip = nan;
    dbinfo_w.violation_wSol_gd_inferior_to_yalmip = nan;
end

% Check if the solution energy is higher than the initialization. 
% If so, return the initialization instead.
energy_w_out = calcEnergy_w(A, x_out, y, lambda, params.residualPower, w);

if isempty(x_init)
    dbinfo_w.violation_wSol_solution_inferior_to_init = 0;
else
    dbinfo_w.violation_wSol_solution_inferior_to_init = max(0, ...
        (energy_w_out - energy_w_init) / energy_w_init);
    
    if energy_w_out >= energy_w_init
        x_out = x_init;
    end
end


% Evaluate projected solution
if ~isEqualityConstrained
    xProj = projectVec(x_out, A, y, k);    
    energyProj_w = calcEnergy_w(A, xProj, y, lambda, params.residualPower, w);

    % In case the energy of the projected solution is lower than the energy of
    % returned solution, replace them.
    if energy_w_out > energyProj_w        
        x_out = xProj;
        dbinfo_w.violation_wSol_solution_inferior_to_projSol = ...
            (energy_w_out - energyProj_w) / energyProj_w;
    else
        dbinfo_w.violation_wSol_solution_inferior_to_projSol = 0;
    end
end


end


function [x_out, energy_gamma_out, gsm_out, w_next, dbinfo_gamma] = solve_P_lambda_gamma(A, y, k, lambda, gamma, x_init, optData, params)
%function [x_out, energy_gamma_out, gsm_out, w_next, dbinfo_gamma] = solve_P_lambda_gamma(A, y, k, lambda, gamma, x_init, optData, params)
%
% Seeks a solution to the problem (P_lambda,gamma)
%
% Input arguments
% ---------------
% A, y, k, lambda, gamma - Clear from the context
%
% x_init - Initialization. 
%          Ignored when set to [];
%
% optData, params - Structs containing the optimization data and
%                   parameters.
%
% Output arguments
% ----------------
% x_out   - Final iterate
% energy_gamma_out - Energy of x_out in P_lambda,gamma
% gsm_out - tau_k,gamma(x_out)
% w_next  - w_k,gamma(x_out)
% dbinfo_gamma - A struct containing debug info


dbinfo_gamma = struct();
[n,d] = size(A);

% Tells if we're now running with the final gamma value in the homotopy. 
% In that case, stricter stopping criteria are used, as defined in the parameters.
isFinalGamma = gamma == inf;

% Tells if we seek a solution to the equality-constrained problem
%             min tau_k(x) s.t. A*x==y
isEqualityConstrained = (lambda == 0);

% Number of non-decrease iterations for stopping
if isFinalGamma
    num_non_decreases_for_stopping = params.Plg_num_small_decreases_for_stopping_on_infinite_gamma;
else
    num_non_decreases_for_stopping = params.Plg_num_small_decreases_for_stopping;
end

% This function calculates the energy in (P_lambda,gamma) by the gsm term 
if isEqualityConstrained
    calcEnergyFromGsm = @(x, gsm_term) gsm_term;
else
    calcEnergyFromGsm = @(x, gsm_term) (1/params.residualPower)*norm(A*x-y)^params.residualPower + lambda * gsm_term;
end

%% Handle initialization arguments
if ~exist('init', 'var')
    init = [];
end

if ~isempty(x_init)
    x_curr = x_init;
    [energy_gamma_curr, gsm_curr, w_next] = calcEnergy_gamma(A, x_curr, y, k, lambda, gamma, params.residualPower);
    energy_gamma_init = energy_gamma_curr;
else
    x_curr = [];
    gsm_curr = nan;
    energy_gamma_curr = inf;
    w_next = (d-k)/d * ones(d,1);
    energy_gamma_init = inf;
end


%% Initialize vilation monitors
dbinfo_gamma.violationMax_wSol_solution_inferior_to_init = 0;
dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol = 0;
dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease = 0;
dbinfo_gamma.violationMax_abs_gammaSol_gsm_larger_than_majorizer = 0;

dbinfo_gamma.violationMax_wSol_mosek_inferior_to_yalmip = 0;
dbinfo_gamma.violationMax_wSol_gurobi_inferior_to_yalmip = 0;
dbinfo_gamma.violationMax_wSol_quadprog_inferior_to_yalmip = 0;
dbinfo_gamma.violationMax_wSol_gd_inferior_to_yalmip = 0;

dbinfo_gamma.violationMax_wSol_energy_increase_majorizer_decrease = 0;
dbinfo_gamma.violationMax_wSol_majorizer_increase = 0;

dbinfo_gamma.violationMax_wSol_nonZero_residual_for_small_lambda = 0;
dbinfo_gamma.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;


%% Initialize loop
non_decrease_counter = 0;

% These counters keep track, respectively, of how many weight vectors were used,
% and how many internal iterations were used for optimizing the subproblems
% for all w's in total.
num_ws = 0;
nIter_w_solver = 0;

tElapsed_w_solver = 0;
tElapsed_w_goldenSearch = 0;

for i=1:params.Plg_max_num_mm_iters
    num_ws = num_ws + 1;    
    
    %% Solve current reweighted problem
    w_curr = w_next;
    x_prev = x_curr;
    [x_curr, dbinfo_w] = solve_P_lambda_w(A, y, k, lambda, w_curr, x_curr, optData, params);    
        
    %% Calculate energy and reweight
    gsm_prev = gsm_curr;    
    energy_gamma_prev = energy_gamma_curr;        
    [energy_gamma_curr, gsm_curr, w_next] = calcEnergy_gamma(A, x_curr, y, k, lambda, gamma, params.residualPower);

    if ~isempty(x_prev)
        energy_w_prev = calcEnergy_w(A, x_prev, y, lambda, params.residualPower, w_curr);
    else
        energy_w_prev = inf;
    end
        
    energy_w_curr = calcEnergy_w(A, x_curr, y, lambda, params.residualPower, w_curr);

    % Linear majorizer of gsm with respect to x_prev
    if ~isempty(x_prev)
        gsm_curr_majorizer = gsm_prev + dot(abs(x_curr)-abs(x_prev), w_curr);
    else
        gsm_curr_majorizer = nan;
    end
    
          

    %% Debugging and violation monitoring
    % Keep track of running times
    tElapsed_w_solver = tElapsed_w_solver + dbinfo_w.tElapsed_w_solver;
    tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_w.tElapsed_w_goldenSearch;
    nIter_w_solver = nIter_w_solver + dbinfo_w.dbinfo_w_method.nIter_w_solver;
    
    
    %% Record violations   
    viol_curr = dbinfo_w.violation_wSol_solution_inferior_to_init;
    dbinfo_gamma.violationMax_wSol_solution_inferior_to_init = max(...
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_init, ...
        viol_curr);
        
    viol_curr = dbinfo_w.violation_wSol_solution_inferior_to_projSol;
    dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol = max(...
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol, ...
        viol_curr);
    
    viol_curr = (energy_gamma_curr - energy_gamma_prev) / energy_gamma_prev * ...
        subplus(sign(energy_w_prev - energy_w_curr));
    dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease = max(...
        dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease, ...
        viol_curr);
    
    viol_curr = (gsm_curr - gsm_curr_majorizer);
    dbinfo_gamma.violationMax_abs_gammaSol_gsm_larger_than_majorizer = max(...
        dbinfo_gamma.violationMax_abs_gammaSol_gsm_larger_than_majorizer, ...
        viol_curr);
     
    if strcmp(params.solver, 'debug')
        viol_curr = dbinfo_w.violation_wSol_mosek_inferior_to_yalmip;
        dbinfo_gamma.violationMax_wSol_mosek_inferior_to_yalmip = max(...
            dbinfo_gamma.violationMax_wSol_mosek_inferior_to_yalmip, ...
            viol_curr);
        
        viol_curr = dbinfo_w.violation_wSol_gurobi_inferior_to_yalmip;
        dbinfo_gamma.violationMax_wSol_gurobi_inferior_to_yalmip = max(...
            dbinfo_gamma.violationMax_wSol_gurobi_inferior_to_yalmip, ...
            viol_curr);

        viol_curr = dbinfo_w.violation_wSol_quadprog_inferior_to_yalmip;
        dbinfo_gamma.violationMax_wSol_quadprog_inferior_to_yalmip = max(...
            dbinfo_gamma.violationMax_wSol_quadprog_inferior_to_yalmip, ...
            viol_curr);

        viol_curr = dbinfo_w.violation_wSol_gd_inferior_to_yalmip;
        dbinfo_gamma.violationMax_wSol_gd_inferior_to_yalmip = max(...
            dbinfo_gamma.violationMax_wSol_gd_inferior_to_yalmip, ...
            viol_curr); 
    end
    
    
    viol_curr = dbinfo_w.dbinfo_w_method.violationMax_wSol_energy_increase_majorizer_decrease;
    dbinfo_gamma.violationMax_wSol_energy_increase_majorizer_decrease = max(...
        dbinfo_gamma.violationMax_wSol_energy_increase_majorizer_decrease, ...
        viol_curr);
    
    viol_curr = dbinfo_w.dbinfo_w_method.violationMax_wSol_majorizer_increase;
    dbinfo_gamma.violationMax_wSol_majorizer_increase = max(...
        dbinfo_gamma.violationMax_wSol_majorizer_increase, ...
        viol_curr);

    % Verify that the guarantee Ax==y for appropriate lambda is satisfied.
    if (lambda < optData.lambdaEqualityGuarantee)
        viol_curr = norm(A*x_curr-y)/norm(y);
    else
        viol_curr = 0;
    end
    
    dbinfo_gamma.violationMax_wSol_nonZero_residual_for_small_lambda = max(...
        dbinfo_gamma.violationMax_wSol_nonZero_residual_for_small_lambda, ...
        viol_curr);
    
    % Verify that the guarantee that x is k-sparse for appropriate lambda
    % is satisfied.
    if (lambda > optData.lambdaSparseGuarantee)
        % This is the average absolute value of the (d-k)-smallest
        % magnitude entries of x.
        viol_curr = norm(x_curr - truncVec(x_curr,k), 1) / (d-k);
    else
        viol_curr = 0;
    end
    
    dbinfo_gamma.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = max(...
        dbinfo_gamma.violationMax_abs_wSol_nonSparse_solution_for_large_lambda, ...
        viol_curr);


    %% Loop control 
    if (gamma == inf) && (all(w_next == w_curr))
        break
    end
    
    if (energy_w_prev - energy_w_curr) <= energy_w_prev * params.Plg_minimal_decrease_immediate
        break
    end
    
    if (energy_w_prev - energy_w_curr) <= energy_w_prev * params.Plg_minimal_decrease
        non_decrease_counter = non_decrease_counter + 1;
        
        if non_decrease_counter >= num_non_decreases_for_stopping
            break
        end
    else
        non_decrease_counter = 0;
    end
end


x_out = x_curr;
energy_gamma_out = energy_gamma_curr;
gsm_out = gsm_curr;

[~, gsm_test, ~] = calcEnergy_gamma(A, x_out, y, k, lambda, gamma, params.residualPower);

dbinfo_gamma.energy_init = energy_gamma_init;
dbinfo_gamma.energy_sol  = energy_gamma_curr;

dbinfo_gamma.num_ws = num_ws;
dbinfo_gamma.nIter_w_solver = nIter_w_solver;

dbinfo_gamma.tElapsed_w_goldenSearch = tElapsed_w_goldenSearch;
dbinfo_gamma.tElapsed_w_solver = tElapsed_w_solver;
end


function [x_out, dbinfo_w] = solve_P_lambda_w_gd(A, y, lambda, w, x_init, optData, params)
% A smooth surrogate for the lp-norm
%g_p_eps = @(x,p,epsilon) (p/2) * max(abs(x),epsilon).^(p-2).*x.^2 + (1-p/2) * max(abs(x),epsilon).^p;
g_1_eps = @(x,epsilon) 0.5* ( max(abs(x),epsilon).^(-1).*x.^2 + 0.5 * max(abs(x),epsilon) );

% Energy with respect to the given w
%energy_w_eps = @(x,w,epsilon) g_p_eps(norm(A*x-y),1,epsilon) + lambda*dot(w, g_p_eps(x,1,epsilon));
energy_w_eps = @(x,w,epsilon) g_1_eps(norm(A*x-y),epsilon) + lambda*dot(w, g_1_eps(x,epsilon));

x_curr = x_init;

nIter = 0;
nEpsVals = numel(params.wSol_gd_epsVals);

dbinfo_w = struct();
dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = 0;
dbinfo_w.violationMax_wSol_majorizer_increase = 0;
minEnergy_w_eps_curr = inf;

for i1=1:nEpsVals
    % Optimize using the current epsilon
    eps_curr = params.wSol_gd_epsVals(i1);
    energy_w_eps_curr = energy_w_eps(x_curr,w,eps_curr);
    minEnergy_w_eps_curr = min(energy_w_eps_curr, minEnergy_w_eps_curr);
    
    num_non_decreases_for_convergence = params.wSol_gd_num_non_decreases_for_convergence(i1);
    
    % Optimize using Gradient Descent with respect to the given vector w
    i2 = 0;
    non_decrease_counter = 0;
    
    while i2 < params.wSol_gd_nMaxItersPerEps(i1)
        i2 = i2 + 1;
        nIter = nIter + 1;
        
        x_prev = x_curr;
        
        w_curr = lambda * w ./ max(abs(x_curr), eps_curr);
        err_coeff_curr = 1 ./ max(norm(A*x_curr-y), eps_curr);
        
        max_coeff = max(max(w_curr), err_coeff_curr);
        
        w_curr = w_curr ./ max_coeff;
        err_coeff_curr = err_coeff_curr ./ max_coeff;
        
        if (params.exact_gd_solve_every_n_iter ~= 0) && ...
                (mod(i2-1, params.exact_gd_solve_every_n_iter) == 0)
            x_curr1 = solve_weighted_least_squares_direct(err_coeff_curr, w_curr, optData);            
            x_curr2 = solve_weighted_least_squares_lsqlin(err_coeff_curr, w_curr, x_curr, optData);
            
            x_curr = x_curr2;
            
            % Compare the solution obtained with the backslash operator
            % with the lsqlin / mosek solution.
            f1 = err_coeff_curr*norm(A*x_curr1-y)^2 + dot(w_curr,x_curr1.^2);
            f2 = err_coeff_curr*norm(A*x_curr2-y)^2 + dot(w_curr,x_curr2.^2);
            
            regretRel = (f1-f2)/f2;
            if (regretRel > 1e-7)
                fprintf('Regret: %g\n', regretRel);
            end
        else
            % Note: This is not a good option. Try to avoid using this.
            fprintf('Using the bad Gradient Descent method\n');
            x_curr = solve_weighted_least_squares_ssf(lambda_curr, w_curr, x_curr, optData);
        end
                
        energy_majorizer_prev = err_coeff_curr*norm(A*x_prev-y)^2 + dot(w_curr,x_prev.^2);
        energy_majorizer_curr = err_coeff_curr*norm(A*x_curr-y)^2 + dot(w_curr,x_curr.^2);
        
        %This replaces the solution with the truncated solution in case the
        %latter is better.
        if true
            xTrunc_curr = truncVec(x_curr,optData.k);
            truncEnergy_majorizer_curr = err_coeff_curr*norm(A*xTrunc_curr-y)^2 + dot(w_curr,xTrunc_curr.^2);
            
            if truncEnergy_majorizer_curr < energy_majorizer_curr
                energy_majorizer_curr = truncEnergy_majorizer_curr;
                x_curr = xTrunc_curr;
            end
        end
        
        energy_w_eps_prev = energy_w_eps_curr;
        energy_w_eps_curr = energy_w_eps(x_curr, w, eps_curr);

        minEnergy_w_eps_prev = minEnergy_w_eps_curr;
        minEnergy_w_eps_curr = min(minEnergy_w_eps_curr, energy_w_eps_curr);
        
        if energy_majorizer_curr <= energy_majorizer_prev
            viol_curr = (energy_w_eps_curr - energy_w_eps_prev) / energy_w_eps_prev;
        else
            viol_curr = 0;
        end        
        dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = max(...
            dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease, ...
            viol_curr);
        
        % If the analytical solution for the global minimizer of the
        % majorizer doesn't work (usually due to ill-conditioning), skip to
        % the next epsilon value.
        if energy_majorizer_curr > energy_majorizer_prev
            viol_curr = (energy_majorizer_curr - energy_majorizer_prev) / energy_majorizer_prev;
            dbinfo_w.violationMax_wSol_majorizer_increase = max(...
                dbinfo_w.violationMax_wSol_majorizer_increase, ...
                viol_curr);
            
            x_curr = x_prev;
            break;
        end
        
        % If the minimal entry is already larger than epsilon, there is no
        % point in continuing iterating with the same epsilon
        if min(abs(x_curr)) >= eps_curr
            break
        end
        
        %% Loop control
                
        if (minEnergy_w_eps_curr < minEnergy_w_eps_prev * params.wSol_gd_decreaseThresh)
            non_decrease_counter = 0;
        else
            non_decrease_counter = non_decrease_counter + 1;
            
            if non_decrease_counter >= num_non_decreases_for_convergence
                break
            end
        end
    end
end

x_out = x_curr;

dbinfo_w.nIter_w_solver = nIter;
end


function x_sol = solve_weighted_least_squares_direct(err_coeff, w, optData)
%function x_sol = solve_weighted_least_squares_direct(err_coeff, w, optData)
%
% Returns a direct solution to the optimization problem:
% min err_coeff*norm(A*x-y)^2 + sum(w.*(x.^2))

%tic;

% Option 1: Sparse system. Best option
x_sol = full( [sqrt(err_coeff)*optData.A_sparse; diag(sparse(sqrt(w)))] \ (sqrt(err_coeff)*optData.y_aug) );

% Note: Important! All the other methods lested below are not suitable to
%       work with the err_coeff, but with the older formulation that uses lambda.
% Option 1.5: Full system
%x_sol = [optData.A; diag(sqrt(lambda*w))] \ optData.y_aug;

% Option 2: Change of variables that doesn't work well numerically
%[n,d] = size(optData.A);

%if rank( full([optData.A_sparse./repmat(sqrt(w)', [n,1]); sqrt(lambda)*speye(d) ]))  <= 113
  %  fprintf('low rank\n');
%end

%z_sol = full( [optData.A_sparse./repmat(sqrt(w)', [n,1]); sqrt(lambda)*speye(d) ] \ optData.y_aug );
%x_sol = z_sol ./ sqrt(w); 

%d = sqrt(max(w,100));
%z_sol = full( [optData.A_sparse./repmat(d', [n,1]); diag(sparse(sqrt(lambda*d.^2)))] \ optData.y_aug );
%x_sol2 = z_sol ./ d;

%f1 = norm(optData.A*x_sol-optData.y)^2 + lambda*dot(w, x_sol.^2);
%f2 = norm(optData.A*x_sol2-optData.y)^2 + lambda*dot(w, x_sol2.^2);

%fprintf('Gain: %g\n', (f1-f2)/f2);
%fprintf('%g\n', norm(optData.A*x_sol-optData.y)^2 + lambda*dot(w, x_sol.^2));

% Option 2: Full system
%x_sol = [A; diag(sqrt(lambda*w))] \ optData.y_aug;

% Option 3: Pseudo-inverse (Returns a minimum-norm solution)
%x_sol = pinv([A; diag(sqrt(lambda*w))])* optData.y_aug;

% Refine the solution using mosek lsqlin solver
%x_sol = lsqlin(full([optData.A_sparse; diag(sparse(sqrt(lambda*w)))]), optData.y_aug, [], [], [], [], [], [], x_sol, ...
%    mskoptimset('Algorithm','active-set'));

%fprintf('%g\n',toc);
end


function x_sol = solve_weighted_least_squares_lsqlin(err_coeff, w, x_curr, optData)
%function x_sol = solve_weighted_least_squares_lsqlin(A_coeff, w, x_curr, optData)
%
% Returns a solution to the optimization problem:
% min A_coeff * norm(A*x-y)^2 + sum(w.*(x.^2))

use_matlab_lsqlin = true; %TODO: Fix this also for MOSEK

if use_matlab_lsqlin        
    [x_sol, ~, ~, exitflag, output] = lsqlin([sqrt(err_coeff)*optData.A_sparse; diag(sparse(sqrt(w)))], sqrt(err_coeff)*optData.y_aug, [], [], [], [], [], [], x_curr, ...
        optData.lsqlin_options);

%     q = optData.AtA + lambda*diag(w);
%     c = -optData.Aty';
%     
%     options = optimoptions('quadprog');
%     options.Algorithm = 'trust-region-reflective';
%     options.Display = 'iter';
%     
%     [x_sol, ~, ~, exitflag, output] = quadprog(q, c, [], [], [], [], [], [], x_curr, options);
    
    x_sol = full(x_sol);
        
    if (exitflag ~= 1)
        error('LSQLIN problem: %s\n', output.message);
    end
    
else
    q = optData.AtA + lambda*diag(w);
    c = -optData.Aty';
    a = sparse(optData.d, optData.d);
    
    mskParam = mskoptimset('MSK_IPAR_NUM_THREADS', params.maxNumThreads);
    
    res = mskqpopt(q, c, a, [], [], [], [], mskParam, 'minimize echo(0)');
    
    if res.rcode ~= 0
        error('MOSEK error. MSKQPOPT code string: %s', res.rcodestr);        
    end
    
    x_sol = res.sol.itr.xx;
end    


% Compare matlab result to mosek result
% x_mat = full(lsqlin([optData.A_sparse; diag(sparse(sqrt(lambda*w)))], optData.y_aug, [], [], [], [], [], [], x_curr, ...
%         optimoptions(@lsqlin, 'Algorithm','active-set')));
% 
% res_msk = norm(optData.A*x_sol)^2 + lambda*dot(w,x_sol.^2);
% res_mat = norm(optData.A*x_mat)^2 + lambda*dot(w,x_mat.^2);
% loss = max(0,(res_mat-res_msk)/res_msk);
% if loss <= 1e-6
%     loss = 0;
% end
% fprintf('Matlab loss: %g\n', loss);

end



function x_sol = solve_weighted_least_squares_ssf(lambda, w, x_curr, optData)
%function x_sol = solve_weighted_least_squares_ssf(lambda, w, x_curr, optData)
%
% Uses SSF to solve the optimization problem:
% min norm(A*x-y)^2 + lambda * sum(w.*(x.^2))
%

distDecayThresh = 0.99;
num_non_decreases_for_convergence = 4;

dist_curr = inf;
minDist_curr = inf;
energy_curr = inf;
minEnergy_curr = inf;

%tic 

x_prev = x_curr;

c_ssf = 0.5*optData.gamma_max^2;

for iIter=1:300
    %v = x + A'*(y-A*x)/c;
    %x = v ./ (1+lambda*w/c);
    v = c_ssf * x_curr + (optData.Aty - optData.AtA * x_curr);
    x_curr = v ./ (c_ssf+lambda*w);
    
    dist_prev = dist_curr;
    minDist_prev = minDist_curr;
    
    dist_curr = norm(x_curr-x_prev);
    minDist_curr = min(minDist_curr, dist_curr);
    
    energy_prev = energy_curr;
    minEnergy_prev = minEnergy_curr;
    
    energy_curr = norm(optData.A * x_curr - optData.y)^2 + lambda*dot(w, x_curr.^2);
    minEnergy_curr = min(energy_curr, minEnergy_curr);
    
    if (minEnergy_curr < minEnergy_prev * distDecayThresh) || (dist_curr > 1e-5 * norm(x_curr))
        non_decrease_counter = 0;
    else
        non_decrease_counter = non_decrease_counter + 1;
        
        if non_decrease_counter >= num_non_decreases_for_convergence
            break
        end
    end
end

x_sol = x_curr;

%t = toc;
%fprintf('Time: %g\n', t);
%fprintf('nIter: %d\n', iIter);

% This compares the result to the direct method
%energy_sol = norm(optData.A * x_sol - optData.y)^2 + lambda*dot(w, x_sol.^2);
%x_direct = full( [optData.A_sparse; diag(sparse(sqrt(lambda*w)))] \ optData.y_aug );
%energy_direct = norm(optData.A * x_anal - optData.y)^2 + lambda*dot(w, x_anal.^2);
%fprintf('Relative energy: %g\n', energy_sol/energy_direct);

end


function [x, dbinfo_w] = solve_P_lambda_w_yalmip(A, y, lambda, w, optData, params)

isEqualityConstrained = (lambda == 0);

if ~isEqualityConstrained
    %% Solve the SOCP problem min (1/p)*||A*x-y||^p + lambda*dot(w,abs(x))
    objective = optData.socp_yalmipData.resNorm_opt + dot(lambda*w, optData.socp_yalmipData.xBound_opt);
    sol = optimize(optData.socp_yalmipData.constraints, objective, optData.socp_yalmipData.settings);
    
    if sol.problem ~= 0
        if sol.problem == 4
            if vl >= 4
                warning('YALMIP error. Info: %s', sol.info);
            end
        else
            error('YALMIP error. Info: %s', sol.info);
        end
    end
    
    x = double(optData.socp_yalmipData.x_opt);
    
elseif isEqualityConstrained
    %% Solve the equality-constrained problem: min dot(w,abs(x)) s.t. A*x==y
    objective = dot(w, optData.constrlp_yalmipData.xBound_opt);
    sol = optimize(optData.constrlp_yalmipData.constraints, objective, optData.constrlp_yalmipData.settings);
    
    if sol.problem ~= 0
        if sol.problem == 4
            if vl >= 4
                warning('YALMIP error. Info: %s', sol.info);
            end
        else
            error('YALMIP error. Info: %s', sol.info);
        end
    end
    
    x = double(optData.constrlp_yalmipData.x_opt);
end

dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;
dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end



function [x, dbinfo_w] = solve_P_lambda_w_mosek(A, y, lambda, w, optData, params)
[n,d] = size(A);
vl = params.verbosity;

w = w(:);

isEqualityConstrained = (lambda == 0);

if ~isEqualityConstrained
    if params.residualPower == 1
        %Variable: [x; abs(x); residual; norm(residual)]
        optData.socp_mosekProb.c = [zeros(d,1); lambda*w; zeros(n,1); 1];
        [~,res] = mosekopt('minimize echo(0)', optData.socp_mosekProb, optData.socp_mosekParams);
        
        if res.rcode ~= 0
            if strcmp(res.rcodestr, 'MSK_RES_TRM_STALL')
                if vl >= 4
                    warning('MOSEK error. Code string: %s', res.rcodestr);
                end                
            else
                error('MOSEK error. Code string: %s', res.rcodestr);
            end
        elseif ~strcmp(res.sol.itr.solsta, 'OPTIMAL')
            if vl >= 4
                warning('MOSEK: res.sol.itr.solsta = ''%s''', res.sol.itr.solsta);
            end
        else
            %fprintf('Good!\n');
        end
        
        x = res.sol.itr.xx(1:d);
        
    elseif params.residualPower == 2
        if optData.mosek_socp_formulation == 1
            %Variable: [x; abs(x); residual]
            optData.socp_mosekProb.c = [zeros(d,1); lambda*w; zeros(n,1)];
            [~,res] = mosekopt('minimize echo(0)', optData.socp_mosekProb, optData.socp_mosekParams);
            
            if res.rcode ~= 0
                if strcmp(res.rcodestr, 'MSK_RES_TRM_STALL')
                    if vl >= 4
                        warning('MOSEK error. Code string: %s', res.rcodestr);
                    end
                else
                    error('MOSEK error. Code string: %s', res.rcodestr);
                end
            elseif ~strcmp(res.sol.itr.solsta, 'OPTIMAL')
                if vl >= 4
                    warning('MOSEK: res.sol.itr.solsta = ''%s''', res.sol.itr.solsta);
                end
            else
                %fprintf('Good!\n');
            end
            
            x = res.sol.itr.xx(1:d);
        elseif optData.mosek_socp_formulation == 2
            %Variable: [x; abs(x); residual; 0.5*||residual||^2; 1]
            optData.socp_mosekProb.c = [zeros(d,1); lambda*w; zeros(n,1); 1; 0];
            [~,res] = mosekopt('minimize echo(0)', optData.socp_mosekProb, optData.socp_mosekParams);
            
            if res.rcode ~= 0
                if strcmp(res.rcodestr, 'MSK_RES_TRM_STALL')
                    if vl >= 4
                        warning('MOSEK error. Code string: %s', res.rcodestr);
                    end
                else
                    error('MOSEK error. Code string: %s', res.rcodestr);
                end
            elseif ~strcmp(res.sol.itr.solsta, 'OPTIMAL')
                if vl >= 4
                    warning('MOSEK: res.sol.itr.solsta = ''%s''', res.sol.itr.solsta);
                end
            else
                %fprintf('Good!\n');
            end
            
            x = res.sol.itr.xx(1:d);
        end
    end
elseif isEqualityConstrained
    %Variable: [x+; x-]
    optData.constrlp_mosekProb.c = [w; w];
    [~,res] = mosekopt('minimize echo(0)', optData.constrlp_mosekProb, optData.constrlp_mosekParams);
    
        if res.rcode ~= 0
            if strcmp(res.rcodestr, 'MSK_RES_TRM_STALL')
                if vl >= 4
                    warning('MOSEK error. Code string: %s', res.rcodestr);
                end                
            else
                error('MOSEK error. Code string: %s', res.rcodestr);
            end
        elseif ~strcmp(res.sol.bas.solsta, 'OPTIMAL')
            if vl >= 4
                warning('MOSEK: res.sol.itr.solsta = ''%s''', res.sol.itr.solsta);
            end
        else
            %fprintf('Good!\n');
        end
    
    %x = res.sol.itr.xx(1:d) - res.sol.itr.xx(d+1:2*d);
    x = res.sol.bas.xx(1:d) - res.sol.bas.xx(d+1:2*d);
end

% TODO: This is a test
x2 = projectToZeroRangeError(x, optData);

energy_x = calcEnergy_w(A, x, y, lambda, params.residualPower, w);
energy_x2 = calcEnergy_w(A, x2, y, lambda, params.residualPower, w);

if energy_x2 <= energy_x
    x = x2;
end

dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;
dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end




function [x, dbinfo_w] = solve_P_lambda_w_gurobi(A, y, lambda, w, optData, params)
[n,d] = size(A);
vl = params.verbosity;

isEqualityConstrained = (lambda == 0);

if ~isEqualityConstrained
    if params.residualPower == 1
        %Variable: [x; abs(x); residual; norm(residual)]
        optData.socp_gurobiProb.obj = [zeros(d,1); lambda*w(:); zeros(n,1); 1];

        result = gurobi(optData.socp_gurobiProb, optData.socp_gurobiParams);
        x = res.sol.itr.xx(1:d);
        
    elseif params.residualPower == 2
        %Variable: [x; abs(x); residual; norm(residual)]
        optData.socp_gurobiProb.obj = [zeros(d,1); lambda*w(:); zeros(n+1,1)];

        result = gurobi(optData.socp_gurobiProb, optData.socp_gurobiParams);
        
        if ~strcmp(result.status, 'OPTIMAL')
        end
        
        x = result.x(1:d);
    end
elseif isEqualityConstrained
    %Variable: [x+; x-]
    x = res.sol.itr.xx(1:d) - res.sol.itr.xx(d+1:2*d);
end
        
dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;
dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end


function [x, dbinfo_w] = solve_P_lambda_w_quadprog(A, y, lambda, w, x0, optData, params)
[n,d] = size(A);
vl = params.verbosity;

isEqualityConstrained = (lambda == 0);

if ~isEqualityConstrained
    if params.residualPower == 1
        error('This should not happen');
        
    elseif params.residualPower == 2        
        fvec = [zeros(d,1); lambda*w; zeros(n,1)];
        
        if isempty(x0)
            % Keep x0 empty
        elseif optData.quadprogData.formulation == 1
            % Option 1: Variable = [x; abs(x); residual]
            x0 = [x0; abs(x0); A*x0-y];
        elseif optData.quadprogData.formulation == 2
            % Option 2: Variable = [abs(x)-x; abs(x); residual]
            x0 = [2*subplus(-x0); abs(x0); A*x0-y];
        end        
        
        [x,fval,exitflag,output] = quadprog(optData.quadprogData.H, fvec, ...
            optData.quadprogData.A_leq, optData.quadprogData.b_leq, ...
            optData.quadprogData.A_eq, optData.quadprogData.b_eq, ...
            optData.quadprogData.lb, optData.quadprogData.ub, ...
            x0, ...
            optData.quadprogData.quadprogParams);
        
        if exitflag < 1
            if exitflag == 0
                if vl >= 4
                    warning('quadprog: %s', output.message);
                end
            else
                warning('quadprog: %s', output.message);
            end
        end
        
        if optData.quadprogData.formulation == 1
            % Option 1: Variable = [x; abs(x); residual]
            x = reshape(x(1:d),[d,1]);
        elseif optData.quadprogData.formulation == 2
            % Option 2: Variable = [abs(x)-x; abs(x); residual]
            x = reshape(x(d+1:2*d) - x(1:d),[d,1]);
        end
        
    end
elseif isEqualityConstrained
    %Variable: [x+; x-] %TODO: Implement this
    %x = res.sol.itr.xx(1:d) - res.sol.itr.xx(d+1:2*d);
    error('Not written yet');
end
        
dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;
dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end



function A_data = get_A_data(A, k)
%function A_data = get_A_data(A, k)
%
% This function conducts a basic analysis of A and returns a struct A_data
% that contains the following fields:
%
% n, d - The size of A = n x d
% k    - Desired sparsity level
% A_pinv     - Pseudo-inverse of A
% gamma_max  - Maximal singular value of A
% gamma_n    - n'th singular value of A
% maxColNorm - Maximal l2-norm of the columns of A
%
% Numbers relevant to the problem with power-1 residual:
% lambdaSmall_p1 - A number t such that if lambda < t, the solution x to
%                  problem (P_lambda) satisfies A*x-y
% lambdaLarge_p1 - A number t such that if lambda > t, the solution x to
%                  problem (P_lambda) is k-sparse.
%
% Numbers relevant to the problem with power-2 residual:
% lambdaSmallMultiplier_p2 - A number t such that if lambda < epsilon*t
%                            then the solution x to (P_lambda) satisfies
%                            ||A*x-y|| <= epsilon
% lambdaLargeMultiplier_p2 - A number t such that if lambda > t * ||y||_2
%                            then the solution x to (P_lambda) is k-sparse.
[n,d] = size(A);
v = svd(A);

A_data = struct();

A_data.n = n;
A_data.d = d;
A_data.k = k;

A_data.A_pinv = pinv(A);
A_data.A = A;

A_data.gamma_max = v(1);
A_data.gamma_n   = v(n);
A_data.maxColNorm = sqrt(max(sum(A.^2,1)));

A_data.lambdaSmall_p1 = A_data.gamma_n/sqrt(d-k);
A_data.lambdaLarge_p1 = A_data.maxColNorm;

A_data.lambdaSmallMultiplier_p2 = A_data.gamma_n/sqrt(d-k);
A_data.lambdaLargeMultiplier_p2 = A_data.maxColNorm;

% These will be used for optimization
A_data.A = A;

%A_data.AtA = A'*A;

%A_data.NA = null(A);
%A_data.NANAt = A_data.NA*A_data.NA';
A_data.rangeAt = orth(A');
end


function optData = makeOptData(A, y, k, A_data, params)
%% Pre-calculations done as a preprocessing step for optimization speedup
[n,d] = size(A);
optData = struct();

optData.A_data = A_data;

optData.A_sparse    = sparse(A);
optData.A_aug       = sparse([A; zeros(d,d)]);
optData.A_aug_wInds = find([zeros(n,d); eye(d)]);
optData.y_aug       = sparse([y; zeros(d,1)]);

optData.y = y;
optData.Aty = A'*y;
optData.PInvAy = A_data.A_pinv * y;

if params.residualPower == 1
    optData.lambdaEqualityGuarantee     = A_data.lambdaSmall_p1;
    optData.lambdaSparseGuarantee = A_data.lambdaLarge_p1;
elseif params.residualPower == 2
    optData.lambdaEqualityGuarantee = 0;
    optData.lambdaSparseGuarantee = A_data.lambdaLargeMultiplier_p2 * norm(y);
end

%% Matrices that will come handy for modeling optimization problems
Zn = zeros(n,n);
Zd = zeros(d,d);
Znd = zeros(n,d);
Zn1 = zeros(n,1);
Zd1 = zeros(d,1);
Zdn = zeros(d,n);
Z1n = zeros(1,n);
Z1d = zeros(1,d);
Zn1 = zeros(n,1);
Zd1 = zeros(d,1);

In = eye(n);
Id = eye(d);

Nnd = nan(n,d);
Nn1 = nan(n,1);

INFd1 = inf(d,1);
INFn1 = inf(n,1);

ONEn1 = ones(n,1);


%% Verify solver names
if ~any(strcmp(params.solver, {'mosek', 'gurobi', 'quadprog', 'gd', 'yalmip', 'debug'}))
    error('Invalid solver name ''%s'' given in params.solver', params.socpSolver);
end

%% Determine which solvers we need
need_yalmip = any(strcmp(params.solver, {'yalmip', 'debug'}));
need_mosek = strcmp(params.solver, 'mosek') || (strcmp(params.solver, 'debug') && params.debug_mosek);
need_gurobi = strcmp(params.solver, 'gurobi') || (strcmp(params.solver, 'debug') && params.debug_gurobi);
need_quadprog = strcmp(params.solver, 'quadprog') || (strcmp(params.solver, 'debug') && params.debug_quadprog);
need_gd = strcmp(params.solver, 'gd') || (strcmp(params.solver, 'debug') && params.debug_gd);

%% Determine which optimization methods we need to initialize
need_yalmip_constrlp = need_yalmip && params.start_with_equality_constraint;
need_mosek_constrlp  = need_mosek && params.start_with_equality_constraint;
need_gurobi_constrlp  = need_gurobi && params.start_with_equality_constraint;
need_quadprog_constrlp  = need_quadprog && params.start_with_equality_constraint;
need_gd_constrlp  = need_gd && params.start_with_equality_constraint;


%% Parameters and options for various solvers

if need_gd
    % lsqlin options
    lsqlin_options = optimoptions('lsqlin');
    lsqlin_options.Algorithm = 'active-set';
    optData.lsqlin_options = lsqlin_options;
end


%% Mosek
if need_mosek            
    % Mosek parameters
    basic_mosekParams = struct();
    basic_mosekParams.MSK_IPAR_NUM_THREADS = params.maxNumThreads;
    basic_mosekParams.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_ON';
    basic_mosekParams.MSK_IPAR_CHECK_CONVEXITY = 'MSK_CHECK_CONVEXITY_NONE';
    basic_mosekParams.MSK_IPAR_PRESOLVE_LINDEP_USE = 'MSK_OFF';
    basic_mosekParams.MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0;
    basic_mosekParams.MSK_IPAR_INTPNT_SOLVE_FORM = 'MSK_SOLVE_DUAL'; % MSK_SOLVE_FREE, MSK_SOLVE_PRIMAL, MSK_SOLVE_DUAL
    
    basic_mosekParams.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8;
    basic_mosekParams.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8;
    basic_mosekParams.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8;
    basic_mosekParams.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-8;
    
    %basic_mosekParams.MSK_DPAR_INTPNT_QO_TOL_REL_GAP = 1e-5;
    %basic_mosekParams.MSK_DPAR_INTPNT_QO_TOL_MU_RED = 1e-5;
    %basic_mosekParams.MSK_DPAR_INTPNT_QO_TOL_NEAR_REL = 1000;
    
    % Mosek parameters for the SOCP problem
    socp_mosekParams = basic_mosekParams;
    
    %socp_mosekParams.MSK_IPAR_INTPNT_BASIS = 'MSK_BI_NEVER';
    
    if params.maxNumThreads > 1
        socp_mosekParams.MSK_IPAR_INTPNT_MULTI_THREAD = 'MSK_ON';
    else
        socp_mosekParams.MSK_IPAR_INTPNT_MULTI_THREAD = 'MSK_OFF';
    end
        
    % Default is 1. Recently encountered performance problems that might be
    % solved when switching to 2. [TODO]
    optData.mosek_socp_formulation = 1;
    
    % Mosek parameters for the Constrained LP problem
    constrlp_mosekParams = basic_mosekParams;
    constrlp_mosekParams.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_ON'; %TODO: Check this parameter
    %constrlp_mosekParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE_SIMPLEX'; %TODO: Check which one is the best
    %constrlp_mosekParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_PRIMAL_SIMPLEX';
    constrlp_mosekParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_DUAL_SIMPLEX';
    %constrlp_mosekParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_CONIC';
    %constrlp_mosekParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_INTPNT';
    
    clear basic_mosekParams
    
    % More MOSEK
    mosek_options = mskoptimset('Display', 'off');
    
    % These will be needed for solving SOCP with MOSEK
    [~,res] = mosekopt('symbcon echo(0)');
    MSK_CT_QUAD = res.symbcon.MSK_CT_QUAD;
    MSK_CT_RQUAD = res.symbcon.MSK_CT_RQUAD;
    clear res
    
    
    %% Problem: SOCP. Solver: mosek2
    if (params.residualPower == 1)
        %TODO: Rewrite this
        optData.socp_mosekProb = struct();
        
        optData.socp_mosekParams = socp_mosekParams;
        
        % Variable: [x; abs(x); residual; norm(residual)]
        
        mosekMat = [A, Znd, -In, Zn1; Id, -Id, Zdn, Zd1; -Id, -Id, Zdn, Zd1];
        uc = [y; zeros(2*d,1)];
        lc = [y; -inf(2*d,1)];
        
        % Construct problem
        optData.socp_mosekProb.a = sparse(mosekMat);
        optData.socp_mosekProb.blc = lc;
        optData.socp_mosekProb.buc = uc;
        optData.socp_mosekProb.blx = [];
        optData.socp_mosekProb.bux = [];
        optData.socp_mosekProb.cones.type = MSK_CT_QUAD;
        optData.socp_mosekProb.cones.sub = [2*d+n+1, 2*d+1:2*d+n];
        optData.socp_mosekProb.cones.subptr = 1;
        optData.socp_mosekProb.options = mosek_options;
        
    elseif (params.residualPower == 2)
        if optData.mosek_socp_formulation == 1
            optData.socp_mosekProb = struct();
            
            optData.socp_mosekParams = socp_mosekParams;
            
            % Variable: [x; abs(x); residual]
            
            mosekMat = [A, Znd, -In; Id, -Id, Zdn; -Id, -Id, Zdn];
            uc = [y; zeros(2*d,1)];
            lc = [y; -inf(2*d,1)];
            
            %mosekQ = sparse(2*d+n,2*d+n);
            %mosekQ(2*d+1:2*d+n, 2*d+1:2*d+n) = eye(n);
            optData.socp_mosekProb.qosubi = 2*d+1:2*d+n;
            optData.socp_mosekProb.qosubj = 2*d+1:2*d+n;
            optData.socp_mosekProb.qoval  = ones(size(optData.socp_mosekProb.qosubi));
            
            % Construct problem
            optData.socp_mosekProb.a = sparse(mosekMat);
            %optData.socp_mosekProb.q = mosekQ;
            optData.socp_mosekProb.blc = lc;
            optData.socp_mosekProb.buc = uc;
            optData.socp_mosekProb.blx = [];
            optData.socp_mosekProb.bux = [];
            optData.socp_mosekProb.options = mosek_options;
        elseif optData.mosek_socp_formulation == 2
            optData.socp_mosekProb = struct();
            
            optData.socp_mosekParams = socp_mosekParams;
            
            % Variable: [x; abs(x); residual; residual norm; 1]
            
            mosekMat = [A, Znd, -In, Zn1, Zn1; Id, -Id, Zdn, Zd1, Zd1; -Id, -Id, Zdn, Zd1, Zd1; Z1d, Z1d, Z1n, 0, 1];
            uc = [y; zeros(2*d,1); 1];
            lc = [y; -inf(2*d,1); 1];
                        
            % Construct problem
            optData.socp_mosekProb.a = sparse(mosekMat);
            optData.socp_mosekProb.blc = lc;
            optData.socp_mosekProb.buc = uc;
            optData.socp_mosekProb.blx = [];
            optData.socp_mosekProb.bux = [];
            optData.socp_mosekProb.cones.type = MSK_CT_RQUAD;
            optData.socp_mosekProb.cones.sub = [2*d+n+1, 2*d+n+2, 2*d+1:2*d+n];
            optData.socp_mosekProb.cones.subptr = 1;
            optData.socp_mosekProb.options = mosek_options;
        end
    end
    
    %% Problem: Constrained-LP. Solver: mosek
    if need_mosek_constrlp
        % Create problem struct
        constrlp_mosekProb = struct();
        
        constrlp_mosekProb.a   = [A, -A];
        constrlp_mosekProb.blc = y;
        constrlp_mosekProb.buc = y;
        constrlp_mosekProb.blx = sparse([Zd1; Zd1]);
        constrlp_mosekProb.bux = [INFd1; INFd1];

        % This vector should be updated iteratively to:
        % [w; w]        
        constrlp_mosekProb.c   = sparse([Nn1; Nn1]);
        
        % Old formulation:
        %constrlp_mosekProb.a   = sparse([A, -A; -2*Id, Id]);
        %constrlp_mosekProb.blc = sparse([y; -INFd1]);
        %constrlp_mosekProb.buc = sparse([y; Zn1]);
        %constrlp_mosekProb.blx = sparse([-INFd1; Zn1]);
        %constrlp_mosekProb.bux = sparse([y; Zn1]);        
        % This vector should be updated iteratively to:
        % [w, zeros(d,1)]        
        %constrlp_mosekProb.c   = sparse([Nn1, Zn1]);
        
        % Options
        constrlp_mosekProb.options = mosek_options;
        
        % Save problem struct
        optData.constrlp_mosekProb = constrlp_mosekProb;
        
        % Save MOSEK parameters
        optData.constrlp_mosekParams = constrlp_mosekParams;
    end
end




%% Gurobi
if need_gurobi
    
    % Gurobi parameters
    basic_gurobiParams = struct();
    basic_gurobiParams.outputflag = 0;
    
    basic_gurobiParams.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8;
    basic_gurobiParams.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8;
    basic_gurobiParams.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8;
    basic_gurobiParams.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-8;
    basic_gurobiParams.MSK_IPAR_CHECK_CONVEXITY = 'MSK_CHECK_CONVEXITY_NONE';
    basic_gurobiParams.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_ON';
    basic_gurobiParams.MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 0;
    basic_gurobiParams.MSK_IPAR_PRESOLVE_LINDEP_USE = 'MSK_OFF';
    basic_gurobiParams.MSK_IPAR_CHECK_CONVEXITY = 'MSK_CHECK_CONVEXITY_NONE';
    
    % Mosek parameters for the SOCP problem
    socp_gurobiParams = basic_gurobiParams;
           
    
    % Mosek parameters for the Constrained LP problem
    constrlp_gurobiParams = basic_gurobiParams;
    constrlp_gurobiParams.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_ON'; %TODO: Check this parameter
    constrlp_gurobiParams.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE_SIMPLEX'; %TODO: Check which one is the best
    %constrlp_gurobiParams.MSK_IPAR_OPTIMIZER = 'MSK_IPAR_PRIMAL_SIMPLEX';
    %constrlp_gurobiParams.MSK_IPAR_OPTIMIZER = 'MSK_IPAR_DUAL_SIMPLEX';
    
    clear basic_gurobiParams
       
    
    %% Problem: SOCP. Solver: gurobi
    if (params.residualPower == 1)
        optData.socp_gurobiProb = struct();
        
        optData.socp_gurobiParams = socp_gurobiParams;
        
        optData.socp_gurobiProb.modelsense = 'min';
        
        % Variable: [x; abs(x); residual; norm(residual)]
        
        gurobiMat = [A, Znd, -In, Zn1; Id, -Id, Zdn, Zd1; -Id, -Id, Zdn, Zd1];
        rhs = [y; zeros(2*d,1)];
        sense = [repmat('=',[1,n]), repmat('<',[1,2*d])];
        
        Qc = sparse(2*d+n+1);
        Qc(2*d+1:2*d+n, 2*d+1:2*d+n) = speye(n);
        Qc(2*d+n+1, 2*d+n+1) = -1;
        
        q = zeros(2*d+n+1, 1);
        conrhs = 0;
                        
        % Construct problem
        optData.socp_gurobiProb.A = sparse(gurobiMat);
        optData.socp_gurobiProb.rhs = rhs;
        optData.socp_gurobiProb.sense = sense;
        optData.socp_gurobiProb.quadcon.Qc = Qc;
        optData.socp_gurobiProb.quadcon.q = q;
        optData.socp_gurobiProb.quadcon.rhs = conrhs;
        optData.socp_gurobiProb.quadcon.name = 'std_cone';
        
        % This should be updated to:
        %obj = [zeros(d,1); lambda*w; zeros(n,1); 1];
        optData.socp_gurobiProb.obj = nan(2*d+n+1, 1);
        
    elseif (params.residualPower == 2)
        optData.socp_gurobiProb = struct();
        
        optData.socp_gurobiParams = socp_gurobiParams;
        
        optData.socp_gurobiProb.modelsense = 'min';
        
        % Variable: [x; abs(x); residual; norm(residual)]
        
        gurobiMat = [A, Znd, -In, Zn1; Id, -Id, Zdn, Zd1; -Id, -Id, Zdn, Zd1];
        rhs = [y; zeros(2*d,1)];
        sense = [repmat('=',[1,n]), repmat('<',[1,2*d])];
        
        Qc = sparse(2*d+n+1);
        Qc(2*d+1:2*d+n, 2*d+1:2*d+n) = speye(n);
        Qc(2*d+n+1, 2*d+n+1) = -1;
        
        Q = sparse(2*d+n+1);
        Q(2*d+n+1, 2*d+n+1) = 1;
        
        q = zeros(2*d+n+1, 1);
        conrhs = 0;
                        
        % Construct problem
        optData.socp_gurobiProb.A = sparse(gurobiMat);
        optData.socp_gurobiProb.rhs = rhs;
        optData.socp_gurobiProb.sense = sense;
        optData.socp_gurobiProb.quadcon.Qc = Qc;
        optData.socp_gurobiProb.quadcon.q = q;
        optData.socp_gurobiProb.quadcon.rhs = conrhs;
        optData.socp_gurobiProb.quadcon.name = 'std_cone';
        optData.socp_gurobiProb.Q = Q;
        
        % This should be updated to:
        %optData.socp_gurobiProb.obj = [zeros(d,1); lambda*w; zeros(n+1,1)];
        optData.socp_gurobiProb.obj = nan(2*d+n+1, 1);

    end
    
    %% Problem: Constrained-LP. Solver: gurobi
    if need_gurobi_constrlp
        % Create problem struct
        constrlp_gurobiProb = struct();
        
        constrlp_gurobiProb.a   = [A, -A];
        constrlp_gurobiProb.blc = y;
        constrlp_gurobiProb.buc = y;
        constrlp_gurobiProb.blx = sparse([Zd1; Zd1]);
        constrlp_gurobiProb.bux = [INFd1; INFd1];

        % This vector should be updated iteratively to:
        % [w; w]        
        constrlp_gurobiProb.c   = sparse([Nn1; Nn1]);
        
        % Old formulation:
        %constrlp_gurobiProb.a   = sparse([A, -A; -2*Id, Id]);
        %constrlp_gurobiProb.blc = sparse([y; -INFd1]);
        %constrlp_gurobiProb.buc = sparse([y; Zn1]);
        %constrlp_gurobiProb.blx = sparse([-INFd1; Zn1]);
        %constrlp_gurobiProb.bux = sparse([y; Zn1]);        
        % This vector should be updated iteratively to:
        % [w, zeros(d,1)]        
        %constrlp_gurobiProb.c   = sparse([Nn1, Zn1]);
        
        % Options
        constrlp_gurobiProb.options = gurobi_options;
        
        % Save problem struct
        optData.constrlp_gurobiProb = constrlp_gurobiProb;
        
        % Save MOSEK parameters
        optData.constrlp_gurobiParams = constrlp_gurobiParams;
    end
end


%% Quadprog
if need_quadprog
    
    % Quadprog parameters
    quadprogParams = optimoptions('quadprog', ...
        'display', 'off', ... % 'final' or 'off'
        'diagnostics', 'off', ... $ 'on' or 'off'
        'algorithm', 'interior-point-convex', ... % 'interior-point-convex' or 'trust-region-reflective'
        'StepTolerance', 1e-12, ...
        'OptimalityTolerance', 1e-12);
    
    %% QP problem, quadprog solver
    if (params.residualPower == 1)
        error('Can only use quadratic residuals when solving with MATLAB''s quadprog');
        
    elseif (params.residualPower == 2)
        optData.quadprogData = struct();
        
        optData.quadprogData.formulation = 1;
        optData.quadprogData.quadprogParams = quadprogParams;
                        
        % Construct problem

        if optData.quadprogData.formulation == 1
            % Formulation 1
            % Variable: [x; abs(x); residual]

            optData.quadprogData.A_eq = ([A, Znd, -In]);
            optData.quadprogData.b_eq = y;
            
            optData.quadprogData.A_leq = ([Id, -Id, Zdn; -Id, -Id, Zdn]);
            optData.quadprogData.b_leq = zeros(2*d,1);
            
            optData.quadprogData.lb = [];
            optData.quadprogData.ub = [];
            
            if false
                % Convert equality constraints to inequality.
                % Necessary for using initialization with quadprog's
                % trust-region-reflective algorithm
                optData.quadprogData.A_leq = [optData.quadprogData.A_leq; optData.quadprogData.A_eq; -optData.quadprogData.A_eq];
                optData.quadprogData.b_leq = [optData.quadprogData.b_leq; optData.quadprogData.b_eq; -optData.quadprogData.b_eq];
                
                optData.quadprogData.A_eq = [];
                optData.quadprogData.b_eq = [];                
            end
                
        elseif optData.quadprogData.formulation == 2
            % Formulation 2
            % Variable: [abs(x)-x; abs(x); residual]

            optData.quadprogData.A_eq = [-A, A, -In];
            optData.quadprogData.b_eq = y;

            optData.quadprogData.A_leq = sparse([-2*Id, Id, Zdn]);
            optData.quadprogData.b_leq = zeros(d,1);
            
            optData.quadprogData.lb = [zeros(2*d,1); -inf(n,1)];
            optData.quadprogData.ub = [];

            if false
                % Convert equality constraints to inequality.
                % Necessary for using initialization with quadprog's
                % trust-region-reflective algorithm
                optData.quadprogData.A_leq = [optData.quadprogData.A_leq; optData.quadprogData.A_eq; -optData.quadprogData.A_eq];
                optData.quadprogData.b_leq = [optData.quadprogData.b_leq; optData.quadprogData.b_eq; -optData.quadprogData.b_eq];
                
                optData.quadprogData.A_eq = [];
                optData.quadprogData.b_eq = [];                
            end
        end
        
        H = sparse(2*d+n);
        H(2*d+1:2*d+n, 2*d+1:2*d+n) = speye(n);        
        optData.quadprogData.H = H;
                
        % This should be updated to:
        %optData.quadprogData.f = [zeros(d,1); lambda*w; zeros(n,1)];
        optData.quadprogData.f = nan(2*d+n, 1);
    end
    
    %% Problem: Constrained-LP. Solver: gurobi
    if need_quadprog_constrlp
        % Create problem struct
        constrlp_gurobiProb = struct();
        
        constrlp_gurobiProb.a   = [A, -A];
        constrlp_gurobiProb.blc = y;
        constrlp_gurobiProb.buc = y;
        constrlp_gurobiProb.blx = sparse([Zd1; Zd1]);
        constrlp_gurobiProb.bux = [INFd1; INFd1];

        % This vector should be updated iteratively to:
        % [w; w]        
        constrlp_gurobiProb.c   = sparse([Nn1; Nn1]);
        
        % Old formulation:
        %constrlp_gurobiProb.a   = sparse([A, -A; -2*Id, Id]);
        %constrlp_gurobiProb.blc = sparse([y; -INFd1]);
        %constrlp_gurobiProb.buc = sparse([y; Zn1]);
        %constrlp_gurobiProb.blx = sparse([-INFd1; Zn1]);
        %constrlp_gurobiProb.bux = sparse([y; Zn1]);        
        % This vector should be updated iteratively to:
        % [w, zeros(d,1)]        
        %constrlp_gurobiProb.c   = sparse([Nn1, Zn1]);
        
        % Options
        constrlp_gurobiProb.options = gurobi_options;
        
        % Save problem struct
        optData.constrlp_gurobiProb = constrlp_gurobiProb;
        
        % Save MOSEK parameters
        optData.constrlp_gurobiParams = constrlp_gurobiParams;
    end
end


%% Yalmip
if need_yalmip
    basic_yalmipSettings    = params.yalmip_settings;
    
    if isempty(basic_yalmipSettings)
        basic_yalmipSettings = sdpsettings('verbose', 0, 'cachesolvers', 1);
    end

    socp_yalmipSettings     = basic_yalmipSettings;
    constrlp_yalmipSettings = basic_yalmipSettings;
    clear basic_yalmipSettings
    
    %% Problem: SOCP. Solver: yalmip
    % Create optimization variables and constraints
    x_opt = sdpvar(d,1);
    xBound_opt = sdpvar(d,1);
    resNorm_opt = sdpvar(1,1);
    
    constraints = cone(A*x_opt-y, resNorm_opt);
    constraints = constraints + (x_opt <= xBound_opt);
    constraints = constraints + (-x_opt <= xBound_opt);
    
    % Create problem struct
    socp_yalmipData = struct();
    socp_yalmipData.x_opt = x_opt;
    socp_yalmipData.xBound_opt = xBound_opt;
    socp_yalmipData.resNorm_opt = resNorm_opt;
    socp_yalmipData.constraints = constraints;
    socp_yalmipData.settings = socp_yalmipSettings;
    
    % Objective should be updated iteratively:
    % objective = resNorm_opt + lambda*dot(w,xBound_opt)
    
    % Save problem struct
    optData.socp_yalmipData = socp_yalmipData;
    
    
    %% Problem: Constrained-LP. Solver: yalmip
    if need_yalmip_constrlp
        % Create optimization variables and constraints
        x_opt = sdpvar(d,1);
        xBound_opt = sdpvar(d,1);
        constraints = (x_opt <= xBound_opt) + (x_opt >= -xBound_opt) + ...
            (A*x_opt == y);
        
        % Objective will be updated iteratively:
        % objective = dot(w, xBound_opt)
        
        % Create problem struct
        constrlp_yalmipData = struct();
        constrlp_yalmipData.x_opt = x_opt;
        constrlp_yalmipData.xBound_opt = xBound_opt;
        constrlp_yalmipData.constraints = constraints;
        constrlp_yalmipData.settings = constrlp_yalmipSettings;
        
        % Save problem struct
        optData.constrlp_yalmipData = constrlp_yalmipData;
    end
    
end

end


function out = calcEnergy(A, x, y, k, lambda, residualPower)
%function out = calcEnergy(A, x, y, k, lambda, residualPower)
%
% This function calculates the energy of the untruncated solution.
% If lambda is zero, it returns just the tailnorm.

% This is true when we solve the Equality-constrained LP problem
% min tau_k(x) s.t. % A*x==y
isEqualityConstrained = (lambda == 0);

if isEqualityConstrained
    % Project to nearest feasible vector
    x = x + orth(A')*orth(A')'*(A\y-x);
    out = tailNorm(x,k);
else
    out = (1/residualPower)*norm(A*x-y)^residualPower + lambda * tailNorm(x,k);
end

end

function energy = calcEnergy_w(A, x, y, lambda, residualPower, w)
%function energy = calcEnergy_w(A, x, y, lambda, residualPower, w)

% This tells whether we solve the equality-constrained problem
%              min tau_k(x) s.t. % A*x==y
isEqualityConstrained = (lambda == 0);

if ~isEqualityConstrained
    energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda * dot(w,abs(x));
else
    % Project to nearest feasible vector
    %x = x + orth(A')*orth(A')'*(A\y-x);
    energy = dot(w,abs(x));
end

end

function [energy, gsm_term, w] = calcEnergy_gamma(A, x, y, k, lambda, gamma, residualPower)
%function [energy, gsm_term, w] = calcEnergy_gamma(A, x, y, k, lambda, gamma, residualPower)

% This tells whether we solve the equality-constrained problem
%              min tau_k(x) s.t. % A*x==y
isEqualityConstrained = (lambda == 0);

[n,d] = size(A);


% Calculate mu_{d-k,gamma}(x)
[gsm_term, w] = gsm_v4_5(abs(x), d-k, -gamma);

if ~isEqualityConstrained    
    energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda * gsm_term;
else
    % Project to nearest feasible vector
    %x = x + orth(A')*orth(A')'*(A\y-x);
    energy = gsm_term;
end

end


% Returns the difference of the smallest and largest sums of absolute
% values of d-k terms of x.
function out = maxDiffOfSums(x,k)
d = numel(x);
x = sort(abs(x));
out = sum(x(k+1:d)) - sum(x(1:d-k));
end


function v = stepVec_pwlin(varargin)
%function v = stepVec_pwlin(varargin)
%
% This function takes a list of values and numbers of steps and returns a
% piecewise-linear vector which moves between these values according to the
% given numbers of steps.
%
% Example: v = makeStepVec(0, 100, 5, 20, 4, 30, 7)
%          Creates a vector that starts at 0, progresses to 5 at 100 steps,
%          goes down to 4 at 20 steps and goes up to 7 at 30 steps.
v = varargin{1};

for i=1:(nargin-1)/2
    nSteps  = varargin{2*i};
    nextVal = varargin{2*i+1};
    stepSize = ( nextVal - v(end) ) / nSteps;
    v = [v, v(end)+stepSize*(1:nSteps)];
end

end


function v = stepVec_exp(a, b, decreaseFactor, nVals)
%function v = stepVec_exp(a, b, decreaseFactor, nVals)
%
% Given a < b and a decrease factor in (0,1), returns a vector v of nVals
% values in the interval [a,b] that start and b and decrease exponentially
% towards a by decreaseFactor at each step. The vector is shifted affinely
% such that v(1) = b and v(end) = a.
v = decreaseFactor .^ (0:(nVals-1));
v = v - min(v);
v = v / max(v);
v = a + v*(b-a);
end


function v = stepVec_tan(a, b, nVals)
%function v = stepVec_tan(a, b, nVals)
%
% Returns a vector v of nVals values in [a,b] such that arctan(v/b) is
% equi-spaced on the interval [arctan(a/b), arctan(1)]. In practice, the
% result is that the values are about twice as dense around point a as they
% are around point b.
v = b * tan(atan(a/b) + (0:(nVals-1))*(atan(1)-atan(a/b))/(nVals-1));
end

function lambdaVals = get_lambda_vals(params, A_data, norm_y)
%function lambdaVals = get_lambda_vals(params, A_data, norm_y)
%
% Determines the values of lambda to be used for solving P0.

%stepVec_tan = @(a,b,nPoints) b * tan(atan(a/b) + (0:(nPoints-1))*(atan(1)-atan(a/b))/(nPoints-1));

if ~isempty(params.lambdaVals)
    % If the user overrides the lambda values, use his override.    
    lambdaVals = params.lambdaVals;
else
    % We need to determine the values of lambda automatically.
    
    % We count solving with equality constraint Ax=y as one lambda value.
    if params.start_with_equality_constraint
        nVals = params.nLambdaVals - 1;
    else
        nVals = params.nLambdaVals;
    end

    if params.residualPower == 1
        % If using power 1 residual norm, the default lambda values
        % progress tangentially from the minimal lambda to the laximal
        % lambda. They are about twice more dense around lambda min than
        % around lambda max.
        lambdaVals = stepVec_tan((1-1e-4)*A_data.lambdaSmall_p1, (1+1e-4)*A_data.lambdaLarge_p1, nVals);
        
    elseif params.residualPower == 2
        % If using power 2 residual norm, the default lambda values
        % progress exponentially from 10^-8 * lambda^bar * norm(y), with
        % lambda^bar as defined in the paper.
        
        lambdaVals = exp(stepVec_pwlin(log(1e-8*(1+1e-4)*A_data.lambdaLargeMultiplier_p2*norm_y),nVals-1,log((1+1e-4)*A_data.lambdaLargeMultiplier_p2 * norm_y), nVals));
        
        % Older versions
        %lambdaVals = (stepVec_tan(sqrt(1e-4*A_data.lambdaLargeMultiplier_p2*norm_y),sqrt(A_data.lambdaLargeMultiplier_p2 * norm_y), nVals)).^2;
        %lambdaVals = exp(stepVec_pwlin(log(1e-4*A_data.lambdaLargeMultiplier_p2*norm_y),nVals-1,log(A_data.lambdaLargeMultiplier_p2 * norm_y), nVals));
    end
    
    lambdaVals = sort(lambdaVals);
    
    % We denote the equality-constrained problem min tau_k(x) s.t. Ax=y
    % by lambda=0.
    if params.start_with_equality_constraint
        lambdaVals = [0; lambdaVals(:)];
    end
end
end


function defaultParams = getDefaultParams(baseParams)
%function defaultParams = getDefaultParams(baseParams)
%
% Returns a struct containing the default parameters.
%
% Some of the default parameters depend on user overrides (e.g. some default
% thresholds depend on the user choice of profile ('fast'/'thourough'
% etc.). These special parameters that affect values of default parameters
% are called 'base parameters'.

defaultParams = baseParams;

%% Documented parameters
% See "List of parameters.txt" for a full documentation of the parameters.

% Profile. Can be: 'fast', 'normal', 'thorough', 'crazy'
profile = baseParams.profile;

defaultParams.x_init = [];
defaultParams.init_x_from_previous_lambda = false;
defaultParams.full_homotopy_with_init = false;

% Absolute sparsity threshold for a solution x of (P_lambda)
defaultParams.sparsityThreshAbs_x = chooseByKeyStr(profile, 'fast', 1e-5, 'normal', 1e-6, 'thorough', 1e-6, 'crazy', 1e-7);

% Loose absolute sparsity threshold for a solution x of (P_lambda)
defaultParams.looseSparsityThreshAbs_x = 1e-5;

% Loose relative sparsity threshold for a solution x of (P_lambda)
defaultParams.looseSparsityThreshRel_x = 1e-4;

% Stop running more instances of P_lambda when this number of sparse
% solutions is reached.
defaultParams.nLambdas_sparse_x_to_stop = chooseByKeyStr(profile, ...
    'fast', 4, 'normal', 7, 'thorough', 7, 'crazy', inf); 

% Stop solving more instances of P_{lambda,gamma} when this number of sparse
% solutions is reached. It is the safest to set it to inf. Values as high
% as 50 were observed to degrade performance, while leading to a speed up
% of no more than 25%.
defaultParams.nGammas_sparse_x_to_stop = chooseByKeyStr(profile, ...
    'fast', 5, 'normal', 10, 'thorough', 10, 'crazy', inf); 

defaultParams.sparsityThreshAbs_w = 1e-5;

defaultParams.nGammas_sparse_w_to_stop = ...
        chooseByKeyStr(profile, 'fast', 3, 'normal', 4, 'thorough', 4, 'crazy', 6);


% Verbosity level
% 0 - Quiet
% 1 - Summary only
% 2 - Summary and iteration reporting
% 3 - Summary, iteration reporting violation reporting
% 4 - All of the above, including warnings on each mosek / yalmip call 
%     that results in numerical problems.
defaultParams.verbosity = 2;

% Residual power. The power p in the objective:
% (1/p)*||Ax-y||^p + lambda * tau_k(x)
% p can be 1 or 2. Default: 1
defaultParams.residualPower = baseParams.residualPower;

defaultParams.nLambdaVals = chooseByKeyStr(profile, ...
    'fast', 25, 'normal', 50, 'thorough', 50, 'crazy', 100); 

defaultParams.start_with_equality_constraint = false;
defaultParams.P0_objective_stop_threshold = 0;

defaultParams.nGammaVals_max = 10000;

% Energy decrease threshold when solving for a single gamma
defaultParams.Plg_minimal_decrease = ...
    chooseByKeyStr(profile, 'fast', 1e-3, 'normal', 1e-3, 'thorough', 1e-3, 'crazy', 1e-4);

% Energy decrease threshold when solving for a single gamma
defaultParams.Plg_minimal_decrease_immediate = 1e-6;

% When solving for one value of gamma, number of consecutive non-decreases of
% the energy in order to conclude convergence.
defaultParams.Plg_num_small_decreases_for_stopping = ...
        chooseByKeyStr(profile, 'fast', 2, 'normal', 2, 'thorough', 2, 'crazy', 3);

% Same as above, but when using the final w
defaultParams.Plg_num_small_decreases_for_stopping_on_infinite_gamma = ...
    chooseByKeyStr(profile, 'fast', 3, 'normal', 6, 'thorough', 6, 'crazy', 6);

% Maximal iterations over different w's when solving for one value of gamma
defaultParams.Plg_max_num_mm_iters = 1000;

% The first nonzero gamma will be chosen such that the difference between the
% smallest and the largest sum of d-k absolute entries will be the value
% set here.
defaultParams.gamma_first_max_difference_of_sums = 1e-4;

% When optimizing for a single lambda:
% Each time gamma is multiplied by growthFactor.
defaultParams.gamma_growth_factor = ...
    chooseByKeyStr(profile, 'fast', 1.03, 'normal', 1.02, 'thorough', 1.02, 'crazy', 1.01);

% Taking larger numbers here than the regular gamma_growth_factor was only
% found to slow down the performance.
defaultParams.gamma_growth_factor_when_w_doesnt_move = defaultParams.gamma_growth_factor;
%    chooseByKeyStr(profile, 'fast', 1.15, 'normal', 1.1, 'thorough', 1.1, 'crazy', 1.05);

defaultParams.gamma_test_growth_factors = [10, 2, 1.2];
defaultParams.gamma_test_maximal_x_distance_abs = 1e-6;
defaultParams.gamma_test_maximal_x_distance_rel = 1e-6;

defaultParams.gamma_test_every_n_iters = 10;
defaultParams.gamma_test_counter_init = 9;

% Reducing this threshold below 1e-3 did not seem to make any improvement
% in performance or increase running time.
defaultParams.w_diff_thresh_to_keep_increasing_gamma = ...
    chooseByKeyStr(profile, 'fast', 1e-2, 'normal', 1e-3, 'thorough', 1e-4, 'crazy', 1e-4);

%% Undocumented parameters
defaultParams.use_golden_search = false;

% Chooses the solver we use for problems of the form
% min (1/p)*norm(A*x-y)^p + sum(w.*abs(x))
% Can be: 'mosek', 'gurobi', 'yalmip', 'gd', or 'debug'
% The 'debug' setting uses MOSEK but compares all methods to it.

% Settings to use when solver is set to yalmip
defaultParams.yalmip_settings = [];

% Optimization solver to use
if detect_mosek()
    defaultParams.solver = 'mosek';
elseif detect_gurobi()
    defaultParams.solver = 'gurobi';
elseif detect_yalmip()
    defaultParams.solver = 'yalmip';    
elseif detect_quadprog()
    defaultParams.solver = 'quadprog';
else
    defaultParams.solver = 'gd';
end


% Maximal number of threads to use in optimization. Note: This has no
% effect when using yalmip as the solver. For this to take effect, the
% parameter needs to pass to yalmip nanually in yalmip_settings.
defaultParams.maxNumThreads = 2;

defaultParams.use_omp = true;

defaultParams.wSol_gd_decreaseThresh = 0.999999;
%defaultParams.wSol_gd_epsVals = 10.^[-5,-6,-8,-11,-14];
defaultParams.wSol_gd_epsVals = 10.^[-4,-5,-10];
defaultParams.wSol_gd_epsVals = 10.^[-4,-4,-4];
defaultParams.wSol_gd_num_non_decreases_for_convergence = [3,1,1];
defaultParams.wSol_gd_num_non_decreases_for_convergence = [50,50,50];
defaultParams.wSol_gd_nMaxItersPerEps = [50,2,2];
defaultParams.wSol_gd_nMaxItersPerEps = [50,50,50];


% When solving least squares problem, solve exactly, using direct solution
% or lsqlin (slower and more precise) every d iterations. On the rest of
% the itrations solve the least squares using SSF. 
% Can be between 0 and above.
% Set to 1 for always using QP, or 0 for always using SSF.
defaultParams.exact_gd_solve_every_n_iter = 1;

% These parameters tell which of the methods we compare to SOCP when the
% 'debug' method was chosen.
defaultParams.debug_mosek     = detect_mosek();
defaultParams.debug_gurobi    = detect_gurobi();
defaultParams.debug_quadprog  = detect_quadprog();
defaultParams.debug_gd        = false;

% Report violations only when they are above this threshold
defaultParams.reportThresh_lambdaSol_solution_inferior_to_projSol = 0;
defaultParams.reportThresh_lambdaSol_nonZero_residual_for_small_lambda = 1e-6;
defaultParams.reportThresh_gammaSol_solution_inferior_to_init = 0;

defaultParams.reportThresh_wSol_solution_inferior_to_init = 0;
defaultParams.reportThresh_wSol_solution_inferior_to_projSol = 0;
defaultParams.reportThresh_gammaSol_energy_increase_majorizer_decrease = 0;
defaultParams.reportThresh_abs_gammaSol_gsm_larger_than_majorizer = 0;

defaultParams.reportThresh_wSol_mosek_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_gd_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_gurobi_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_quadprog_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_quadprog_inferior_to_yalmip = 0;

defaultParams.reportThresh_wSol_energy_increase_majorizer_decrease = 0;
defaultParams.reportThresh_wSol_majorizer_increase = 0;
defaultParams.reportThresh_wSol_nonZero_residual_for_small_lambda = 1e-6;
defaultParams.reportThresh_abs_wSol_nonSparse_solution_for_large_lambda = 1e-6;
end


function params = processParams(nameValPairs)
%function params = processParams(nameValPairs)
%
% Takes the user-defined parameter overrides, given in name-value pairs,
% and returns a struct 'params' that contraints the complete set of parameters.
%
% See the external documentation for a complete list of parameters.

% Convert name-value pairs to struct format
params = namevals2struct(nameValPairs);

%% Handle basic parameters
% Basic parameters are parameters that affect the default values of 
% other parameters. Like other parameters, they also have default values,
% which can be overridden by the user.

% Set default basic parameters
defaultBaseParams = struct();
defaultBaseParams.profile = 'normal';
defaultBaseParams.lambdaVals = [];
defaultBaseParams.residualPower = 2;
defaultBaseParams.start_with_equality_constraint = false;

% Fill in basic parameters that were not defined by the user
baseParams = addDefaultFields(params, defaultBaseParams, 'discard');
clear defaultBaseParams

% Verify and process basic parameters
baseParams.profile = lower(baseParams.profile);

if ~ismember(baseParams.profile, {'fast', 'normal', 'thorough', 'crazy'})
    error('Invalid profile name ''%s'' in parameter ''profile''', baseParams.profile);
end

if ~ismember(baseParams.residualPower, [1,2])
    error('Parameter ''residualPower'' should be 1 or 2');
end

%% Combine default parameters with user overrides
defaultParams = getDefaultParams(baseParams);

% Apply user overrides
params = addDefaultFields(params, defaultParams);

% Post-processing of parameters
params.solver = lower(params.solver);

if ~islogical(params.init_x_from_previous_lambda) || ~isscalar(params.init_x_from_previous_lambda)
    error('init_x_from_previous_lambda must be a logical scalar');
end

if ~islogical(params.full_homotopy_with_init) || ~isscalar(params.full_homotopy_with_init)
    error('full_homotopy_with_init must be a logical scalar');
end

end


function x_out = projectToZeroRangeError(x, optData)
%function x_out = projectToZeroRangeError(x, optData)
%
% Returns the closest vector to x which satisfies A*x=y
%x_out = optData.PInvAy + optData.A_data.NANAt*(x - optData.PInvAy);
x_out = x + optData.A_data.rangeAt*optData.A_data.rangeAt'*(optData.PInvAy-x);
end


function energy = calcEnergyW(x, lambda, residualPower, w, A, y)
isConstraindProb = (lambda == 0);

if ~isConstraindProb
    energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda*dot(w,abs(x));
else
    % Project to nearest feasible vector
    x = x + orth(A')*orth(A')'*(A\y-x);
    energy = dot(w,abs(x));
end

end


function [x_out, energy_out] = goldenSearch(A, y, lambda, w, x_curr, x_next, residualPower)
if isempty(x_curr)
    x_out = x_next;
    energy_out = [];
    return
end

%% Tuning #3
% Golden-search parameters
rad_growth_ratio = 2;
relStepSize_thresh = 1e-5;

% The golden ratio
phi = (sqrt(5)+1)/2;

% Perform golden search
x_step = x_next - x_curr;
relStepSize = norm(x_step,1) / norm(x_curr,1);

s0 = 0;
s1 = 1;
s2 = phi;

x0 = (1-s0)*x_curr + s0*x_next;
x1 = x_next;
x2 = (1-s2)*x_curr + s2*x_next;

f0 = calcEnergyW(x0, lambda, residualPower, w, A, y);
f1 = calcEnergyW(x1, lambda, residualPower, w, A, y);
f2 = calcEnergyW(x2, lambda, residualPower, w, A, y);

s2_good = (f2 >= f1);

while ~s2_good
    
    s2 = rad_growth_ratio * s2;     
    x2 = (1-s2)*x_curr + s2*x_next;
    f2 = calcEnergyW(x2, lambda, residualPower, w, A, y);
    s2_good = (f2 >= f1);
end

%x0 = (1-s0)*x_curr + s0*x_next; % This is not used
%x2 = (1-s2)*x_curr + s2*x_next; % This is not used

%f0 = calcEnergyW(x0, lambda, w, A, y); % This is not used
%f2 = calcEnergyW(x2, lambda, w, A, y); % This is not used

% Now we know that:
% A. s0 = 0, s1 = 1 and s2 >= phi
%    (Unlike traditional golden search, in which s2 = phi).
% B. f1 <= f0 and f1 <= f2
%    (Meaning that f1 is the initial possible candidate for minimum)

% Create the first probe point
sP = s0 + (phi-1)*(s1-s0);
xP = (1-sP)*x_curr + sP*x_next;
fP = calcEnergyW(xP, lambda, residualPower, w, A, y);

iIter = 0;

while (relStepSize*(s2 - s0) > relStepSize_thresh) && (iIter < 100)
    iIter = iIter + 1;
    
    if fP <= f1
        s2 = s1;
        %f2 = f1; % This is not used
        
        s1 = sP;
        f1 = fP;
        
        sP = s0 + (2-phi)*(s2-s0);
        xP = (1-sP)*x_curr + sP*x_next;
        fP = calcEnergyW(xP, lambda, residualPower, w, A, y);
    else
        s0 = sP;
        %f0 = fP; % This is not used
        
        sP = s1;
        fP = f1;
        
        s1 = s0 + (phi-1)*(s2-s0);
        x1 = (1-s1)*x_curr + s1*x_next;
        f1 = calcEnergyW(x1, lambda, residualPower, w, A, y);
    end
    
end

%fprintf('Golden search) nIter = %d Step size: %g  Diff: %g\n', nIter_goldenSearch, s1, s2-s0);

x_out = x1;
energy_out = f1;
end





function [x_out, I] = project(x,k)
[~,I] = sort(abs(x),'descend');
x_out = zeros(size(x));
I = I(1:k);
x_out(I) = x(I);
end

function x_out = refineGreedily(x,A,y,k)
[x, I] = project(x,k);

nIterMax = 300;
for i=1:nIterMax   
    x = zeros(size(x));
    x(I) = A(:,I)\y;
    
    I_prev = I;
    [x, I] = project(x,k);
    
    if all(sort(I) == sort(I_prev))
        break
    end
end

x_out = x;
end

function out = trimmedLasso(x,k)
out = sort(abs(x),'ascend');
out = sum(out(1:numel(x)-k));
end

function is_found = detect_mosek()
%function is_found = detect_mosek()
%
% Detects MOSEK and returns true if found.
is_found = ~isempty(which('mosekopt'));
end

function is_found = detect_yalmip()
%function is_found = detect_yalmip()
%
% Detects YALMIP and returns true if found.
is_found = ~isempty(which('yalmip'));
end

function is_found = detect_gurobi()
%function is_found = detect_gurobi()
%
% Detects GUROBI and returns true if found.
is_found = ~isempty(which('gurobi'));
end


function is_found = detect_quadprog()
%function is_found = detect_quadprog()
%
% Detects QUADPROG and returns true if found.
is_found = ~isempty(which('quadprog'));
end

function dealvec(v)
v = num2cell(v);
out = v{:};
end

function [x_min, energy_min] = update_min(x_min, energy_min, x_new, energy_new)
%function [x_min, energy_min] = update_min(x_min, energy_min, x_new, energy_new)
%
% This function is used for updating a variable x_min which the minimum
% encountered so far with respect to some energy measure.
% The update is as follows:
%
% [x_min, energy_min] = update_min(x_min, energy_min, x_new, energy_new)

if energy_new < energy_min
    x_min = x_new;
    energy_min = energy_new;
end

end


function [l0norm, support] = calc_numerical_l0norm(x, rel_thresh)
%function [l0norm, support] = calc_numerical_l0norm(x, rel_thresh)
%
% Calculates the l0-norm and the support of a given input vector x, according to the
% relative threshold rel_thresh. All entries of x with magnitude larger
% than rel_thresh*max(abs(x)) are considered nonzeros.
x_max = max(abs(x));
support = sort(find(abs(x) > rel_thresh*x_max));
l0norm = numel(support);
end


