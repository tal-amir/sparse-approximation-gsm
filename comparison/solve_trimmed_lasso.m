function [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, varargin)
%function [x_out, dbinfo] = solve_trimmed_lasso(A, y, k, ...params)
%
% Solves the problem:
%
%            min_x 0.5*||A*x-y||^2 + lambda*tau_k(x) + eta*||x||_1  (P1)
%
% based on the work of:
% [1] Dimitris Bertsimas and Martin S. Copenhaver and Rahul Mazumder (2017);
%     The Trimmed Lasso: Sparsity And Robustness, ArXiv.
% 
% This code implements the DC-Programming method, which appears in [1] as
% Algorithm 1, and serves as a wrapper code for the ADMM method, which
% appears in [1] as Algorithm 2. It seeks solutions to the problem (P1)
% as a surrogate for the sparse reconstruction problem
%            min_x ||A*x-y||_2 s.t. ||x||_0 <= k.                    (P0)
% 
% Several instances of Problem (P1), with several values of lambda and eta, 
% are solved, and the k-sparse projection of each solution is taken. 
% Eventually the projected solution with the smallest residual norm 
% ||A*x-y|| is returned.
%
%
% Input arguments
% ===============
% A, y, k - Must be supplied. All other parameters can be omitted or set to
%           an empty vector [].
% method - Can be 'dcp' for DC-programming or 'admm' for ADMM.
%          Default: 'dcp'
%
% lambdaVals - A vector of lambda values. The problem will be solved with
%              lambda having each of those values.
%              Default: 50 values that increse exponentially from
%              1e-8 * (1+1e-4) lambda^bar to (1+1e-4) lambda^bar,
%              where lambda^bar is as defined in [1]. When lambda is above
%              this threshold, the solution is guaranteed to be k-sparse.
%
% eta - A parameter used in Problem (P1). Can also be several values.
%       Default is [1e-2, 1e-6]. 
%       The value 1e-2 was used in the experiments of [1].
%
% objStopThresh - Early stop threshold on ||Ax-y||_2.
%                 When the residual l2-norm of the
%                 projected solution reaches below that threshold, the algorithm stops
%                 without solving for the remaining lambda and eta values.
%
% profile - Can be 'fast' or 'normal'. 
%           Default: 'normal'
%
% use_alternative_code_dcp, use_alternative_code_admm
%                      - Booleans that tell whether to use the alternative
%                        code written by Tal, or the translated Julia code 
%                        of Mazumder et la.
%
% Output arguments
% ================
% x_out - Best projected solution. The solution of (P1) for each lambda and eta is
%         projected to the nearest k-sparse vector. The one with the
%         smallest l2 residual norm is returned.
%
% dbinfo - A struct that contains the following fields:
%          lambdaVals - list of lambda values used in practice
%          etaVals- list of eta values used in practice
%          x_all - cell array of size [numel(eta), numel(lambdaVals)],
%          which contains in cell i,j the solution obtained for the
%          corresponding problem (P1) with eta = eta(i), lambda=
%          lambdaVals(j). For problems that were not solved, the
%          corresponing cell in x_all is left empty, i.e. [].
%

%% Set parameters
lambda_bar = norm(y) * sqrt(max(sum(A.^2,1)));

s = svd(A); s_min = s(size(A,1));
lambda_underscore = 1e-7 * s_min / sqrt(size(A,2)-k);

params = processParams(varargin, lambda_bar, lambda_underscore);

vl = params.verbosity;


% A function that tells if x is sparse according to the threshold parameter
isSparse = @(x) nonSparsityAbs(x, k) <= params.sparsityThreshAbs_x;

switch(params.method)
    case 'dcp'
        methodName = 'DC programming';
    case 'admm'
        methodName = 'ADMM';
end

qprint(vl,1,'\n');
qprint(vl,1,'Solving by Trimmed Lasso\n');
qprint(vl,1,'========================\n');
qprint(vl,1,'\n');
qprint(vl,1,'Method: %s\n', methodName);
qprint(vl,1,'Profile: %s\n', params.profile);

lambdaVals = params.lambdaVals;
etaVals = params.eta;

if ~isnan(params.one_lambda_mode) && (params.one_lambda_mode > 0 ) && (lambdaVals(end) ~= params.one_lambda_mode)
    lambdaVals = [lambdaVals(:); params.one_lambda_mode];
end

qprint(vl,1,'Using %d lambda values: %g --> %g and %d eta values\n', numel(lambdaVals), lambdaVals(1), lambdaVals(end), numel(etaVals));

if params.objStopThresh > 0
    qprint(vl,1,'Objective stop threshold: %g\n', params.objStopThresh);
end

qprint(vl,1,'\n');

nl = numel(lambdaVals);
ne = numel(etaVals);

x_all = cell(ne,nl);
nIter_all = nan(ne,nl);
tElapsed_all = nan(ne,nl);
objPerLambda = nan(ne,nl);
tailRelPerLambda = nan(ne,nl);
nIterPerLambda = nan(ne,nl);
isSparse_all = nan(ne,nl);

% Keeps track of the best P0 residual observed so far
res_best = inf;

x_curr = [];

itot = 0;

for il = 1:nl    
    lambda_curr = lambdaVals(il);
    
    for ie = 1:ne
        itot = itot+1;
        eta_curr = etaVals(ie);
                
        x_prev = x_curr;
        
        t_curr = tic;
        
        if (il == 1) || ~params.propagate_solutions_through_lambdas
            init_curr = params.x_init;
        else
            init_curr = x_prev;
        end
        
        switch(lower(params.method))
            case 'dcp'
                if params.use_alternative_code_dcp
                    [x_curr, nIter_curr] = solve_eta_lambda_dcp(A, y, k, eta_curr, lambda_curr, init_curr, params);
                else
                    [x_curr, nIter_curr] = tl_apx_altmin(size(A,2), k, y, A, eta_curr, lambda_curr);
                end
            case 'admm'
                if params.use_alternative_code_dcp
                    [x_curr, nIter_curr] = solve_eta_lambda_admm(A, y, k, eta_curr, lambda_curr, params.sigma_admm, init_curr, params.gamma_init, params);
                else
                    [x_curr, nIter_curr] = tl_apx_admm(size(A,2), k, y, A, eta_curr, lambda_curr);
                end
        end
                    
        t_curr = toc(t_curr);
        
        x_all{ie,il} = x_curr;
        isSparse_all(ie,il) = isSparse(x_curr);
        nIter_all(ie,il) = nIter_curr;
        tElapsed_all(ie,il) = t_curr;
        
        % Project current solution and perform greedy refinement
        x_proj = projectVec(x_curr,A,y,k);
        
        % Calculate residual of projected solution
        res_curr = norm(A*x_proj-y);
        objPerLambda(ie,il) = res_curr;
        tailRelPerLambda(ie,il) = norm(truncVec(x_curr,k),1)/norm(x_curr,1);        
               
        if res_curr < res_best
            res_best = res_curr;
            x_out = x_proj;
            bestStr = '*';
        else
            bestStr = ' ';
        end
                
        qprint(vl,1,'%s%d: lambda = %g, eta = %g, rel P0 resid. = %g, time = %g, nIter = %d\n', bestStr, itot, lambda_curr, eta_curr, res_curr / norm(y), t_curr, nIter_curr);
        
        if res_curr < params.objStopThresh
            qprint(vl,1,'Objective has reached below stopping threshold. Breaking\n');
            break
        end
    end
    
    % If, for the last <params.nLambdas_sparse_x_to_stop> values of lambda,
    % a sparse solution was obtained for all eta values, break.
    first_lambda_idx_to_check = min(nl,max(1,il+1-params.nLambdas_sparse_x_to_stop));
    sparseSolutionStop = all(all(isSparse_all(:,first_lambda_idx_to_check)));
    
    if sparseSolutionStop
        qprint(vl,1,'Sparse solution obtained for several consecutive values of lambda. Breaking\n');
        break
    end
end

dbinfo = struct();
dbinfo.lambda_bar = lambda_bar;
dbinfo.lambda_underscore = lambda_underscore;
dbinfo.x_all = x_all;
dbinfo.tElapsed_all = tElapsed_all;
%dbinfo.nIter_all = nIter_all(I);
dbinfo.nIter_all = nIter_all;
dbinfo.obj_all = objPerLambda;
dbinfo.tailRel_all = tailRelPerLambda;
end


function [x_sol, nIter] = solve_eta_lambda_dcp(A, y, k, eta, lambda, x_init, params)
vl = params.verbosity;
[m,n] = size(A);
%gamma_curr = lambda * k/n * ones(n,1);

% Initialize optimization variables
yalmip clear

options = sdpsettings('verbose',0,'solver','mosek','cachesolvers',1);
options.mosek.MSK_IPAR_NUM_THREADS = 2;

x_opt = sdpvar(n,1);
xbnd_opt = sdpvar(n,1);
res_opt = sdpvar(m,1);

const1 = (res_opt == A*x_opt - y) + (x_opt <= xbnd_opt) + (-x_opt <= xbnd_opt);
obj11 = 0.5*dot(res_opt,res_opt) + (lambda+eta)*sum(xbnd_opt);

gamma_opt = sdpvar(n,1);
gammabnd_opt = sdpvar(n,1);
const2 = (gamma_opt <= gammabnd_opt) + (-gamma_opt <= gammabnd_opt) + ...
    (sum(gammabnd_opt) <= lambda*k) + (gammabnd_opt <= lambda);

% Loop parameters
nIterMax = 10000;
objective_rel_decay_thresh = 1e-3;
tolRel_x = 0; %1e-8;
tolAbs_x = 0; %1e-9;

if isempty(x_init)
    x_curr = zeros(n,1);
else
    x_curr = x_init;
end

objCurr = 0.5*norm(A*x_curr-y)^2 + eta*norm(x_curr,1) + lambda * trimmedLasso(x_curr,k);

x_best = x_curr;
objBest = inf;

break_counter = 0;

for iIter = 1:nIterMax
    % Optimize over gamma
    if nnz(isnan(x_curr)) > 1
        qprint(vl,1,'Warning: NaNs in x_curr. Breaking\n');
        x_curr = x_best;
        break
    end
    obj2 = -dot(gamma_opt, x_curr);
    optimize(const2,obj2,options);

    gamma_curr = double(gamma_opt);

    % Optimize over x
    x_prev = x_curr;
    
    obj12 = -dot(gamma_curr, x_opt);
    obj1 = obj11 + obj12;
    
    optimize(const1,obj1,options);
    
    x_curr = double(x_opt);        
        
    % Report current iteration
    objPrev = objCurr;
    objCurr = 0.5*norm(A*x_curr-y)^2 + eta*norm(x_curr,1) + lambda * trimmedLasso(x_curr,k);
    
    qprint(vl,1,'%d: obj = %g\n', iIter, objCurr);
    
    if objCurr < objBest
        x_best = x_curr;
        objBest = objCurr;
    end
    
    % Monitor for convergence
    if iIter == 1
        continue
    end
    
    absDiff_curr = norm(x_curr-x_prev);
    relDiff_curr = norm(x_curr-x_prev)/norm(x_prev);        

    if objCurr > objPrev
        qprint(vl,1,'Objective has increased. Breaking\n');
        break
    elseif iIter == nIterMax
        qprint(vl,1,'Reached iteration limit. Breaking\n');
        break
    elseif relDiff_curr <= tolRel_x
        %qprint(vl,1,'Relative movement below threshold. Breaking\n');
        break_counter = break_counter + 1;
        %break
    elseif absDiff_curr <= tolAbs_x
        %qprint(vl,1,'Absolute movement below threshold. Breaking\n');
        break_counter = break_counter + 1;
        %break
    elseif (objPrev - objCurr) <= objPrev * objective_rel_decay_thresh
        break_counter = break_counter + 1;
    else
        break_counter = 0;
    end
    
    if break_counter >= 5
        qprint(vl,1,'Stopping criterion is met. Breaking\n');
        break
    end
end

x_sol = x_curr;
nIter = iIter;
end

function [x_sol, nIter] = solve_eta_lambda_admm(A, y, k, eta, lambda, sigma, x_init, gamma_init, params)
vl = params.verbosity;
[m,n] = size(A);

% Initialize optimization variables
yalmip clear

options = sdpsettings('verbose',0,'solver','mosek','cachesolvers',1);
options.mosek.MSK_IPAR_NUM_THREADS = 2;

x_opt = sdpvar(n,1);
xbnd_opt = sdpvar(n,1);
res_opt = sdpvar(m,1);

const1 = (res_opt == A*x_opt - y) + (x_opt <= xbnd_opt) + (-x_opt <= xbnd_opt);
obj11 = 0.5*dot(res_opt,res_opt) + eta*sum(xbnd_opt);

thorough_admm = true;

% Loop parameters
if thorough_admm
    nIterMax = 10000;
    abs_tol = 1e-6;
    rel_tol = 1e-6;

    nIterMin_for_breaking_on_non_decay = 50;
    objBestNonDecayIters = 200;
    objBestDecayThr = 0.96;
    sparsityThreshAbs_x = 1e-6;
else
    nIterMax = 2000;
    abs_tol = 1e-6;
    rel_tol = 1e-4;

    nIterMin_for_breaking_on_non_decay = 200;
    objBestNonDecayIters = 200;
    objBestDecayThr = 0.96;
    sparsityThreshAbs_x = 1e-6;
end

% Initialization
if isempty(x_init)
    x_curr = zeros(n,1);
else
    x_curr = x_init;
end

gamma_curr = zeros(n,1);
q_curr = zeros(n,1);

objCurr = 0.5*norm(A*x_curr-y)^2 + eta*norm(x_curr,1) + lambda * trimmedLasso(x_curr,k);
normCurr = norm(x_curr-gamma_curr);

soft = @(v,t) sign(v) .* max(abs(v)-t, 0);

x_best = x_curr;
objBest = inf;

objBestHistory = inf(objBestNonDecayIters,1);

supp_curr = [];

break_counter = 0;
sparse_x_counter = 0;

for iIter = 1:nIterMax
    % Optimize over gamma
    if nnz(isnan(x_curr)) > 1
        qprint(vl,1,'Warning: NaNs in x_curr. Breaking\n\n');
        x_curr = x_best;
        break
    end
    
    alpha_curr = x_curr + q_curr/sigma;
    [~,I] = sort(abs(alpha_curr), 'descend');
    I = I(1:k);
    
    gamma_prev = gamma_curr;
    
    gamma_curr = soft(alpha_curr, lambda/sigma);
    gamma_curr(I) = alpha_curr(I);
    
    % Verify gamma update
    gammaObj_prev = lambda * trimmedLasso(gamma_prev,k) + sigma/2*norm(x_curr-gamma_prev)^2 - dot(q_curr,gamma_prev);
    gammaObj_curr = lambda * trimmedLasso(gamma_curr,k) + sigma/2*norm(x_curr-gamma_curr)^2 - dot(q_curr,gamma_curr);
    
    if gammaObj_curr > gammaObj_prev
        qprint(vl,1,'Warning: Gamma minimization failed!\n\n');
    end
        
    % Update q
    q_curr = q_curr + sigma*(x_curr - gamma_curr);

    % Optimize over x    
    obj12 = dot(q_curr, x_opt);
    obj13 = sigma/2 * dot(x_opt - gamma_curr, x_opt - gamma_curr);
    obj1 = obj11 + obj12 + obj13;
    
    optimize(const1,obj1,options);
    
    x_prev = x_curr;
    x_curr = double(x_opt);
       
    supp_prev = supp_curr;
    [~, supp_curr] = sort(abs(x_curr), 'descend');
    supp_curr = sort(supp_curr(1:k));
    
    % Report current iteration
    objPrev = objCurr;
    objCurr = 0.5*norm(A*x_curr-y)^2 + eta*norm(x_curr,1) + lambda * trimmedLasso(x_curr,k);
    
    normPrev = normCurr;
    normCurr = norm(x_curr-gamma_curr);
        
    if objCurr < objBest
        x_best = x_curr;
        objBest = objCurr;
    end
    
    objBestHistory = [objBest; objBestHistory(1:numel(objBestHistory)-1, 1)];
    
    %qprint(vl,1,'%d: obj = %g  norm = %g\n', iIter, objCurr, normCurr);
    
    % Monitor for convergence
    if iIter <= 10
        continue
    end
    
    absDiff_curr = max(norm(x_curr-x_prev), norm(gamma_curr-gamma_prev));
    relDiff_curr = max(norm(x_curr-x_prev)/norm(x_prev), norm(gamma_curr-gamma_prev))/norm(gamma_prev);

    if iIter == nIterMax
        %qprint(vl,1,'Reached iteration limit. Breaking\n\n');
        break
    end
    
    if (nonSparsityAbs(x_curr, k) <= sparsityThreshAbs_x) && ...
            (isempty(supp_prev) || all(supp_curr == supp_prev))
        sparse_x_counter = sparse_x_counter + 1;
    else
        sparse_x_counter = 0;
    end
    
    if sparse_x_counter >= 5
        qprint(vl,1,'Sparse solution obtained. Breaking\n');
        break
    end
    
    if (min(objBestHistory) > objBestDecayThr * max(objBestHistory)) && ...
            (iIter >= nIterMin_for_breaking_on_non_decay)
        %qprint(vl,1,'Failed to improve best objective observed during %d iterations (improved only by %g%%). Breaking\n\n', objBestNonDecayIters, 100*(max(objBestHistory)-min(objBestHistory))/max(objBestHistory));
        break_counter = break_counter + 1;
    elseif absDiff_curr <= abs_tol
        %qprint(vl,1,'Absolute movement below threshold. Breaking\n\n');
        break_counter = break_counter + 1;
    elseif relDiff_curr <= rel_tol
        %qprint(vl,1,'Relative movement below threshold. Breaking\n\n');
        break_counter = break_counter + 1;
    else
        break_counter = 0;
    end
    
    if break_counter >= 5
        %qprint(vl,1,'Stopping criterion is met. Breaking\n');
        break
    end
end

x_sol = x_curr;
nIter = iIter;
end



function out = trimmedLasso(x,k)
out = sort(abs(x),'ascend');
out = sum(out(1:numel(x)-k));
end


function params = processParams(nameValPairs, lambda_bar, lambda_underscore)
%function params = processParams(nameValPairs, lambda_bar, lambda_underscore)
%
% This function takes the user-defined parameter overrides, verifies them
% and returns a struct that contraints the complete set of parameters.

% Convert cell array of name / value pairs to a struct
if numel(nameValPairs) == 0
    params = struct();    
elseif numel(nameValPairs) == 1
    params = nameValPairs{1};
    
    if isempty(params)
        params = struct();
    end
else
    params = namevals2struct(nameValPairs);
end


%% Initialize basic parameters
defaultBasicParams = getDefaultBasicParams();

% Add missing basic parameters
basicParams = addDefaultFields(params, defaultBasicParams, 'discard');

% Verify basic parameters
if ~ismember(lower(basicParams.profile), {'fast','normal'})
    error('Parameter ''profile'' should be ''fast'' or ''normal''');
end

%% Add all missing parameters
defaultParams = getDefaultParams(basicParams, lambda_bar, lambda_underscore);
params = addDefaultFields(params, defaultParams);

%% Post-process and verify parameters
params.method = lower(params.method);
params.profile = lower(params.profile);

if ~ismember(params.method, {'dcp', 'admm'})
    error('Invalid method ''%s''. Should be ''dcp'' or ''admm''', params.method);
end

if isempty(params.lambdaVals)
    error('lambdaVals must not be empty');
end

end



function params = getDefaultBasicParams()
params = struct();

params.profile = 'normal';
end


function params = getDefaultParams(basicParams, lambda_bar, lambda_underscore)
params = basicParams;

%% Create default parameters

% Linearly spaced lambda values (old)
lambdaVals_normal = (1+1e-4)*lambda_bar*10.^(-8*(50-(1:50))/(50-1));
lambdaVals_fast   = (1+1e-4)*lambda_bar*10.^(-8*(30-(1:30))/(30-1));

% Exponentially spaced lambda values (As used in GSM)
lambdaVals_normal = exp(stepVec_pwlin(log(1e-8*(1+1e-4)*lambda_bar),50-1, log((1+1e-4)*lambda_bar), 50));
lambdaVals_fast   = exp(stepVec_pwlin(log(1e-8*(1+1e-4)*lambda_bar),30-1, log((1+1e-4)*lambda_bar), 30));

params.lambdaVals = chooseByKeyStr(params.profile, 'normal', lambdaVals_normal, 'fast', lambdaVals_fast);

params.verbosity = 1;

% The parameter sigma used in the ADMM problem. See the ADMM formulation
% in [1].
params.sigma_admm = 1;

params.objStopThresh = 0;
params.eta = [1e-2, 1e-6]; %params.eta = 1e-6;
params.method = 'dcp';

% Initialization of x for DCP and ADMM
% Default: Zero. Leave blank for default.
params.x_init = [];

% Initialization of gamma for ADMM.
% Default: Zero. Leave blank for default.
params.gamma_init = [];

params.sparsityThreshAbs_x = 1e-6;
params.nLambdas_sparse_x_to_stop = 7;

params.one_lambda_mode = false;

% Tells whether we use the alternative code written by Tal, or the
% translated Julia code from the original authors.
params.use_alternative_code_dcp = true;
params.use_alternative_code_admm = false;

% Tells whether to take the solution obtained for lambda_i as
% initialization for lambda_{i+1}. Can only be used with one eta value.
params.propagate_solutions_through_lambdas = false;

end


function out = nonSparsityAbs(x,k)
x = sort(abs(x),'descend');
out = sum(x(k+1:end));
end


%function qprint(vl,1,varargin)
%end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mazumder's translated Julia code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% alternating minimization
%function tl_apx_altmin(p,k,y,X,mu,lambda,lassosolver=aux_lassobeta,max_iter=10000,rel_tol=1e-6,print_every=200)
function [beta, nIter] = tl_apx_altmin(p, k, y, X, mu, lambda, lassosolver, max_iter, rel_tol, print_every)

%%%%%

% This is known as Algorithm 1 in the paper BCM17 (using difference-of-convex optimization)

% Inputs:
%    data matrix `X` and response `y`
%    `p` is the number of columns of X (i.e., the number of features).
%    `mu` is the multipler on the usual Lasso penalty: mu*sum_i |beta_i|
%    `lambda` is the multipler on the trimmed Lasso penalty: lambda*sum_{i>k} |beta_{(i)}|
%    `solver` is the desired mixed integer optimization solver. This should have SOS-1 capabilities (will return error otherwise).
%    `bigM` is an upper bound on the largest magnitude entry of beta. if the constraint |beta_i|<= bigM is binding at optimality, an error will be thrown, as this could mean that the value of `bigM` given may have been too small.

% Optional arguments:
%    `lassosolver`---default value of `aux_lassobeta`, which is a simple Lasso problem solver whose implementation is included above as an auxiliary function. If you would like to solve the Lasso subproblems using your own Lasso solver, you should change this argument. Note that the `lassosolver` values expect as function which has the following characteristics:
%%                 Intput arguments will be as follows: `n` - dimension of row size of `X`;
%%                                                      `p` - as in outer problem;
%%                                                      `k` - as in outer problem;
%%                                                      `mu` - as in outer problem;
%%                                                      `lambda` - as in outer problem;
%%                                                      `XX` - value of transpose(X)*X (can be precomputed and stored offline);
%%                                                      `loc_b_c` - initial value of beta from which to initial the algorithm;
%%                                                      `grad_rest` - the remaining part of the gradient term (-X'*y- gamma).
%%                  Output: solution beta to the Lasso problem
%%                          minimize_beta norm(y???X*beta)^2 +(mu+lambda)*sum_i |beta_i| ??? dot(beta,gamma) (gamma is the solution from the alternating problem, as supplied in the additional gradient information).
%    `max_iter`---default value of 10000. Maximum number of alternating iterations for the algorithm.
%    `rel_tol`---default value of 1e-6. The algorithm concludes when the relative improvement (current_objective-previous_objective)/(previous_objective + .01) is less than `rel_tol`. The additional `0.01` in the denominator ensures no numerical issues.
%    `print_every`---default value of 200. Controls amount of amount output. Set `print_every=Inf` to suppress output.


% Output: estimator beta that is a *possible* solution for the problem
%         minimize_??    0.5*norm(y-X*??)^2 + ??*sum_i |??_i| + ??*T_k(??)

% Method: alternating minimization approach which finds heuristic solutions to the trimmed Lasso problem. See details in Algorithm 1 in BCM17.

%%%%%

if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 10000;
end

if ~exist('rel_tol', 'var') || isempty(rel_tol)
    rel_tol = 1e-6;
end

if ~exist('print_every', 'var') || isempty(print_every)
    print_every = 200;
end

if ~exist('lassosolver', 'var') || isempty(lassosolver)
    lassosolver = @aux_lassobeta;
end

AM_ITER = max_iter;
REL_TOL = rel_tol;
PRINT_EVERY = print_every; % AM will print output on every (PRINT_EVERY)th iteration

beta = randn(p,1);%starter;%zeros(p);
gamma = zeros(p,1);%starter;%zeros(p);

XpX = X'*X; % can separate computation if desired

n = size(X,1);

prev_norm = 0;
prev_obj = 0;

nIter = 0;

for I=0:AM_ITER
    nIter = nIter + 1;
    
    % solve wrt gamma (by sorting beta)
    
    II = zeros(p,1);
    sto = 0; % number set to "one" (really += lambda)
    
    bk = sort(abs(beta)); bk = bk(p-k+1);
    
    for i=1:p
        if (abs(beta(i)) > bk)
            gamma(i) = lambda*sign(beta(i));
            sto = sto + 1;
        else
            if (abs(beta(i)) < bk)
                gamma(i) = 0;
            else
                II(i) = 1;
            end
        end
    end
    
    if sum(II) == 0
        qprint(vl,1,'ERROR!\n');
    else
        if sum(II) == 1
            [~,imax] = max(II);
            gamma(imax) = lambda*sign(beta(imax));
            sto = sto + 1;
        else % |II| >= 2, so need to use special cases as detailed in paper's appendix
            %println(II);
            if bk > 0
                [~, j] = max(II); % arbitrary one from II ---> should probably choose randomly amongst them
                if dot(X(:,j),X*beta-y) + (mu+lambda)*sign(beta(j)) ~= 0
                    gamma(j) = 0;
                else
                    gamma(j) = lambda*sign(beta(j));
                    sto = sto + 1;
                end
                % assign rest of gamma
                for i=randperm(p)
                    if (sto < k) && (II(i) > 0.5)
                        gamma(i) = sign(randn())*lambda;
                        sto = sto + 1;
                    end
                end
                
            else % so bk == 0
                % need to check interval containment over indices in II
                notcontained = false;
                corrindex = -1;
                corrdot = inf;
                for i=randperm(p)
                    if II(i) > 0.5 % i.e. == 1
                        dp = dot(X(:,i),X*beta - y);
                        if (abs(dp) > mu)
                            notcontained = true;
                            corrindex = i;
                            corrdot = dp;
                            break;
                        end
                    end
                end
                
                if notcontained
                    j = corrindex;
                    if corrdot > mu
                        gamma(j) = -lambda;
                        sto = sto + 1;
                    else
                        gamma(j) = lambda;
                        sto = sto + 1;
                    end
                    % fill in rest of gamma
                    for i=randperm(p)
                        if (sto < k) && (II(i) > 0.5) && (i ~= j)
                            gamma(i) = sign(randn())*lambda;
                            sto = sto + 1;
                        end
                    end
                else % any extreme point will do
                    for i=randperm(p)
                        if (sto < k) && (II(i) > 0.5)
                            gamma(i) = sign(randn())*lambda;
                            sto = sto + 1;
                        end
                    end
                end
                
            end
        end
    end
    
    % ensure that sto == k
    
    if sto ~= k
        qprint(vl,1,'ERROR. EXTREME POINT NOT FOUND. ABORTING.\n');
        % println(gamma);
        % println(sto);
        % println(II);
        % println(beta);
        II(1)
    end
    
    
    % solve wrt beta
    
    beta = lassosolver(n,p,k,mu,lambda,XpX,beta,-X'*y- gamma);
    
    % perform updates as necessary
    
    tepmVec = sort(abs(beta));
    cur_obj = .5*norm(y-X*beta)^2 + mu*norm(beta,1) +lambda*sum(tepmVec(1:p-k));
    
    if abs(cur_obj-prev_obj)/(prev_obj+.01) < REL_TOL % .01 in denominator is for numerical tolerance with zero
        %qprint(vl,1,'I: %d\n', I);
        % println(cur_obj);
        % println(prev_obj);
        break; % end AM loops
    end
    
    prev_obj = cur_obj;
    
end

end


%function aux_lassobeta(n::Int,p::Int,k::Int,mu::Float64,lambda::Float64,XX::Array{Float64,2},loc_b_c::Array{Float64,1},grad_rest::Array{Float64,1},max_iter=10000,tol=1e-3)
function lbc = aux_lassobeta(n, p, k, mu, lambda, XX, loc_b_c, grad_rest, max_iter, tol)
% solve subproblem wrt beta, with (outer) beta as starting point

if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 10000;
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

MAX_ITERS = max_iter;
TOL = tol;

lbc = loc_b_c;
lbp = loc_b_c - ones(p,1);
tcur = 1/norm(XX);
iterl = 0;

while (iterl < MAX_ITERS) && ( norm(lbc - lbp) > TOL )
    
    lbp = lbc;
    
    gg = lbc - tcur*(XX*lbc + grad_rest);
    
    lbc = sign(gg).*max(abs(gg)-tcur*(mu+lambda)*ones(p,1), zeros(p,1));
    
    %tcur = TAU*tcur;
    
    iterl = iterl + 1;
    
end

end



%function tl_apx_admm(p,k,y,X,mu,lambda,max_iter=2000,rel_tol=1e-6,sigma=1.,print_every=200)
function [gamma, nIter] = tl_apx_admm(p,k,y,X,mu,lambda,max_iter,rel_tol,sigma,print_every)

%%%%%

% This is known as Algorithm 2 in the paper BCM17 (using augmented Lagranian and alternating direction method of multiplers, a.k.a. ADMM)

% Inputs:
%    data matrix `X` and response `y`
%    `p` is the number of columns of X (i.e., the number of features).
%    `mu` is the multipler on the usual Lasso penalty: mu*sum_i |beta_i|
%    `lambda` is the multipler on the trimmed Lasso penalty: lambda*sum_{i>k} |beta_{(i)}|
%    `solver` is the desired mixed integer optimization solver. This should have SOS-1 capabilities (will return error otherwise).
%    `bigM` is an upper bound on the largest magnitude entry of beta. if the constraint |beta_i|<= bigM is binding at optimality, an error will be thrown, as this could mean that the value of `bigM` given may have been too small.

% Optional arguments:
%    `max_iter`---default value of 2000. Maximum number of (outer) ADMM iterations for the algorithm.
%    `rel_tol`---default value of 1e-6. The algorithm concludes when the relative improvement (current_objective-previous_objective)/(previous_objective + .01) is less than `rel_tol`. The additional `0.01` in the denominator ensures no numerical issues.
%    `sigma`---default value of 1.0. This is the augmented Lagranian penalty as shown in Algorithm 2 in the paper.
%    `print_every`---default value of 200. Controls amount of amount output. Set `print_every=Inf` to suppress output.

% Output: estimator beta that is a *possible* solution for the problem
%         minimize_??    0.5*norm(y-X*??)^2 + ??*sum_i |??_i| + ??*T_k(??)

% Method: alternating minimization approach which finds heuristic solutions to the trimmed Lasso problem. See details in Algorithm 1 in BCM17.

%%%%%

if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 2000;
end

if ~exist('rel_tol', 'var') || isempty(rel_tol)
    rel_tol = 1e-6;
end

if ~exist('sigma', 'var') || isempty(sigma)
    sigma = 1;
end

if ~exist('print_every', 'var') || isempty(print_every)
    print_every = 200;
end

ADMM_ITER = max_iter;
REL_TOL = rel_tol;
% TAU = tau; ---> Could add the scaling parameter tau, but we will neglect to include that in our implementation
SIGMA = sigma;
PRINT_EVERY = print_every; % AM will print output on every (PRINT_EVERY)th iteration


XpX = X'*X; % can separate computation if desired

n = size(X,1);

% ADMM vars
beta = zeros(p,1);%starter;%zeros(p);
gamma = zeros(p,1);%starter;%zeros(p);
q = zeros(p,1);

% <solve ADMM>

prev_norm = 0;
prev_obj = 0;

nIter = 0;

for I=0:ADMM_ITER
    nIter = nIter+1;
    
    beta = aux_admmwrtbeta(n,p,k,mu,lambda,XpX,beta,q-X'*y- SIGMA*gamma,SIGMA);
    
    %%% solve wrt gamma
    
    %aux_sb = min([ ...
    %    SIGMA/2*(beta.^2) + q.*beta+(1/2/SIGMA)*(q.^2), ...
    %    (lambda^2)/(2*SIGMA)*ones(p,1) + lambda*abs(beta+q/SIGMA+lambda/SIGMA*ones(p,1)), ...
    %    (lambda^2)/(2*SIGMA)*ones(p,1) + lambda*abs(beta+q/SIGMA-lambda/SIGMA*ones(p,1)) ...
    %    ], [], 2);
    %
    %[~,sort_inds] = sort(aux_sb);
    %zz = zeros(p,1);
    %zz(sort_inds(1:p-k)) = 1;
    

    soft = @(v,t) sign(v) .* max(abs(v)-t, 0);
    
    alpha = beta + q/SIGMA;
    t_temp = lambda/SIGMA;
    
    gamma = soft(alpha, t_temp);
    
    aux_sb = min([0.5*alpha.^2, 0.5*t_temp.^2 + t_temp.*abs(alpha+t_temp), 0.5*t_temp.^2 + t_temp.*abs(alpha-t_temp)], [], 2);    
    [~,inds] = sort(aux_sb, 'descend'); inds = sort(inds(1:k));
    [~,inds2] = sort(abs(alpha), 'descend'); inds2 = sort(inds2(1:k));
    if any(inds ~= inds2)
        qprint(vl,1,'Different!\n');
    end
    gamma(inds) = alpha(inds);
    
    %sb = sort([(aux_sb(i),i) for i=1:p]);
    % zz = zeros(p);
    % for i=1:(p-k)
    %println(i);
    %   zz[sb(i)[2]] = 1;
    % end
    
%     for i=1:p
%         if zz(i) == 0
%             gamma(i) = (beta(i)) + (q(i))/SIGMA;
%         else % zz(i) = 1
%             aar1 = [SIGMA/2*(beta(i)^2) + q(i)*beta(i)+(1/2/SIGMA)*(q(i)^2), ...
%                 (lambda^2)/(2*SIGMA) + lambda*abs(beta(i)+q(i)/SIGMA+lambda/SIGMA), ...
%                 (lambda^2)/(2*SIGMA) + lambda*abs(beta(i)+q(i)/SIGMA-lambda/SIGMA)];
%             
%             aar2 = [0, ...
%                 beta(i) + q(i)/SIGMA + lambda/SIGMA, ...
%                 beta(i) + q(i)/SIGMA - lambda/SIGMA];
%             %println(aar);
%             [~, i_min] = min(aar1);
%             %gamma(i) = sort(aar)[1][2];
%             gamma(i) = aar2(i_min);
%             %println(gamma(i));
%         end
%     end
%     
    
    q = q + SIGMA*(beta-gamma);
    
    cur_norm = norm(beta-gamma);
    tempVec = sort(abs(beta));
    cur_obj = .5*norm(y-X*beta)^2 + mu*norm(beta,1) +lambda*sum(tempVec(1:(p-k)));
    
    %println(abs(cur_norm-prev_norm)/(prev_norm+.01) ," , ", abs(cur_obj-prev_obj)/(prev_obj+.01) );
    if abs(cur_norm-prev_norm)/(prev_norm+.01) + abs(cur_obj-prev_obj)/(prev_obj+.01) < REL_TOL % .01 in denominator is for numerical tolerance with zero
        % println(I);
        break; % end ADMM loops
    end
    
    prev_norm = cur_norm;
    prev_obj = cur_obj;
    
end

% </ end ADMM>
end



%function aux_admmwrtbeta(n::Int,p::Int,k::Int,mu::Float64,lambda::Float64,XX::Array{Float64,2},loc_b_c::Array{Float64,1},grad_rest::Array{Float64,1},sigma,max_iter=10000,tol=1e-3)
function lbc = aux_admmwrtbeta(n, p, k, mu, lambda, XX, loc_b_c, grad_rest, sigma, max_iter, tol)
% solve subproblem wrt beta, with (outer) beta as starting point

if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 10000;
end

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

MAX_ITERS = max_iter;
TOL = tol;
SIGMA = sigma;

lbc = loc_b_c;
lbp = loc_b_c - ones(p,1);
tcur = 1/norm(XX+SIGMA*eye(p));
iterl = 0;

while (iterl < MAX_ITERS) && ( norm(lbc - lbp) > TOL )
    
    lbp = lbc;
    
    gg = lbc - tcur*((XX+SIGMA*eye(p))*lbc + grad_rest);
    
    lbc = sign(gg).*max(abs(gg)-tcur*mu*ones(p,1), zeros(p,1));
    
    %tcur = TAU*tcur;
    
    iterl = iterl + 1;
    
end

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
