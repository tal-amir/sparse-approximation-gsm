%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'sparse_approx_gsm.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% varargin - name/value pairs of parameters. See the documentation.
%
% Output arguments:
% x_sol - Estimated solution vector to the sparse apprixmation problem
% sol   - A struct containing information about the solution. See external
%         documentation.
%
% This program requires the Mosek optimization solver.
% https://www.mosek.com/downloads/

version = '1.22';  version_date = '30-Oct-2020';

% =========================================================================

% Generates a parameter struct from the default parameters and the user
% overrides.
params = processParams(varargin, A, y, k);
vl = params.verbosity; % Verbosity level

qprintln(vl,2,'\nSparse Approximation by the Generalized Soft-Min Penalty');
qprintln(vl,2,'Tal Amir, Ronen Basri, Boaz Nadler - Weizmann Institute of Science');
qprintln(vl,2,'Version %s, %s', version, version_date);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'addDefaultFields.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s_out = addDefaultFields(s_in, s_defaults, nonExistantFieldAction)
%function s_out = addDefaultFields(s_in, s_defaults, nonExistantFieldAction)
%
% This function takes a struct s_in, and a 'default value' struct
% s_defaults, and for each field that is not set in s_in, sets it to the
% appropriate value given in s_default. The result is returned in s_out.
%
% If s_in contains fields that are not given default values in s_defaults,
% an error is thrown.
%
% If s_in is set to [], s_defaults is returned.
%
% nonExistantFieldAction - (optional) Determines the action taken when
%                          s_in contains a field that does not exist in
%                          s_default. Possible values:
%                          'allow'   - Such fields are added to s_out
%                          'discard' - Such fields are discarded from s_out
%                          'error'   - Return an error message
%                          

if ~exist('nonExistantFieldAction', 'var') || isempty(nonExistantFieldAction)
    nonExistantFieldAction = 'error';
end

nonExistantFieldAction = lower(nonExistantFieldAction);

if ~ismember(nonExistantFieldAction, {'allow', 'discard', 'error'})
    error('Invalid option ''%s'' for argument nonExistantFieldAction.\nValid options: ''allow'', ''discard'', ''error''');
end

if isempty(s_in)
    s_out = s_defaults;
    return
end

userDefinedFieldNames = fieldnames(s_in);
isFieldNameValid = false(size(userDefinedFieldNames));

defaultFieldNames = fieldnames(s_defaults);
ndefs = numel(defaultFieldNames);

s_out = s_in;


for i=1:ndefs
    currName = defaultFieldNames{i};
    currVal  = getfield(s_defaults, currName);
    
    inds = find(strcmp(currName, userDefinedFieldNames));
    
    if ~isempty(inds)
        isFieldNameValid(inds) = 1;
    else
        s_out = setfield(s_out, currName, currVal);
    end
end

if (sum(isFieldNameValid == 0) > 0)
    badFieldNums = find(isFieldNameValid == 0);
    
    if strcmp(nonExistantFieldAction, 'error')
    if numel(badFieldNums) == 1
        error('Bad field name ''%s''', userDefinedFieldNames{badFieldNums});
    else
        badFieldNames = '';
        for i=1:numel(badFieldNums)-1
            badFieldNames = cat(2, badFieldNames, userDefinedFieldNames{badFieldNums(i)}, ', ');
        end
        
        badFieldNames = cat(2, badFieldNames, userDefinedFieldNames{badFieldNums(numel(badFieldNums))});
        
        error('Bad field names: %s', badFieldNames);
    end
    
    elseif strcmp(nonExistantFieldAction, 'discard')
        s_out = rmfield(s_out, userDefinedFieldNames(badFieldNums));
    end    
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calcEnergy.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = calcEnergy(A, x, y, k, lambda, residualPower)
%function out = calcEnergy(A, x, y, k, lambda, residualPower)
%
% This function calculates the energy of the untruncated solution.
% If lambda is zero, it returns just the tailnorm.
out = (1/residualPower)*norm(A*x-y)^residualPower + lambda * tailNorm(x,k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calcEnergyW.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function energy = calcEnergyW(x, lambda, residualPower, w, A, y)
energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda*dot(w,abs(x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calc_F_lambda_gamma.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [energy, gsm_term, w] = calc_F_lambda_gamma(A, x, y, k, lambda, gamma, residualPower)
%function [energy, gsm_term, w] = calc_F_lambda_gamma(A, x, y, k, lambda, gamma, residualPower)
[n,d] = size(A);

% Calculate mu_{d-k,gamma}(x)
[gsm_term, w] = gsm_v5_1(abs(x), d-k, -gamma);

energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda * gsm_term;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calc_F_lambda_w.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function energy = calc_F_lambda_w(A, x, y, lambda, residualPower, w)
%function energy = calc_F_lambda_w(A, x, y, lambda, residualPower, w)
energy = (1/residualPower)*norm(A*x-y)^residualPower + lambda * dot(w,abs(x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calc_numerical_l0norm.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s, support] = calc_numerical_l0norm(x, abs_thresh, rel_thresh)
%function [s, support] = calc_numerical_l0norm(x, abs_thresh, rel_thresh)
%
% Calculates the l0-norm and the support of a given input vector x.
% If |x|_(i) is the ith largest magnitude entry of x, then:
% s_rel = argmin_i  |x|_(i+1) <= rel_thresh * mean(|x|_(1),...,|x|_(i))
% s_abs = number of indices i s.t. |x|_i > abs_thresh
% s = min(s_abs,s_rel)

[mag_sort, I] = sort(abs(x),'descend');
avg_fwd = cumsum(mag_sort);
avg_fwd = avg_fwd(:) ./ (1:numel(avg_fwd))';
%avg_bck = cummax(mag_sort,'reverse');
%avg_bck = avg_bck(:) ./ (numel(avg_bck):-1:1)';
avg_bck = mag_sort;

tauvals = avg_bck(2:end) ./ avg_fwd(1:(end-1));
s_abs = nnz(mag_sort >  abs_thresh);
s_rel = numel(x) - nnz(tauvals <= rel_thresh);
s = min(s_abs,s_rel);
support = I(1:s);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'calc_supp.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function supp = calc_supp(x,k)
[~, supp] = maxk(abs(x),k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'chooseByKeyStr.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = chooseByKeyStr(key, varargin)
%function out = chooseByKeyStr(key, varargin)
%
% Chooses a value according to a given key string.
%
% Example: thresh = chooseByKeyStr(profileName, 'fast', 0.01, 'normal', 0.5, 'thorough', 0.9)
%          Sets 'thresh' to be 0.01, 0.5 or 0.9, depending on whether
%          profileName is 'fast', 'normal' or 'thorough' respectively.
found = false;

for i=1:numel(varargin)/2
    if strcmp(key, varargin{2*i-1})
        if found
            error('Key ''%s'' appears in arguments more than once', key);
        end
        
        out = varargin{2*i};
        found = true;
    end
end

if ~found
    error('Key ''%s'' is not supplied in arguments', key);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'dealvec.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dealvec(v)
v = num2cell(v);
out = v{:};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'detect_gurobi.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function is_found = detect_gurobi()
%function is_found = detect_gurobi()
%
% Detects GUROBI and returns true if found.
is_found = ~isempty(which('gurobi'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'detect_mosek.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function is_found = detect_mosek()
%function is_found = detect_mosek()
%
% Detects MOSEK and returns true if found.
is_found = ~isempty(which('mosekopt'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'detect_quadprog.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function is_found = detect_quadprog()
%function is_found = detect_quadprog()
%
% Detects QUADPROG and returns true if found.
is_found = ~isempty(which('quadprog'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'detect_yalmip.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function is_found = detect_yalmip()
%function is_found = detect_yalmip()
%
% Detects YALMIP and returns true if found.
is_found = ~isempty(which('yalmip'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'getDefaultParams.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function defaultParams = getDefaultParams(baseParams, A,y,k)
%function defaultParams = getDefaultParams(baseParams)
%
% Returns a struct containing the default parameters.
% See documentation for a full specification of the parameters.
%
% Some of the default parameters depend on user overrides (e.g. some default
% thresholds depend on the user choice of profile ('fast'/'thourough'
% etc.). These special parameters that affect values of default parameters
% are called 'base parameters'.

defaultParams = baseParams;

%% General parameters
%  ------------------

% Profile. Can be one of 'fast', 'normal', 'thorough', 'ultra'
profile = baseParams.profile;

% Verbosity level
% 0 - Quiet
% 1 - Summary only
% 2 - Summary and iteration reporting (default)
% 3 - Summary, iteration reporting, violation reporting
% 4 - All of the above, including warnings on each mosek / yalmip call
%     that results in numerical problems.
defaultParams.verbosity = baseParams.verbosity;

defaultParams.monitor_violations = ...
    ~strcmp(baseParams.profile, 'fast') || (baseParams.verbosity >= 3) || strcmp(baseParams.solver, 'debug');

%% Parameters related to (P0)
%  --------------------------

% Residual power. The power p in the objective:
% (1/p)*||Ax-y||^p + lambda * tau_k(x)
% p can be 1 or 2. Default: 2
defaultParams.residualPower = baseParams.residualPower;

defaultParams.P0_objective_stop_threshold = 0;

defaultParams.nLambdaVals = chooseByKeyStr(profile, ...
    'fast', 5, 'normal', 10, 'thorough', 31, 'ultra', 43);

defaultParams.lambdaRel_min = ...
    chooseByKeyStr(profile, 'fast', 1e-2, 'normal', 1e-3, 'thorough', 1e-5, 'ultra', 1e-7);


% Stop running more instances of P_lambda when this number of sparse
% solutions is reached.
defaultParams.nLambdas_sparse_x_to_stop = chooseByKeyStr(profile, ...
    'fast', 1, 'normal', 1, 'thorough', 2, 'ultra', 3);


%% Parameters related to F_lambda(x)
%  ---------------------------------
defaultParams.x_init = [];
defaultParams.init_x_from_previous_lambda = false;
defaultParams.full_homotopy_with_init = false;

% The first nonzero gamma will be chosen such that the difference between the
% smallest and the largest sum of d-k absolute entries will be the value
% set here.
defaultParams.gamma_first_max_difference_of_sums = 1e-4;

defaultParams.gamma_binary_search_uncertainty_ratio = chooseByKeyStr(profile, ...
    'fast', 1.1, 'normal', 1.1, 'thorough', 1.02, 'ultra', 1.01);

% When optimizing for a single lambda:
% Each time gamma is multiplied by growthFactor.
defaultParams.gamma_growth_factors = ...
    chooseByKeyStr(profile, ...
    'fast',     [1.03, 1.1, 1.2, 1.5], ...
    'normal',   [1.02, 1.1, 1.2, 1.5], ...
    'thorough', [1.02, 1.1, 1.2, 1.5], ...
    'ultra',    [1.01, 1.1, 1.2, 1.5]);

defaultParams.gamma_test_every_n_iters = 10;
defaultParams.gamma_test_counter_init = 9;

defaultParams.gamma_test_maximal_x_distance_abs = 1e-6;
defaultParams.gamma_test_maximal_x_distance_rel = 1e-6;

% Reducing this threshold below 1e-3 did not seem to make any improvement
% in performance or increase running time.
defaultParams.w_diff_thresh_to_keep_increasing_gamma = ...
    chooseByKeyStr(profile, 'fast', 1e-2, 'normal', 1e-3, 'thorough', 1e-4, 'ultra', 1e-4);

% Stop solving more instances of P_{lambda,gamma} when this number of sparse
% solutions is reached.
defaultParams.nGammas_sparse_x_to_stop = chooseByKeyStr(profile, ...
    'fast', 2, 'normal', 3, 'thorough', 5, 'ultra', 10);

defaultParams.nGammas_sparse_w_to_stop = ...
    chooseByKeyStr(profile, 'fast', 2, 'normal', 2, 'thorough', 4, 'ultra', 6);

% Upper bound for gamma/lambda.
% Suppose that gamma_over_lambda_upper_bound = (1/delta) * log(nchoosek(d,k)) / (0.5*norm(y)^2).
% Then we are guaranteed that if gamma >= lambda * gamma_over_lambda_upper_bound,
% F_k(x) <= F_{k,gamma}(x) <= F_k(x) + delta * F_k(0)
% for any x.
% When gamma/lambda reaches above this threshold, gamma is set to inf.
defaultParams.gamma_over_lambda_upper_bound = 1e16 * lognchoosek(size(A,2),k) / (0.5*norm(y)^2);

defaultParams.nGammaVals_max = 10000;


%% Parameters related to F_{lambda,gamma}(x)
%  -----------------------------------------

% Energy decrease threshold when solving for a single gamma
defaultParams.Flg_minimal_decrease = ...
    chooseByKeyStr(profile, 'fast', 1e-2, 'normal', 1e-3, 'thorough', 1e-4, 'ultra', 1e-5);

% Energy decrease threshold when solving for a single gamma
defaultParams.Flg_minimal_decrease_immediate = 1e-6;

% When solving for one value of gamma, number of consecutive non-decreases of
% the energy in order to conclude convergence.
defaultParams.Flg_num_small_decreases_for_stopping = ...
    chooseByKeyStr(profile, 'fast', 2, 'normal', 2, 'thorough', 2, 'ultra', 3);

% Same as above, but when using the final w
defaultParams.Flg_num_small_decreases_for_stopping_on_infinite_gamma = ...
    chooseByKeyStr(profile, 'fast', 3, 'normal', 5, 'thorough', 6, 'ultra', 6);

% Maximal iterations over different w's when solving for one value of gamma
defaultParams.Flg_max_num_mm_iters = 1000;


%% Parameters related to F_{lambda,w}(x)
%  -------------------------------------

% Settings to use when solver is set to yalmip
defaultParams.yalmip_settings = [];

% Energy decrease threshold when solving (P_lambda,w) by Gradient Descent
defaultParams.Flw_fista_minimal_decrease = ...
    chooseByKeyStr(profile, 'fast', 1e-5, 'normal', 1e-5, 'thorough', 1e-6, 'ultra', 1e-6);

defaultParams.fista_monitor_decrease_every_nIter = 3;

% When solving (P_lambda,w) by Gradient Descent, number of consecutive
% non-decreases of the energy in order to stop
defaultParams.Flw_fista_num_small_decreases_for_stopping = ...
    chooseByKeyStr(profile, 'fast', 2, 'normal', 2, 'thorough', 3, 'ultra', 4);

defaultParams.nIterMax_fista = 20000;

%% Escaping ambiguous points and postprocessing
%  --------------------------------------------

defaultParams.escape_ambiguous_points = true;

% 'gradient' or 'ls' least-squares
defaultParams.ambiguous_escape_strategy = ...
    chooseByKeyStr(profile, 'fast', 'gradient', 'normal', 'ls', 'thorough', 'ls', 'ultra', 'ls');

defaultParams.postprocess_by_omp = ...
    chooseByKeyStr(profile, 'fast', false, 'normal', false, 'thorough', true, 'ultra', true);

defaultParams.omp_max_num_atoms_to_complete = ...
    chooseByKeyStr(profile, 'fast', 3, 'normal', 5, 'thorough', 10, 'ultra', inf);


%% Other parameters
%  ----------------

% Maximal number of threads to use in optimization. Note: This has no
% effect when using yalmip as the solver. For this to take effect, the
% parameter needs to pass to yalmip nanually in yalmip_settings.
defaultParams.maxNumThreads = 2;



%% Undocumented parameters

% An iterate x is considered k-sparse if |x|_(k+1) <= <sparsityThreshAbs_x> * |x|_(k).
% See also <nLambdas_sparse_x_to_stop>.
defaultParams.sparsityThreshAbs_x = chooseByKeyStr(profile, ...
    'fast', 1e-5, 'normal', 1e-5, 'thorough', 1e-6, 'ultra', 1e-6);

defaultParams.sparsityThreshAbs_w = 1e-5;


defaultParams.use_golden_search = false;

% Chooses the solver we use for problems of the form
% min (1/p)*norm(A*x-y)^p + sum(w.*abs(x))
% Can be: 'mosek', 'gurobi', 'yalmip', 'fista', or 'debug'
% The 'debug' setting uses MOSEK but compares all methods to it.

% Optimization solver to use
defaultParams.solver = baseParams.solver;



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
defaultParams.reportThresh_wSol_fista_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_gurobi_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_quadprog_inferior_to_yalmip = 0;
defaultParams.reportThresh_wSol_quadprog_inferior_to_yalmip = 0;

defaultParams.reportThresh_wSol_energy_increase_majorizer_decrease = 0;
defaultParams.reportThresh_wSol_majorizer_increase = 0;
defaultParams.reportThresh_wSol_nonZero_residual_for_small_lambda = 1e-6;
defaultParams.reportThresh_abs_wSol_nonSparse_solution_for_large_lambda = 1e-6;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'getTimeStr.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sOut = getTimeStr(t)
% Returns a string of the format HH:MM:SS.FFF describing the time given in
% t. t should be given in seconds.

if isnan(t)
    sOut = 'nan';
    return
end

sOut = sprintf('%s', datestr(t/24/60/60,'HH:MM:SS.FFF'));

if strcmp(sOut(1:3), '00:')
    sOut = sOut(4:end);
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'get_A_data.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

if numel(v) >= n
    A_data.gamma_n   = v(n);
else
    A_data.gamma_n = 0;
end

A_data.maxColNorm = sqrt(max(sum(A.^2,1)));

A_data.lambdaSmall_p1 = A_data.gamma_n/sqrt(d-k);
A_data.lambdaLarge_p1 = A_data.maxColNorm;

A_data.lambdaSmallMultiplier_p2 = A_data.gamma_n/sqrt(d-k);
A_data.lambdaLargeMultiplier_p2 = A_data.maxColNorm;

% These will be used for optimization
A_data.A = A;
A_data.norm_A = norm(A);

%A_data.AtA = A'*A;

%A_data.NA = null(A);
%A_data.NANAt = A_data.NA*A_data.NA';
A_data.rangeAt = orth(A');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'get_lambda_vals.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lambdaVals = get_lambda_vals(params, A_data, norm_y)
%function lambdaVals = get_lambda_vals(params, A_data, norm_y)
%
% Determines the values of lambda to be used for solving P0.

%stepVec_tan = @(a,b,nPoints) b * tan(atan(a/b) + (0:(nPoints-1))*(atan(1)-atan(a/b))/(nPoints-1));

if ~isempty(params.lambdaRelVals)
    if params.residualPower == 2
        % If the user overrides the lambda values, use his override.
        lambdaVals = (A_data.lambdaLargeMultiplier_p2 * norm_y) .* params.lambdaRelVals;
    elseif params.residualPower == 1
        lambdaVals = lambdaLarge_p1 .* params.lambdaRelVals;
    end
elseif ~isempty(params.lambdaVals)
    % If the user overrides the lambda values, use his override.
    lambdaVals = params.lambdaVals;
else
    % We need to determine the values of lambda automatically.
    nVals = params.nLambdaVals;
    
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
        
        lambdaVals = exp(stepVec_pwlin(log(params.lambdaRel_min*A_data.lambdaLargeMultiplier_p2*norm_y),nVals-1,log(A_data.lambdaLargeMultiplier_p2 * norm_y), nVals));
        lambdaVals(end) = (1+1e-8)*lambdaVals(end);
        
        % Older versions
        %lambdaVals = (stepVec_tan(sqrt(1e-4*A_data.lambdaLargeMultiplier_p2*norm_y),sqrt(A_data.lambdaLargeMultiplier_p2 * norm_y), nVals)).^2;
        %lambdaVals = exp(stepVec_pwlin(log(1e-4*A_data.lambdaLargeMultiplier_p2*norm_y),nVals-1,log(A_data.lambdaLargeMultiplier_p2 * norm_y), nVals));
    end
    
    lambdaVals = sort(lambdaVals);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'goldenSearch.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'lognchoosek.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = lognchoosek(n,k)
if 2*k > n
    out = lognchoosek(n,n-k);
    return
end

v1 = (n-k+1):n;
v2 = k:-1:1;

out = sum(log(v1./v2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'makeOptData.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function optData = makeOptData(A, y, k, A_data, params)
%% Pre-calculations done as a preprocessing step for optimization speedup
[n,d] = size(A);
optData = struct();

optData.A_data = A_data;

optData.scale_factor_x = norm(y) / sqrt(max(sum(A.^2,1)));

if ~strcmp(params.solver,'fista')
    optData.A_sparse    = sparse(A);
    optData.A_aug       = [optData.A_sparse; sparse(d,d)];
    optData.A_aug_wInds = find([zeros(n,d); eye(d)]);
    optData.y_aug       = sparse([y; zeros(d,1)]);
end

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

if ~strcmp(params.solver,'fista')
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
end

%% Verify solver names
if ~any(strcmp(params.solver, {'mosek', 'gurobi', 'quadprog', 'fista', 'yalmip', 'debug'}))
    error('Invalid solver name ''%s'' given in params.solver', params.socpSolver);
end

%% Determine which solvers we need
need_yalmip = any(strcmp(params.solver, {'yalmip', 'debug'}));
need_mosek = strcmp(params.solver, 'mosek') || (strcmp(params.solver, 'debug') && params.debug_mosek);
need_gurobi = strcmp(params.solver, 'gurobi') || (strcmp(params.solver, 'debug') && params.debug_gurobi);
need_quadprog = strcmp(params.solver, 'quadprog') || (strcmp(params.solver, 'debug') && params.debug_quadprog);
need_gd = strcmp(params.solver, 'fista') || (strcmp(params.solver, 'debug') && params.debug_gd);

%% Parameters and options for various solvers

if need_gd
    % lsqlin options
    %lsqlin_options = optimoptions('lsqlin');
    %lsqlin_options.Algorithm = 'active-set';
    %optData.lsqlin_options = lsqlin_options;
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
end


%% Yalmip
if need_yalmip
    basic_yalmipSettings    = params.yalmip_settings;
    
    if isempty(basic_yalmipSettings)
        basic_yalmipSettings = sdpsettings('verbose', 0, 'cachesolvers', 1);
    end
    
    socp_yalmipSettings     = basic_yalmipSettings;
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
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'maxDiffOfSums.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = maxDiffOfSums(x,k)
% Returns the difference of the smallest and largest sums of absolute
% values of d-k terms of x.
d = numel(x);
x = sort(abs(x));
out = sum(x(k+1:d)) - sum(x(1:d-k));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'merge_files.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function merge_files(fnames, outfname)
%function merge_files(fnames, outfname)
%
% Merges several text input files to an output file.
%
% Example: 
% >> merge_files({'file1.txt','file2.txt','file3.txt'}, 'out.txt');

if ~iscell(fnames) || isempty(fnames) || (numel(fnames) == 0) || ~all(cellfun(@ischar, fnames))
    error('fnames must be a nonempty cell array of strings');
end

n = numel(fnames);

for i=1:n
    fname = fnames{i};
    
    if ~exist(fname, 'file')
        error('File ''%s'' does not exist', fname);
    end
end

f = fopen(outfname, 'w');

for i=1:n
    fname = fnames{i};
    
    text = fileread(fname);
    
    fwrite(f, text);
    fprintf(f, '\n');
end

fclose(f);

%fprintf('\nSuccessfully saved ''%s''\n', outfname);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'minimize_f_on_support.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x_out = minimize_f_on_support(supp,A,y)
x_out = zeros(size(A,2),1);
x_out(supp) = A(:,supp)\y;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'namevals2struct.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = namevals2struct(namevals)
%function s = namevals2struct(v)
%
% This function takes a cell array that contains name/value pairs and
% returns a struct defined by these names and values.
%
% Example:
% >> s = namevals2struct({'a',5,'b',6,'c',[],'d',[1,2,3]});
%
% s = 
%
%    a: 5
%    b: 6
%    c: []
%    d: [1 2 3]

if iscell(namevals) && numel(namevals) == 1 && isstruct(namevals{1})
    s = namevals{1};
    return
elseif ~iscell(namevals) && numel(namevals) == 1 && isstruct(namevals)
    s = namevals;
    return
elseif numel(namevals) == 0
    s = struct();
    return
elseif numel(namevals) == 1
    namevals = namevals{1};
end

if mod(numel(namevals),2) ~= 0
    error('v must contain name/value pairs');
end

n = numel(namevals)/2;

names = namevals(1:2:2*n-1);
values = namevals(2:2:2*n);

if ~all(cellfun(@isstr, names))
    error('v must contain name/value pairs');
end

if numel(unique(names)) ~= n
    error('Parameter names must not repeat');
end

if any(cellfun(@isempty, names))
    error('Parameter names cannot be empty strings');
end

s = cell2struct(values,names,2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'nonSparsityAbs.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = nonSparsityAbs(x,k, scale_factor_x)
%function out = nonSparsityAbs(x, k, scale_factor_x)
%
% Tells how far a vector x is from being k-sparse, in terms of its (k+1)st
% largest-magnitude entry.
%
% scale_factor_x - A factor to divide the output by, to make it
% scale-invariant. When solving an optimization problem
%
%                 min_x F_lambda(x) = f(x) + lambda tau_k(x),
%
% the value norm(x)/scale_factor_x should be scale-invariant, namely,
% that if x0 is some local minimum of f(x), and z0=x0/b is a local minimum of 
% g(z) := a*f(b*z), then
% x0 / (scale_factor_x of f) = z0 / (scale_factor_x of g)
%
% For example,
% if f(x) = 0.5*norm(A*x-y)^2, then the following is a good choice:
% scale_factor_x = (l2-norm of y) / (maximal l2-norm of column of A)
%
% For a general f, any of the following two is ok:
% scale_factor_x = sqrt(2*f(0)) / sqrt(Lipschitz constant of grad(f))
% or,
% scale_factor_x = sqrt(2*f(0)) / sqrt(max diagonal entry of Hessian(f))

out = min(maxk(abs(x),k+1)) / scale_factor_x;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'nonSparsityRel.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = nonSparsityRel(x,k)
%function out = nonSparsityRel(x,k)
%
% Tells how far a vector x is from being k-sparse, in terms of its (k+1)st
% largest-magnitude entry, relative to the average magnitude of its k
% largest-magnitude entries.

out = min(maxk(abs(x),k+1)) / mean(maxk(abs(x),k));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'omp_complete_support.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, S_out] = omp_complete_support(A, y, k, S_in)
%function [x_out, S_out] = omp_complete_support(A, y, k, S_in)
%
% Given a matrix A of size n x d, and a vector y in R^n, seeks x in R^d
% that minimizes ||Ax-y||_2, such that ||x||_0 <= k and the support of x
% contains the indices given in S_in. Solution is obtained by the
% Least-Squares Orthogonal Matching Pursuit (LS-OMP) algorithm, also known
% as Orthogonal Least Squares.
%
% Output arguments:
% x_out - Solution 
% S_out - The indices of the support of x_out

t_start = tic;

S_in = S_in(:);

[n,d] = size(A);

S_in = unique(S_in);


if (numel(S_in) > k)
    error('numel(S_in) must be lower or equal to k');
elseif (k > min(n,d))
    error('k must be lower or equal to size(A,1) and size(A,2)');
end


if numel(S_in) == k
    S_out = S_in;
    
    x_out = zeros(d,1);
    x_out(S_in) = A(:,S_in) \ y;
    
    return
end


S_curr = S_in;

while numel(S_curr) < k
    I_cand = setdiff(1:d, S_curr);
    
    obj_best = inf;
    
    for i=I_cand
        S_cand = [S_curr; i];
        
        x_cand = zeros(d,1);
        x_cand(S_cand) = A(:,S_cand) \ y;
        
        obj_cand = norm(A*x_cand-y);
        
        if obj_cand < obj_best
            obj_best = obj_cand;
            x_best = x_cand;
            S_best = S_cand;
        end
    end
    
    S_curr = S_best;
end
    
x_out = x_best;
S_out = sort(S_best);

%fprintf('OMP time: %g\n', toc(t_start));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_sol, dbinfo_lambda] = optimize_F_lambda(A, y, k, lambda, x_init, optData, params)
%function [x_sol, dbinfo_lambda] = optimize_F_lambda(A, y, k, lambda, x_init, optData, params)
%
% This function solves the optimization problem (P_lambda) for a single lambda value.
%
% Input arguments:
%
% A, y, k - Dictionary matrix, signal to represent sparsely and sparsity
%           level respectively.
%
% lambda - A nonnegative penalty parameter.
%
% x_init - Initialization. Ignored when set to [].
%
% optData, params - Structs containing the data structures for optimization
%                   and parameters respectively.

[n,d] = size(A);
dbinfo_lambda = struct();

%TODO: Parametrize this
% Used for debugging purposes. TODO: Control this by a parameter and return
% in output.
% Tells whether to keep track of several measures throughout iterations,
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

gamma_upper_bound = lambda * params.gamma_over_lambda_upper_bound;

if params.monitor_violations
    %% Initialize violation monitors
    violationMax_gammaSol_solution_inferior_to_init = 0;
    
    violationMax_wSol_solution_inferior_to_init = 0;
    violationMax_wSol_solution_inferior_to_projSol = 0;
    violationMax_gammaSol_energy_increase_majorizer_decrease = 0;
    violationMax_abs_gammaSol_gsm_larger_than_majorizer = 0;
    
    violationMax_wSol_mosek_inferior_to_yalmip = 0;
    violationMax_wSol_gurobi_inferior_to_yalmip = 0;
    violationMax_wSol_quadprog_inferior_to_yalmip = 0;
    violationMax_wSol_fista_inferior_to_yalmip = 0;
    
    violationMax_wSol_energy_increase_majorizer_decrease = 0;
    violationMax_wSol_majorizer_increase = 0;
    
    violationMax_wSol_nonZero_residual_for_small_lambda = 0;
    violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;
end

% Used to count the number of gammas for which the iterate x and weight
% vector w are k-sparse.
sparse_x_counter = 0;
sparse_w_counter = 0;

% Debug information
num_gammas    = 0;
num_ws        = 0;
nIter_w_solver = 0;
tElapsed_w_solver = 0;
tElapsed_w_goldenSearch = 0;

report_gammas = false;

x_curr = [];
w_curr = [];
gamma_curr = [];

gsm_curr = [];
energy_gamma_curr = inf;

x_best = [];

energy_lambda_best = inf;


% ksupp_curr keeps track of the k largest-magnitude entries of x.
ksupp_curr = [];

% Iteration counter
i = 0;

gamma_growth_factors = params.gamma_growth_factors;
i_growth_factor = 1;

% Turn this on when the gamma in the next iteration should be set to infty
next_gamma_is_infty = false;

% Function shortcuts
Fkg = @(x,gamma) calc_F_lambda_gamma(A, x, y, k, lambda, gamma, params.residualPower);
Fk  = @(x) calc_F_lambda_gamma(A, x, y, k, lambda, inf, params.residualPower);

if ~isempty(x_init)
    Fkg_init = @(gamma) Fkg(x_init,gamma);
else
    Fkg_init = @(gamma) inf;
end

while isempty(gamma_curr) || (gamma_curr ~= inf)
    i = i+1;
    
    %% Backup the previous iterate and weight vector
    w_prev = w_curr;
    x_prev = x_curr;

    % Greedily improved solution. See below.
    x_greedy = [];
    
    % There's no current iterate yet
    x_curr = [];    
    gamma_prev = gamma_curr;
    
    % Determine new gamma
    if isempty(gamma_prev)
        gamma_curr = 0;
        
    elseif gamma_prev == 0
        % In the previous iteration we solved for gamma=0. Now we need to
        % determine the first nonzero gamma.
        
        if isempty(x_init) || params.full_homotopy_with_init
            % If we need to do a full homotopy (no init, or init with full homotopy),
            % determine gamma_1 by the maxdiff rule.
            maxDiff_prev = maxDiffOfSums(x_prev, k);
            
            if maxDiff_prev == 0
                gamma_curr = inf;
            else
                gamma_curr = params.gamma_first_max_difference_of_sums / maxDiff_prev;
            end
        else
            % Here we are using x_init with a partial homotopy, so we seek
            % the smallest gamma for which x_init is better than the
            % solution obtained for gamma=0.
            gamma_a = 0;
            gamma_b = optData.scale_factor_x;
            
            % Here we know that Fkg(x_prev,gamma_a) <= Fkg(x_init,gamma_a)
            if Fkg(x_prev,inf) < Fkg_init(inf)
                gamma_curr = inf;
            else
                while Fkg(x_prev,gamma_b) < Fkg_init(gamma_b)
                    gamma_a = gamma_b;
                    gamma_b = min(gamma_b * 10, gamma_upper_bound);
                    
                    if gamma_a >= gamma_upper_bound
                        break
                    end
                end
                
                while gamma_b > params.gamma_binary_search_uncertainty_ratio * gamma_a
                    gamma_test = gamma_a + 0.5*(gamma_b - gamma_a);
                    
                    if Fkg(x_prev,gamma_test) < Fkg_init(gamma_test)
                        gamma_a = gamma_test;
                    else
                        gamma_b = gamma_test;
                    end
                end
                
                if gamma_b < gamma_upper_bound
                    gamma_curr = gamma_b;
                else
                    gamma_curr = inf;
                end
            end            
        end
        
    elseif next_gamma_is_infty || (gamma_prev >= gamma_upper_bound)
        gamma_curr = inf;
        
    else
        % Here we need to determine gamma by a previous nonzero gamma. We
        % do so by a set of update rules.
        
        % By default, we should increase gamma
        should_increase_gamma = true;
        
        % If the obtained solution for the previous gamma was k-sparse, try
        % to greedily improve the residual.
        if (sparse_x_counter >= 1) && params.escape_ambiguous_points            
            [x_greedy, is_improved, improvement_rel] = seek_greedy_improvement(x_prev, A, y, k, lambda, optData, params);
            %fprintf('Gamma: %g\t Improved: %g\t Improvement: %g\n', gamma_curr, is_improved, improvement_rel);
            
            % If the new solution is superior to the current one in terms
            % of F_{k,inf}, seek the smallest gamma for which this
            % superiority holds in terms of F_{k,gamma}.
            if ~is_improved
                x_greedy = [];
            else
                should_increase_gamma = false;
                sparse_x_counter = 0;
                
                % Here we know that Fkg(x_greedy,inf) < Fkg(x_prev,inf).                
                % Seek the smallest gamma for which x_greedy (or x_init, 
                % if available) is better than x_prev.
                gamma_a = gamma_prev;
                gamma_b = gamma_prev;
                
                while Fkg(x_prev,gamma_b) < min(Fkg(x_greedy,gamma_b), Fkg_init(gamma_b))
                    gamma_a = gamma_b;
                    gamma_b = min(gamma_b * 10, gamma_upper_bound);
                    
                    if gamma_a >= gamma_upper_bound
                        break
                    end
                end
                
                while gamma_b > params.gamma_binary_search_uncertainty_ratio * gamma_a
                    gamma_test = gamma_a + 0.5*(gamma_b - gamma_a);
                    
                    if Fkg(x_prev, gamma_test) < min(Fkg(x_greedy,gamma_test), Fkg_init(gamma_test))
                        gamma_a = gamma_test;
                    else
                        gamma_b = gamma_test;
                    end
                end
                
                gamma_curr = gamma_b;
                
                if gamma_curr >= gamma_upper_bound
                    gamma_curr = inf;
                end
            end
        end
        
        % Try increasing gamma by the current growth factor.
        while should_increase_gamma
            % If increasing gamma to inf does not change w much, go
            % straight to inf.
            [~, ~, w_test] = Fkg(x_prev, inf);
            
            if max(abs(w_test-w_prev)) < (d-k)/d * params.w_diff_thresh_to_keep_increasing_gamma
                gamma_curr = inf;
                break
            end

            gamma_curr = gamma_prev * gamma_growth_factors(i_growth_factor);
            
            % If the new weight vector is still close to w_prev, keep
            % increasing gamma_curr
            gamma_a = gamma_curr;
            gamma_b = gamma_curr;

            while gamma_a < gamma_upper_bound
                [~, ~, w_test] = Fkg(x_prev, gamma_b);

                if max(abs(w_test-w_prev)) < (d-k)/d * params.w_diff_thresh_to_keep_increasing_gamma
                    gamma_a = gamma_b;
                    gamma_b = min(gamma_b * 10, gamma_upper_bound);
                    %fprintf('gamma_a: %g, gamma_b: %g\n', gamma_a, gamma_b);
                else
                    break
                end
            end
            
            if gamma_a >= gamma_upper_bound
                gamma_curr = inf;
                break
            end
            
            while gamma_b > params.gamma_binary_search_uncertainty_ratio * gamma_a
                gamma_test = gamma_a + 0.5*(gamma_b-gamma_a);
                [~, ~, w_test] = Fkg(x_prev, gamma_test);
                
                if max(abs(w_test-w_prev)) < (d-k)/d * params.w_diff_thresh_to_keep_increasing_gamma
                    gamma_a = gamma_test;
                else
                    gamma_b = gamma_test;
                end
            end
            
            gamma_curr = gamma_a;
            
            % Now we have a new candidate gamma_curr. Solve (P_lambda,gamma)
            % with that gamma_curr. First, apply init if superior to previous iterate.
            if ~isempty(x_init) && (Fkg_init(gamma_curr) < Fkg(x_prev,gamma_curr))
                x_init_cand = x_init;
            else
                x_init_cand = x_prev;
            end
            
            [x_curr, energy_gamma_curr, gsm_curr, w_curr, dbinfo_gamma_curr] = optimize_F_lambda_gamma(A, y, k, lambda, gamma_curr, x_init_cand, optData, params);
            
            num_gammas = num_gammas + 1;
            num_ws = num_ws + dbinfo_gamma_curr.num_ws;
            nIter_w_solver = nIter_w_solver + dbinfo_gamma_curr.nIter_w_solver;
            tElapsed_w_solver = tElapsed_w_solver + dbinfo_gamma_curr.tElapsed_w_solver;
            tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_gamma_curr.tElapsed_w_goldenSearch;
            
            if (norm(x_curr-x_prev, inf) <= params.gamma_test_maximal_x_distance_abs) || ...
                    (norm(x_curr-x_prev, inf) <= norm(x_prev,inf)*params.gamma_test_maximal_x_distance_rel)
                % Here the resulting iterate is close
                
                if i_growth_factor == numel(gamma_growth_factors)
                    % If movement is small with the largest growth factor,
                    % keep the new iterate
                    break
                else
                    % If movement is small and there are larger growth
                    % factors to try, next time try the next growth factor
                    i_growth_factor = i_growth_factor + 1;
                    break
                end
            else
                % If stepped far...
                if i_growth_factor == 1
                    % ... using the smallest growth factor, keep the
                    % change.
                    break
                else
                    % Otherwise, try again with a smaller growth factor.
                    i_growth_factor = i_growth_factor - 1;
                    i_growth_factor = 1;
                    %TODO: Check which of these two works better
                    continue;
                end
            end
        end
    end
    
    % If we still didn't calculate x_curr, do so now.
    if isempty(x_curr)
        % Choose the best among x_prev, a_greedy, x_init for the current
        % initialization.
        if ~isempty(x_greedy) && ...
                Fkg(x_greedy, gamma_curr) <= min(Fkg(x_prev, gamma_curr), Fkg_init(gamma_curr))
            x_init_curr = x_greedy;
        elseif ~isempty(x_prev) && (Fkg(x_prev, gamma_curr) < Fkg_init(gamma_curr))
            x_init_curr = x_prev;
        else
            x_init_curr = x_init;
        end
        
        [x_curr, energy_gamma_curr, gsm_curr, w_curr, dbinfo_gamma_curr] = optimize_F_lambda_gamma(A, y, k, lambda, gamma_curr, x_init_curr, optData, params);
        num_gammas = num_gammas + 1;
        num_ws = num_ws + dbinfo_gamma_curr.num_ws;
        nIter_w_solver = nIter_w_solver + dbinfo_gamma_curr.nIter_w_solver;
        tElapsed_w_solver = tElapsed_w_solver + dbinfo_gamma_curr.tElapsed_w_solver;
        tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_gamma_curr.tElapsed_w_goldenSearch;
        
        % If we just solved for gamma=inf, try solving again with the
        % given initialization and pick the better solution.
        if (gamma_curr == inf) && ~isempty(x_init) && any(x_init_curr ~= x_init)
            [x_curr2, energy_gamma_curr2, gsm_curr2, w_curr2, dbinfo_gamma_curr2] = optimize_F_lambda_gamma(A, y, k, lambda, gamma_curr, x_init, optData, params);
            num_gammas = num_gammas + 1;
            num_ws = num_ws + dbinfo_gamma_curr2.num_ws;
            nIter_w_solver = nIter_w_solver + dbinfo_gamma_curr2.nIter_w_solver;
            tElapsed_w_solver = tElapsed_w_solver + dbinfo_gamma_curr2.tElapsed_w_solver;
            tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_gamma_curr2.tElapsed_w_goldenSearch;
            
            if energy_gamma_curr2 < energy_gamma_curr
                x_curr = x_curr2;
                energy_gamma_curr = energy_gamma_curr2;
                gsm_curr = gsm_curr2;
                w_curr = w_curr2;
                dbinfo_gamma_curr = dbinfo_gamma_curr2;
            end
        end
    end
    
    % Here we have a new iterate x_curr, which is an estimated solution
    % of (P_lambda,gamma) with gamma=gamma_curr.
    
    % Measure the cardinality of the new iterate
    %s_curr = calc_numerical_l0norm(x_curr, params.looseSparsityThreshAbs_x, params.looseSparsityThreshRel_x);
    
    % Check ideal energy of x_curr and keep it if it is the best so far
    energy_lambda_curr = Fk(x_curr);
    
    xProj_curr = projectVec(x_curr, A, y, k);
    energyProj_lambda_curr = Fk(xProj_curr);
    
    % Update best solution to (P_lambda) seen so far, using both x_curr and
    % its k-sparse projection with respect to A,y.
    [x_best, energy_lambda_best] = update_min(x_best, energy_lambda_best, x_curr, energy_lambda_curr);
    [x_best, energy_lambda_best] = update_min(x_best, energy_lambda_best, xProj_curr, energyProj_lambda_curr);
    
    % If we are at gamma=inf, run one more optimization of P_lambda, this
    % time initialized by x_best, which might have been obtained at an
    % earlier gamma.
    if (gamma_curr == inf) && any(x_best ~= x_curr)
        [x_greedy, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = optimize_F_lambda_gamma(A, y, k, lambda, gamma_curr, x_best, optData, params);
        num_gammas = num_gammas + 1;
        num_ws = num_ws + dbinfo_gamma_temp.num_ws;
        nIter_w_solver = nIter_w_solver + dbinfo_gamma_temp.nIter_w_solver;
        tElapsed_w_solver = tElapsed_w_solver + dbinfo_gamma_temp.tElapsed_w_solver;
        tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_gamma_temp.tElapsed_w_goldenSearch;
        
        % This comparison is ok since here gamma=inf
        if energy_gamma_temp < energy_lambda_best
            x_curr = x_greedy;
            energy_gamma_curr = energy_gamma_temp;
            gsm_curr = gsm_temp;
            w_curr = w_temp;
            dbinfo_gamma_curr = dbinfo_gamma_temp;
            
            x_best = x_curr;
            energy_lambda_best = energy_gamma_curr;
        end
    end
    
    ksupp_prev = ksupp_curr;
    ksupp_curr = calc_supp(x_curr, k);    
    
    % Here we try to improve the solution by an aggressive OMP
    % post-processing.
    if (gamma_curr == inf) && params.postprocess_by_omp
        t_omp = tic;
        
        for i_omp = max(0, k - params.omp_max_num_atoms_to_complete):(k-1)
            x_omp = omp_complete_support(A, y, k, ksupp_curr(1:i_omp));
            
            %energy_omp = calcEnergy(A, x_omp, y, k, lambda, params.residualPower);
            
            % Optimizing only when OMP improves the objective slightly
            % degrades performance and yields a negligible speedup.
            if true %energy_omp < energy_gamma_curr
                [x_temp, energy_gamma_temp, gsm_temp, w_temp, dbinfo_gamma_temp] = optimize_F_lambda_gamma(A, y, k, lambda, gamma_curr, x_omp, optData, params);
                
                % This comparison is ok since here gamma=inf
                if energy_gamma_temp < energy_lambda_best
                    x_curr = x_temp;
                    energy_gamma_curr = energy_gamma_temp;
                    gsm_curr = gsm_temp;
                    w_curr = w_temp;
                    dbinfo_gamma_curr = dbinfo_gamma_temp;
                    
                    x_best = x_curr;
                    energy_lambda_best = energy_gamma_curr;

                    ksupp_curr = calc_supp(x_curr, k);
               end
            end
        end
        
        dbinfo_lambda.t_omp = toc(t_omp);
    else
        dbinfo_lambda.t_omp = nan;
    end
    
    
    %% Collect debug info
    if keep_track
        track.x = [track.x, x_curr];
        track.w = [track.w, w_curr];
        track.gamma = [track.gamma, gamma_curr];
        track.num_ws = [track.num_ws, dbinfo_gamma_curr.num_ws];
        track.used_init = [track.used_init, used_x_init_now];
    end
    
    
    if params.monitor_violations
        %% Monitor violations
        violationCurr_gammaSol_solution_inferior_to_init = ...
            (dbinfo_gamma_curr.energy_sol - dbinfo_gamma_curr.energy_init) / dbinfo_gamma_curr.energy_init;
        
        violationMax_gammaSol_solution_inferior_to_init = max( ...
            violationCurr_gammaSol_solution_inferior_to_init, ...
            violationMax_gammaSol_solution_inferior_to_init);
        
        violationMax_wSol_solution_inferior_to_init = max( ...
            violationMax_wSol_solution_inferior_to_init, ...
            dbinfo_gamma_curr.violationMax_wSol_solution_inferior_to_init);
        
        violationMax_wSol_solution_inferior_to_projSol = max( ...
            violationMax_wSol_solution_inferior_to_projSol, ...
            dbinfo_gamma_curr.violationMax_wSol_solution_inferior_to_projSol);
        
        violationMax_gammaSol_energy_increase_majorizer_decrease = max( ...
            violationMax_gammaSol_energy_increase_majorizer_decrease, ...
            dbinfo_gamma_curr.violationMax_gammaSol_energy_increase_majorizer_decrease);
        
        violationMax_abs_gammaSol_gsm_larger_than_majorizer = max( ...
            violationMax_abs_gammaSol_gsm_larger_than_majorizer,...
            dbinfo_gamma_curr.violationMax_abs_gammaSol_gsm_larger_than_majorizer);
        
        violationMax_wSol_mosek_inferior_to_yalmip = max( ...
            violationMax_wSol_mosek_inferior_to_yalmip, ...
            dbinfo_gamma_curr.violationMax_wSol_mosek_inferior_to_yalmip);
        
        violationMax_wSol_gurobi_inferior_to_yalmip = max( ...
            violationMax_wSol_gurobi_inferior_to_yalmip, ...
            dbinfo_gamma_curr.violationMax_wSol_gurobi_inferior_to_yalmip);
        
        violationMax_wSol_quadprog_inferior_to_yalmip = max( ...
            violationMax_wSol_quadprog_inferior_to_yalmip, ...
            dbinfo_gamma_curr.violationMax_wSol_quadprog_inferior_to_yalmip);
        
        violationMax_wSol_fista_inferior_to_yalmip = max( ...
            violationMax_wSol_fista_inferior_to_yalmip, ...
            dbinfo_gamma_curr.violationMax_wSol_fista_inferior_to_yalmip);
        
        violationMax_wSol_energy_increase_majorizer_decrease = max( ...
            violationMax_wSol_energy_increase_majorizer_decrease, ...
            dbinfo_gamma_curr.violationMax_wSol_energy_increase_majorizer_decrease);
        
        violationMax_wSol_majorizer_increase = max( ...
            violationMax_wSol_majorizer_increase, ...
            dbinfo_gamma_curr.violationMax_wSol_majorizer_increase);
        
        violationMax_wSol_nonZero_residual_for_small_lambda = max( ...
            violationMax_wSol_nonZero_residual_for_small_lambda, ...
            dbinfo_gamma_curr.violationMax_wSol_nonZero_residual_for_small_lambda);
        
        violationMax_abs_wSol_nonSparse_solution_for_large_lambda = max( ...
            violationMax_abs_wSol_nonSparse_solution_for_large_lambda, ...
            dbinfo_gamma_curr.violationMax_abs_wSol_nonSparse_solution_for_large_lambda);
    end
    
    %% Loop control & gamma update
    
    % If we have just used the final gamma in this iteration, break.
    %if (gamma_curr == inf)
    %    break
    %end
    
    
    %% Reporting
    if report_gammas
        relP0Obj = norm(A*projectVec(x_curr,A,y,k)-y) / norm(y);
        nonSparsity_x_curr = nonSparsityRel(x_curr,k);
        nIterw_curr = dbinfo_gamma_curr.num_ws;
        
        if ~isempty(w_prev) && ~isempty(x_prev)
            maxWDist = norm(w_curr-w_prev,inf);
            x_dist_curr = norm(x_curr-x_prev,1)/norm(x_prev,1);
        else
            maxWDist = nan;
            x_dist_curr = nan;
        end
        
        %fac_curr = gamma_growth_factors(i_growth_factor);    
        if ~isempty(gamma_prev)
            fac_curr = gamma_curr / gamma_prev;
        else
            fac_curr = inf;
        end
                
        fprintf('%d: Gamma: %g  Fac: %g  x-dist: %g  nIter w: %d  w-dist: %g    x-tail: %g  relP0Obj = %g\n', num_gammas, gamma_curr, fac_curr, x_dist_curr, nIterw_curr, maxWDist, nonSparsity_x_curr, relP0Obj);
        %fprintf('%d: Gamma: %g  nIter w: %d  w-dist: %g  x-dist: %g  w-tail: %g  x-tail: %g  relP0Obj = %g\n', num_gammas, gamma_curr, nIterw_curr, maxWDist, x_dist_curr, nonSparsity_curr, nonSparsity_x_curr, relP0Obj);
    end
    
    %% Loop control
    
    % If the weight vector is accutely biased towards the k largest
    % entries, there is no reason to run more iterations with larger
    % values of gamma, which would only make it more accute.
    if nonSparsityAbs(w_curr, d-k, 1) > params.sparsityThreshAbs_w
        sparse_w_counter = 0;
    else
        sparse_w_counter = sparse_w_counter + 1;
        
        % If w turned out to be sparse a number of times in a row, jump
        % to the final gamma.
        if sparse_w_counter >= params.nGammas_sparse_w_to_stop
            next_gamma_is_infty = true;
        end
    end
    
    % If the iterate x_curr is k-sparse for a given number of
    % consecutive iterations, and its sparsity pattern remains fixed,
    % we assume that the pattern will remain fixed for all the
    % remaining gamma values, and therefore skip to gamma=inf.
    
    if (nonSparsityAbs(x_curr, k, optData.scale_factor_x) <= params.sparsityThreshAbs_x) && ...
            (~isempty(ksupp_prev) && (numel(intersect(ksupp_curr, ksupp_prev)) == k))
        sparse_x_counter = sparse_x_counter + 1;
        
        % If x turned out to be sparse a number of times in a row, jump
        % to the final gamma.
        if (sparse_x_counter >= params.nGammas_sparse_x_to_stop) 
            next_gamma_is_infty = true;
        end
    else
        sparse_x_counter = 0;
    end
    
    % If we reached the iteration number limit, run a last iteration with
    % gamma = inf.
    if num_gammas >= params.nGammaVals_max - 1
        next_gamma_is_infty = true;
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


if params.monitor_violations
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
    dbinfo_lambda.violationMax_wSol_fista_inferior_to_yalmip = violationMax_wSol_fista_inferior_to_yalmip;
    
    dbinfo_lambda.violationMax_wSol_energy_increase_majorizer_decrease = violationMax_wSol_energy_increase_majorizer_decrease;
    dbinfo_lambda.violationMax_wSol_majorizer_increase = violationMax_wSol_majorizer_increase;
    
    dbinfo_lambda.violationMax_wSol_nonZero_residual_for_small_lambda = violationMax_wSol_nonZero_residual_for_small_lambda;
    dbinfo_lambda.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = violationMax_abs_wSol_nonSparse_solution_for_large_lambda;    
end

dbinfo_lambda.num_gammas        = num_gammas;
dbinfo_lambda.num_ws            = num_ws;
dbinfo_lambda.nIter_w_solver     = nIter_w_solver;
dbinfo_lambda.tElapsed_w_solver = tElapsed_w_solver;
dbinfo_lambda.tElapsed_w_goldenSearch = tElapsed_w_goldenSearch;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_gamma.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, energy_gamma_out, gsm_out, w_curr, dbinfo_gamma] = optimize_F_lambda_gamma(A, y, k, lambda, gamma, x_init, optData, params)
%function [x_out, energy_gamma_out, gsm_out, w_next, dbinfo_gamma] = optimize_F_lambda_gamma(A, y, k, lambda, gamma, x_init, optData, params)
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

% Number of non-decrease iterations for stopping. A stricter parameter is
% optionally used for gamma=inf
if gamma == inf
    num_non_decreases_for_stopping = params.Flg_num_small_decreases_for_stopping_on_infinite_gamma;
else
    num_non_decreases_for_stopping = params.Flg_num_small_decreases_for_stopping;
end

if ~isempty(x_init)
    x_curr = x_init;
    [Flg_curr, tau_curr, w_curr] = calc_F_lambda_gamma(A, x_curr, y, k, lambda, gamma, params.residualPower);
    w_init = w_curr;
    energy_gamma_init = Flg_curr;
else
    x_curr = [];
    tau_curr = nan;
    Flg_curr = inf;
    w_curr = (d-k)/d * ones(d,1);
    energy_gamma_init = inf;
end

if params.monitor_violations
    %% Initialize vilation monitors
    dbinfo_gamma.violationMax_wSol_solution_inferior_to_init = 0;
    dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol = 0;
    dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease = 0;
    dbinfo_gamma.violationMax_abs_gammaSol_gsm_larger_than_majorizer = 0;
    
    dbinfo_gamma.violationMax_wSol_mosek_inferior_to_yalmip = 0;
    dbinfo_gamma.violationMax_wSol_gurobi_inferior_to_yalmip = 0;
    dbinfo_gamma.violationMax_wSol_quadprog_inferior_to_yalmip = 0;
    dbinfo_gamma.violationMax_wSol_fista_inferior_to_yalmip = 0;
    
    dbinfo_gamma.violationMax_wSol_energy_increase_majorizer_decrease = 0;
    dbinfo_gamma.violationMax_wSol_majorizer_increase = 0;
    
    dbinfo_gamma.violationMax_wSol_nonZero_residual_for_small_lambda = 0;
    dbinfo_gamma.violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;
end

%% Initialize loop
non_decrease_counter = 0;

% These counters keep track, respectively, of how many weight vectors were used,
% and how many internal iterations were used for optimizing the subproblems
% for all w's in total.
num_ws = 0;
nIter_w_solver = 0;

tElapsed_w_solver = 0;
tElapsed_w_goldenSearch = 0;

should_break = false;

for i=1:params.Flg_max_num_mm_iters
    num_ws = num_ws + 1;
    
    %% Solve current reweighted problem
    w_prev = w_curr;
    x_prev = x_curr;

    [x_curr, dbinfo_w] = optimize_F_lambda_w(A, y, k, lambda, w_prev, x_curr, optData, params);
    
    %% Calculate energy and reweight
    tau_prev = tau_curr;
    energy_gamma_prev = Flg_curr;
    [Flg_curr, tau_curr, w_curr] = calc_F_lambda_gamma(A, x_curr, y, k, lambda, gamma, params.residualPower);
    
    if ~isempty(x_prev)
        energy_w_prev = calc_F_lambda_w(A, x_prev, y, lambda, params.residualPower, w_prev);
    else
        energy_w_prev = inf;
    end
    
    energy_w_curr = calc_F_lambda_w(A, x_curr, y, lambda, params.residualPower, w_prev);
    
    % Linear majorizer of tau_(k,gamma)(x_curr) with respect to x_prev
    if ~isempty(x_prev)
        gsm_curr_majorizer = tau_prev + dot(abs(x_curr)-abs(x_prev), w_prev);
    else
        gsm_curr_majorizer = nan;
    end
    
    % Keep track of running times
    tElapsed_w_solver = tElapsed_w_solver + dbinfo_w.tElapsed_w_solver;
    tElapsed_w_goldenSearch = tElapsed_w_goldenSearch + dbinfo_w.tElapsed_w_goldenSearch;
    nIter_w_solver = nIter_w_solver + dbinfo_w.dbinfo_w_method.nIter_w_solver;    
    
    
    if params.monitor_violations
        %% Record violations
        viol_curr = dbinfo_w.violation_wSol_solution_inferior_to_init;
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_init = max(...
            dbinfo_gamma.violationMax_wSol_solution_inferior_to_init, ...
            viol_curr);
        
        viol_curr = dbinfo_w.violation_wSol_solution_inferior_to_projSol;
        dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol = max(...
            dbinfo_gamma.violationMax_wSol_solution_inferior_to_projSol, ...
            viol_curr);
        
        viol_curr = (Flg_curr - energy_gamma_prev) / energy_gamma_prev * ...
            subplus(sign(energy_w_prev - energy_w_curr));
        dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease = max(...
            dbinfo_gamma.violationMax_gammaSol_energy_increase_majorizer_decrease, ...
            viol_curr);
        
        viol_curr = (tau_curr - gsm_curr_majorizer);
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
            
            viol_curr = dbinfo_w.violation_wSol_fista_inferior_to_yalmip;
            dbinfo_gamma.violationMax_wSol_fista_inferior_to_yalmip = max(...
                dbinfo_gamma.violationMax_wSol_fista_inferior_to_yalmip, ...
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
    end
    
    %% Loop control
    if (gamma == inf) && (all(w_curr == w_prev))
        should_break = true;
    elseif (energy_w_prev - energy_w_curr) <= energy_w_prev * params.Flg_minimal_decrease_immediate
        should_break = true;
    elseif (energy_w_prev - energy_w_curr) <= energy_w_prev * params.Flg_minimal_decrease
        non_decrease_counter = non_decrease_counter + 1;
        
        if non_decrease_counter >= num_non_decreases_for_stopping
            should_break = true;
        end
    else
        non_decrease_counter = 0;
    end
    
    
    if should_break && ~params.escape_ambiguous_points
        break
    elseif should_break
        if gamma ~= inf
            break
        end
        
        %s_curr = calc_numerical_l0norm(x_curr, params.looseSparsityThreshAbs_x, params.looseSparsityThreshRel_x);        
        %fprintf('k=%g, s=%g: ', k, s_curr);
        
        % To escape ambiguous points, try to improve x_curr greedily.
        [x_temp, is_improved, improvement_rel] = seek_greedy_improvement(x_curr, A, y, k, lambda, optData, params);
        %fprintf('Gamma: -%g-\t Improved: %g\t Improvement: %g\n', inf, is_improved, improvement_rel);
        
        [Flg_greedy, tau_greedy, w_greedy] = calc_F_lambda_gamma(A, x_temp, y, k, lambda, gamma, params.residualPower);

        x_curr = x_temp;
        w_curr = w_greedy;
        Flg_curr = Flg_greedy;
        tau_curr = tau_greedy;
        
        if is_improved
            should_break = false;
            non_decrease_counter = 0;
        else
            break
        end
    end
end

x_out = x_curr;
energy_gamma_out = Flg_curr;
gsm_out = tau_curr;

dbinfo_gamma.energy_init = energy_gamma_init;
dbinfo_gamma.energy_sol  = Flg_curr;

dbinfo_gamma.num_ws = num_ws;
dbinfo_gamma.nIter_w_solver = nIter_w_solver;

dbinfo_gamma.tElapsed_w_goldenSearch = tElapsed_w_goldenSearch;
dbinfo_gamma.tElapsed_w_solver = tElapsed_w_solver;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, dbinfo_w] = optimize_F_lambda_w(A, y, k, lambda, w, x_init, optData, params)
%function [x_out, dbinfo_w] = optimize_F_lambda_w(A, y, k, lambda, w, x_init, optData, params)

dbinfo_w = struct();

%% Used in debug mode
energy_w_yalmip = nan;
energy_w_mosek  = nan;
energy_w_gurobi = nan;
energy_w_quadprog    = nan;
energy_w_gd   = nan;

if ~isempty(x_init)
    energy_w_init = calc_F_lambda_w(A, x_init, y, lambda, params.residualPower, w);
else
    energy_w_init = inf;
end

% Time measurement for Golden Search is NaN when we don't use it
t_gs = nan;


switch(params.solver)
    case 'yalmip'
        tic;
        [x_out, dbinfo_w_method] = optimize_F_lambda_w_yalmip(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'mosek'
        tic;
        [x_out, dbinfo_w_method] = optimize_F_lambda_w_mosek(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'gurobi'
        tic;
        [x_out, dbinfo_w_method] = optimize_F_lambda_w_gurobi(A, y, lambda, w, optData, params);
        t = toc;
        
    case 'quadprog'
        tic;
        [x_out, dbinfo_w_method] = optimize_F_lambda_w_quadprog(A, y, lambda, w, x_init, optData, params);
        t = toc;
        
    case 'fista'
        tic;
        
        [x_out, dbinfo_w_method] = optimize_F_lambda_w_fista(A, y, lambda, w, x_init, optData, params);
        t = toc;
        
    case 'debug'
        tic;
        [x_debug_yalmip, dbinfo_w_yalmip] = optimize_F_lambda_w_yalmip(A, y, lambda, w, optData, params);
        t = toc;
        
        if params.use_golden_search
            tic;
            x_debug_yalmip = goldenSearch(A, y, lambda, w, x_init, x_debug_yalmip, params.residualPower);
            t_gs = toc;
        end
        
        % Take the YALMIP solution as the output
        x_out = x_debug_yalmip;
        dbinfo_w_method = dbinfo_w_yalmip;
        
        energy_w_yalmip = calc_F_lambda_w(A, x_debug_yalmip, y, lambda, params.residualPower, w);
        
        % Compare other methods to YALMIP.
        % Results of methods that are not included in this debug session
        % remain empty vectors.
        
        % Mosek
        if params.debug_mosek
            [x_debug_mosek, dbinfo_w_mosek] = optimize_F_lambda_w_mosek(A, y, lambda, w, optData, params);
            
            if params.use_golden_search
                x_debug_mosek = goldenSearch(A, y, lambda, w, x_init, x_debug_mosek, params.residualPower);
            end
            
            energy_w_mosek = calc_F_lambda_w(A, x_debug_mosek, y, lambda, params.residualPower, w);
        end
        
        % Gurobi
        if params.debug_gurobi
            [x_debug_gurobi, dbinfo_w_gurobi] = optimize_F_lambda_w_gurobi(A, y, lambda, w, optData, params);
            
            if params.use_golden_search
                x_debug_gurobi = goldenSearch(A, y, lambda, w, x_init, x_debug_gurobi, params.residualPower);
            end
            
            energy_w_gurobi = calc_F_lambda_w(A, x_debug_gurobi, y, lambda, params.residualPower, w);
        end
        
        % Quadprog
        if params.debug_quadprog
            [x_debug_quadprog, dbinfo_w_quadprog] = optimize_F_lambda_w_quadprog(A, y, lambda, w, x_out, optData, params);
            
            if params.use_golden_search
                x_debug_quadprog = goldenSearch(A, y, lambda, w, x_init, x_debug_quadprog, params.residualPower);
            end
            
            energy_w_quadprog = calc_F_lambda_w(A, x_debug_quadprog, y, lambda, params.residualPower, w);
        end
        
        % Gradient Descent
        if params.debug_gd
            [x_debug_gd, dbinfo_w_gd] = optimize_F_lambda_w_gd(A, y, lambda, w_curr, x_out, optData, params);
            
            if params.use_golden_search
                x_debug_gd = goldenSearch(A, y, lambda, w, x_init, x_debug_gd, params.residualPower);
            end
            
            energy_w_gd = calc_F_lambda_w(A, x_debug_gd, y, lambda, params.residualPower, w);
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


% Check if the energy of the solution is higher than that of the projected
% solution, or the initialization. In that case, return the better
% solution.
energy_w_out = calc_F_lambda_w(A, x_out, y, lambda, params.residualPower, w);

xProj = projectVec(x_out, A, y, k);
energyProj_w = calc_F_lambda_w(A, xProj, y, lambda, params.residualPower, w);

if min(energy_w_out, energy_w_init) >= energyProj_w
    x_out = xProj;
elseif min(energy_w_out, energyProj_w) >= energy_w_init
    x_out = x_init;
end

dbinfo_w.dbinfo_w_method = dbinfo_w_method;
dbinfo_w.tElapsed_w_solver = t;
dbinfo_w.tElapsed_w_goldenSearch = t_gs;

if params.monitor_violations
    %% Monitor violations
    % Compare solution to initialization
    
    % Compare all methods to yalmip
    if strcmp(params.solver, 'debug')
        dbinfo_w.violation_wSol_mosek_inferior_to_yalmip = max(0, (energy_w_mosek - energy_w_yalmip) / energy_w_yalmip);
        dbinfo_w.violation_wSol_gurobi_inferior_to_yalmip = max(0, (energy_w_gurobi - energy_w_yalmip) / energy_w_yalmip);
        dbinfo_w.violation_wSol_quadprog_inferior_to_yalmip = max(0, (energy_w_quadprog - energy_w_yalmip) / energy_w_yalmip);
        dbinfo_w.violation_wSol_fista_inferior_to_yalmip = max(0, (energy_w_gd - energy_w_yalmip) / energy_w_yalmip);
    else
        dbinfo_w.violation_wSol_mosek_inferior_to_yalmip = nan;
        dbinfo_w.violation_wSol_gurobi_inferior_to_yalmip = nan;
        dbinfo_w.violation_wSol_quadprog_inferior_to_yalmip = nan;
        dbinfo_w.violation_wSol_fista_inferior_to_yalmip = nan;
    end
    
    if isempty(x_init)
        dbinfo_w.violation_wSol_solution_inferior_to_init = 0;
    else
        dbinfo_w.violation_wSol_solution_inferior_to_init = max(0, ...
            (energy_w_out - energy_w_init) / energy_w_init);
    end
    
    if energy_w_out > energyProj_w
        dbinfo_w.violation_wSol_solution_inferior_to_projSol = ...
            (energy_w_out - energyProj_w) / energyProj_w;
    else
        dbinfo_w.violation_wSol_solution_inferior_to_projSol = 0;
    end
end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w_fista.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, dbinfo_w, z_out, t_out] = optimize_F_lambda_w_fista(A, y, lambda, w, x_init, optData, params, z_init, t_init)
if isempty(x_init)
    x_init = zeros(size(A,2),1);
end

if ~exist('z_init','var') || isempty(z_init)
    z_init = x_init;
end

if ~exist('t_init','var') || isempty(t_init)
    t_init = 1;
end

L = optData.A_data.norm_A^2;

obj = @(x) 0.5*norm(A*x-y)^2 + lambda*dot(w,abs(x));
obj2 = @(x,residual_x) 0.5*norm(residual_x)^2 + lambda*dot(w,abs(x));

%maj = @(x,z) 0.5*norm(A*z-y)^2 + dot(x-z, A'*(A*z-y)) + L/2*norm(x-z)^2 + lambda*dot(w,abs(x));
maj2 = @(x,z,residual_z,grad_z) 0.5*norm(residual_z)^2 + dot(x-z, grad_z) + L/2*norm(x-z)^2 + lambda*dot(w,abs(x));

soft_thresh = @(x,a) max(abs(x)-a, 0) .* sign(x);
prox = @(z, v, a) soft_thresh(z - (1/a)*v, (lambda/a)*w);
% v = A'*(A*z-y)

t_curr = t_init;
x_curr = x_init;
z_curr = z_init;
residual_z_curr = A*z_curr-y;
grad_z_curr = A'*residual_z_curr;

obj_curr = obj(x_curr);

x_best = x_curr;
obj_best = obj_curr;

viol_obj = 0;
viol_maj = 0;

obj_no_decrease_counter = 0;

for i=1:params.nIterMax_fista
    t_prev = t_curr;
    t_curr = (1 + sqrt(1 + 4*t_prev^2))/2;
    
    z_prev = z_curr;
    residual_z_prev = residual_z_curr;
    grad_z_prev = grad_z_curr;
    
    x_prev = x_curr;
    x_curr = prox(z_prev, grad_z_prev, L);
    
    % z_curr is determined by an adaptive restart rule:
    if (i >= 2) && (dot(x_curr - z_prev, x_curr - x_prev) < 0)
        % Restart
        z_curr = x_curr;
    else
        % Proceed as usual. Don't restart.
        z_curr = x_curr + (t_prev-1)/t_curr * (x_curr - x_prev);
    end
    
    residual_z_curr = A*z_curr-y;
    grad_z_curr = A'*residual_z_curr;
    
    % Violation monitoring and loop control. This only takes place every
    % several iterations.
    if mod(i, params.fista_monitor_decrease_every_nIter) == 0
        %obj_prev = obj_curr;
        obj_curr = obj(x_curr);
                
        if params.monitor_violations
            obj_z_prev = obj2(z_prev,residual_z_prev);
            viol_maj_curr = maj2(x_curr,z_prev,residual_z_prev,grad_z_prev) - obj_z_prev;
            viol_maj = max(viol_maj, viol_maj_curr);
            
            if viol_maj_curr <= 0
                viol_obj = max(viol_obj, obj_curr - obj_z_prev);
            end
        end
        
        % Keep best solution
        obj_best_prev = obj_best;
        if obj_curr < obj_best
            x_best = x_curr;
            obj_best = obj_curr;
        end
        
        % Loop control
        if obj_best_prev - obj_best <= params.Flw_fista_minimal_decrease * obj_best_prev
            obj_no_decrease_counter = obj_no_decrease_counter + 1;
            
            if obj_no_decrease_counter >= params.Flw_fista_num_small_decreases_for_stopping
                break
            end
        else
            obj_no_decrease_counter = 0;
        end
    end
    
    %fprintf('Objective: %g\n', 0.5*norm(A*x_curr-y)^2 + lambda*dot(w,abs(x_curr)));
end

%if i > 1
%    fprintf('nIter: %g \tBroke by limit: %d\n', i, broke_by_nIter_limit);
%end

x_out = x_best;

dbinfo_w.nIter_w_solver = i;

if params.monitor_violations
    dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = viol_obj;
    dbinfo_w.violationMax_wSol_majorizer_increase = viol_maj;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w_gurobi.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dbinfo_w] = optimize_F_lambda_w_gurobi(A, y, lambda, w, optData, params)
[n,d] = size(A);
vl = params.verbosity;

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

if params.monitor_violations
    dbinfo_w = struct();
    dbinfo_w.nIter_w_solver = nan;
    dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
    dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w_mosek.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dbinfo_w] = optimize_F_lambda_w_mosek(A, y, lambda, w, optData, params)
[n,d] = size(A);
vl = params.verbosity;

w = w(:);

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

% TODO: This is a test
x2 = projectToZeroRangeError(x, optData);

energy_x = calc_F_lambda_w(A, x, y, lambda, params.residualPower, w);
energy_x2 = calc_F_lambda_w(A, x2, y, lambda, params.residualPower, w);

if energy_x2 <= energy_x
    x = x2;
end

dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;

if params.monitor_violations
    dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
    dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w_quadprog.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dbinfo_w] = optimize_F_lambda_w_quadprog(A, y, lambda, w, x0, optData, params)
[n,d] = size(A);
vl = params.verbosity;

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

if params.monitor_violations
    dbinfo_w = struct();
    dbinfo_w.nIter_w_solver = nan;
    dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
    dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'optimize_F_lambda_w_yalmip.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, dbinfo_w] = optimize_F_lambda_w_yalmip(A, y, lambda, w, optData, params)

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

dbinfo_w = struct();
dbinfo_w.nIter_w_solver = nan;

if params.monitor_violations
    dbinfo_w.violationMax_wSol_energy_increase_majorizer_decrease = nan;
    dbinfo_w.violationMax_wSol_majorizer_increase = nan;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'pad_str.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sOut = pad_str(s, totLen, justify, padChar)
%function sOut = pad_str(s, totLen, justify, padChar)
%
% This function takes a string s and a number totLen and outputs s padded 
% in order to reach the length totLen.
%
% justify: 'l' / 'c' / 'r' for left / center / right. Default: center
%
% padChar is an optional argument that controls the character used for
% padding. Default: ' '
%

if ~exist('padChar','var') || isempty(padChar)
    padChar = ' ';
end

if ~exist('justify','var') || isempty(justify)
    justify = 'c';
end

if ~any(strcmp(justify,{'l','c','r'}))
    error('justify must be one of ''l''/''c''/''r''');
end

padLen = max(0, totLen - numel(s));

if strcmp(justify,'c')
lpad = double(idivide(int32(padLen),int32(2)));
rpad = lpad + mod(int32(padLen),int32(2));

lpadStr = repmat(padChar, [1, lpad]);
rpadStr = repmat(padChar, [1, rpad]);

sOut = [lpadStr, s, rpadStr];
elseif strcmp(justify,'l')
    padStr = repmat(padChar, [1, padLen]);
    sOut = [padStr, s];
elseif strcmp(justify,'r')
    padStr = repmat(padChar, [1, padLen]);
    sOut = [s, padStr];
end

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'processParams.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function params = processParams(nameValPairs, A, y, k)
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
defaultBaseParams.lambdaRelVals = [];
defaultBaseParams.residualPower = 2;
defaultBaseParams.verbosity = 2;
defaultBaseParams.solver = [];

% Fill in basic parameters that were not defined by the user
baseParams = addDefaultFields(params, defaultBaseParams, 'discard');
clear defaultBaseParams

% Verify and process basic parameters
baseParams.profile = lower(baseParams.profile);

if ~ismember(baseParams.profile, {'fast', 'normal', 'thorough', 'ultra'})
    error('Invalid profile name ''%s'' in parameter ''profile''', baseParams.profile);
end

if ~ismember(baseParams.residualPower, [1,2])
    error('Parameter ''residualPower'' should be 1 or 2');
end

% Solver
baseParams.solver = lower(baseParams.solver);

if ~isempty(baseParams.solver) && ~any(strcmp(baseParams.solver, {'fista','mosek','yalmip'}))
    error('<solver> must be one of ''fista'', ''mosek'' or ''yalmip''');
end

if isempty(baseParams.solver)
    if baseParams.residualPower == 2
        baseParams.solver = 'fista';
    else
        if detect_mosek()
            baseParams.solver = 'mosek';
        elseif detect_yalmip()
            baseParams.solver = 'yalmip';
        else
            error('<residualPower> = 1 requires MOSEK or YALMIP');
        end
    end
elseif (baseParams.residualPower == 1) && ~any(strcmp(baseParams.solver, {'mosek','yalmip'}))
    error('<residualPower> = 1 can only be used with <solver> = ''mosek'' or ''yalmip''');
end

if ~isempty(baseParams.lambdaVals) && ~isempty(baseParams.lambdaRelVals)
    error('Only one of <lambdaVals>, <lambdaRelVals> can be nonempty');
end

if strcmp(baseParams.solver, 'mosek') && ~detect_mosek()
    error('<solver> = ''mosek'' but MOSEK not detected');
end

if strcmp(baseParams.solver, 'yalmip') && ~detect_yalmip()
    error('<solver> = ''yalmip'' but YALMIP not detected');
end

%% Combine default parameters with user overrides
defaultParams = getDefaultParams(baseParams, A, y, k);

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

if ~params.monitor_violations
    if strcmp(params.solver,'debug')
        error('<monitor_violations> must be 1 when using the ''debug'' solver');
    end
    
    if params.verbosity >= 3
        error('<monitor_violations> must be 1 when <verbosity> is >= 3');
    end
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'project.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, I] = project(x,k)
[~,I] = sort(abs(x),'descend');
x_out = zeros(size(x));
I = I(1:k);
x_out(I) = x(I);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'projectToZeroRangeError.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x_out = projectToZeroRangeError(x, optData)
%function x_out = projectToZeroRangeError(x, optData)
%
% Returns the closest vector to x which satisfies A*x=y
%x_out = optData.PInvAy + optData.A_data.NANAt*(x - optData.PInvAy);
x_out = x + optData.A_data.rangeAt*optData.A_data.rangeAt'*(optData.PInvAy-x);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'projectVec.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_proj, supp] = projectVec(x,A,y,k)
%function [x_proj, supp] = projectVec(x,A,y,k)
[~,I] = sort(abs(x),'descend');
supp = I(1:k);

x_proj = zeros(numel(x),1);
x_proj(supp) = A(:,supp)\y;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'qprint.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usage: qprint(qp, varargin)
%        qprint(vl, vl_thresh, varargin)
% 
% Prints output only if qp is a logical that equals true, or vl and
% vl_thresh are numbers such that vl >= vl_thresh.
%
% Examples: qprint(true, 'Results: %d, %d\n', result1, result2);
%           qprint(vl, 2, 'Here is some verbose report: %g %g %g', number1, number2, number3);
function printed = qprint(varargin)
printed = false;

if nargin == 0
    error('Not enough arguments');
end

if (nargin == 1) && iscell(varargin{1})
    args = varargin{1};
else
    args = varargin;
end

if islogical(args{1}) && isscalar(args{1})
    qp = args{1};
    args = args(2:end);
elseif (numel(args) >= 2) && all(isnumeric([args{1} args{2}])) && isscalar(args{1}) && isscalar(args{2})
    qp = args{1} >= args{2};
    args = args(3:end);
else
    error('Invalid input format');
end

if ~isempty(args) && ~isstr(args{1})
    error('Invalid input format');
end

if ~qp
    return
end

% If we are here, we can print
printed = true;

if numel(args) == 0
    return
end

commandStr = 'fprintf(';
for i= 1:numel(args)
    commandStr = [commandStr, sprintf('args{%d}', i)];
    
    if i < numel(args)
        commandStr = [commandStr, ', '];
    else
        commandStr = [commandStr, ');'];
    end
end

eval(commandStr);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'qprintln.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usage: qprintln(qp, varargin)
%        qprintln(vl, vl_thresh, varargin)
% 
% Prints output only if qp is a logical that equals true, or vl and
% vl_thresh are numbers such that vl >= vl_thresh.
% Unlike qprint, this function prints an additional '\n' at the end.
%
% Examples: qprintln(true, 'Results: %d, %d\n', result1, result2);
%           qprintln(vl, 2, 'Here is some verbose report: %g %g %g', number1, number2, number3);
%           qprintln(true); % Just jumps to a new line
function printed = qprintln(varargin)
printed = qprint(varargin);
if printed
    fprintf('\n');
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'refineGreedily.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'rpad_num.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sOut = rpad_num(x, totLen, formatStr, padChar)
%function sOut = rpad_num(x, totLen, formatStr, padChar)
%
% This function takes a number x and a number totLen and outputs x in
% string format, padded from the right in order to reach the length totLen.
%
% padChar is an optional argument that controls the character used for
% padding. Default: ' '
%
% formatStr is the format string (as in printf). Default: '%g'


if ~exist('padChar','var') || isempty(padChar)
    padChar = ' ';
end

if ~exist('formatStr','var') || isempty(formatStr)
    formatStr = '%g';
end

sIn = num2str(x, formatStr);

padLen = max(0, totLen - numel(sIn));

padStr = repmat(padChar, [1, padLen]);
sOut = [sIn, padStr];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'seek_greedy_improvement.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_out, is_improved, improvement_rel] = seek_greedy_improvement(x_in, A, y, k, lambda, optData, params)
d = size(A,2);
supp_orig = calc_supp(x_in, k);
obj_orig = calc_F_lambda_gamma(A, x_in, y, k, lambda, inf, params.residualPower);
candidates = setdiff(1:d, supp_orig);

%n = size(A,1);
%v = abs(y' * (A(:,supp_orig) .* repmat(x_in(supp_orig)', [n,1])));
%[~,vv] = sort(v,'descend');
%supp_orig = supp_orig(vv);

if strcmp(params.ambiguous_escape_strategy, 'gradient')
    x_trunc = x_in;
    x_trunc(candidates) = 0;

    grad_orig = A'*(A*x_trunc-y);
    [~,c_best] = max(abs(grad_orig(candidates)));
    i_best = candidates(c_best);
    
    supp_best = supp_orig;
    supp_best(k) = i_best;
    w_preopt = ones(size(x_in));
    w_preopt(supp_best) = 0;
    x_preopt = x_in;
    obj_preopt = calc_F_lambda_w(A, x_preopt, y, lambda, params.residualPower, w_preopt);
    
elseif strcmp(params.ambiguous_escape_strategy, 'ls')    
    obj_preopt = inf;

    supp_curr = supp_orig;
    
    for j=1:(d-k)
        supp_curr(k) = candidates(j);
        
        x_curr = minimize_f_on_support(supp_curr,A,y);
        [obj_curr,~,w_curr] = calc_F_lambda_gamma(A, x_curr, y, k, lambda, inf, params.residualPower);
        
        if obj_curr < obj_preopt
            obj_preopt = obj_curr;
            x_preopt = x_curr;
            w_preopt = w_curr;            
        end
    end    
else
    error('Invalid value ''%s'' for <ambiguous_escape_strategy>', params.ambiguous_escape_strategy);
end

obj_opt = obj_preopt;
x_opt = x_preopt;
w_opt = w_preopt;
improved = true;

while improved
    obj_prev = obj_opt;
    w_prev = w_opt;
    x_opt = optimize_F_lambda_w(A, y, k, lambda, w_opt, x_opt, optData, params);
    [obj_opt, ~, w_opt] = calc_F_lambda_gamma(A, x_opt, y, k, lambda, inf, params.residualPower);
    improved = (obj_opt < obj_prev) && any(w_opt ~= w_prev);
end

supp_opt = calc_supp(x_opt, k);

if obj_opt < obj_orig
    x_out = x_opt;
    improvement_rel = (obj_orig - obj_opt) / obj_orig;
else
    x_out = x_in;
    improvement_rel = 0;
end

is_improved = any(sort(supp_opt) ~= sort(supp_orig)) && ...
    (obj_orig - obj_opt) >= obj_orig * 1e-4;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'solve_P0_main.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% Any lambda above this values guarantees that the solution is sparse
lambdaSparse   = optData.lambdaSparseGuarantee;

% Initial reporting
qprintln(vl,1,'');
qprint(vl,1,'Profile: %s', lower(params.profile));

if vl == 1
    fprintf('. ');
end

qprintln(vl,2,'\nResidual power: %d', params.residualPower);
qprintln(vl,2,'Weighted l1 solver: %s', (params.solver));
qprintln(vl,2,'');

nn = 8214;
s2 = 8322;
s1 = 8321;
uGamma = 947;
uNabla = 8711;

uTruncx = [char(928), char(8342), '(x)'];
uProjx = 'LS(x)';
uLambda = char(955);
uTau = char(964);

qprintln(vl, 2, 'Legend:');
%qprintln(vl, 2, ' truncResNormRel = %cA*%s - y%c%c / %cy%c%c', nn, uProjx, nn, s2, nn, nn, s2);
qprintln(vl, 2, '   LS_resNormRel = %cA*%s - y%c%c / %cy%c%c', nn, uProjx, nn, s2, nn, nn, s2);
qprintln(vl, 2, '     nonSparsity = |x|_(k+1) / |x|_(k)');
qprintln(vl, 2, '      resNormRel = %cA*x - y%c%c / %cy%c%c', nn, nn, s2, nn, nn, s2);
qprintln(vl, 2, '           %s_rel = %s / %s_bar', uLambda, uLambda, uLambda);
qprintln(vl,2);
qprintln(vl,2,'%s_bar \t- %s that guarantees a k-sparse solution. %s_bar = %.3g', uLambda, uLambda, uLambda, optData.lambdaSparseGuarantee);
%qprintln(vl,2,'%s_rel \t- %s / %s_bar', uLambda, uLambda, uLambda);
qprintln(vl,2,'%s \t- k-sparse projection of x', uTruncx);
qprintln(vl,2,'%s\t- minimizer of %cA*u - y%c%c over all u that have the same support as %s', uProjx, nn, nn, s2, uTruncx);

qprintln(vl,2);

qprintln(vl,1,'%dx%d matrix, k=%d. Using %d %s-values: %.3g <= %s_rel <= %.3g', n, d, k, numel(lambda_vals), uLambda,  min(lambda_vals) / lambdaSparse, uLambda, max(lambda_vals) / lambdaSparse);
qprintln(vl,1);

if params.monitor_violations
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
    violationMax_wSol_fista_inferior_to_yalmip = nan;
    
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
        violationMax_wSol_fista_inferior_to_yalmip = 0;
    end
    
    violationMax_wSol_energy_increase_majorizer_decrease = 0;
    violationMax_wSol_majorizer_increase = 0;
    
    violationMax_wSol_nonZero_residual_for_small_lambda = 0;
    violationMax_abs_wSol_nonSparse_solution_for_large_lambda = 0;
end

%% Initialize loop variables

% A function that tells if x is sparse according to the threshold parameter
isSparse = @(x) nonSparsityAbs(x, k, optData.scale_factor_x) <= params.sparsityThreshAbs_x;

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
    [xUnproj_curr, dbinfo_lambda] = optimize_F_lambda(A, y, k, lambda_curr, x_init_curr, optData, params);
    tElapsed_curr = toc(tElapsed_curr);
    
    xProj_curr = projectVec(xUnproj_curr, A, y, k);
    
    
    %% Analyze current solution
    % Energy of current solution in (P_lambda)
    energy_P_lambda_curr = calcEnergy(A, xUnproj_curr, y, k, lambda_curr, params.residualPower);
    
    % Energy of projected solution in (P_lambda).
    energyProj_P_lambda_curr = calcEnergy(A, xProj_curr, y, k, lambda_curr, params.residualPower);
    
    % Energy of projected solution in (P0)
    energy_P0_curr   = norm(A*xProj_curr-y);
    
    % Relative tail and residual norm of solution
    nonSparsity_curr = nonSparsityRel(xUnproj_curr, k);
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
    
    db(i).LS_resNorm      = norm(A*xProj_curr-y);
    db(i).LS_resNormRel   = norm(A*xProj_curr-y)/norm(y);
    
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
        sol.stop_message = sprintf('Obtained minimizer of F_%c(x) was k-sparse for %d consecutive values of lambda.', uLambda, params.nLambdas_sparse_x_to_stop);
    elseif (i_next > numel(db)) && reached_sparsity
        sol.stop_reason = 'finished';
        sol.stop_message = 'Finished solving for all values of lambda. Got a sparse solution.';
    elseif (i_next > numel(db)) && ~reached_sparsity
        sol.stop_reason = 'finished_no_sparse';
        sol.stop_message = 'Finished solving for all values of lambda, but failed to get a sparse solution.';
    end
    
    
    if params.monitor_violations
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
        
        violationMax_wSol_fista_inferior_to_yalmip = max( ...
            violationMax_wSol_fista_inferior_to_yalmip, ...
            dbinfo_lambda.violationMax_wSol_fista_inferior_to_yalmip);
        
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
    end
    
    %% Report current iteration
    % Round the time for reporting
    if tElapsed_curr < 10
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
    lambdaStr = sprintf('%srel = %s', uLambda, rpad_num(lambda_curr / lambdaSparse, num_disp_length));
    
    lengths = struct();
    lengths.iterStr = 3;
    lengths.lambdaRel = 9;
    lengths.time = 7;
    lengths.nGammas = 4;
    lengths.nws = 4;
	lengths.ngrads = 5;
    lengths.LS_resNormRel = 13;
    lengths.nonSparsity = 13;
    lengths.resNormRel = 13;
    
    if i == 1
        qprintln(vl, 1, ' %s  %s | %s | %s | %s | %s | %s | %s | %s\n', ...
            pad_str('   ',lengths.iterStr), ...
            pad_str([uLambda, '_rel'], lengths.lambdaRel), ...
            pad_str('time[s]', lengths.time), ...
            pad_str(['#',uGamma], lengths.nGammas), ...
            pad_str('#w', lengths.nws), ...
			pad_str(['#',uNabla, 'f'], lengths.ngrads), ...
            pad_str('LS_resNormRel', lengths.LS_resNormRel), ...
            pad_str('nonSparsity', lengths.nonSparsity), ...
            pad_str('resNormRel', lengths.resNormRel));
    end
    
    % Report the current iteration's performance
    %qprintln(vl, 2, '%s) %s | %s | %s | %s | %s | %s', ...
    %    iterStr, lambdaStr, rpad_num(tLambdaRound,5), rpad_num(dbinfo_lambda.num_gammas,4), uTau, rpad_num(tailRel_curr,num_disp_length), rpad_num(energy_P0_curr/norm(y),num_disp_length), rpad_num(resNormRel_curr,num_disp_length));
    %qprintln(vl, 2, '%s) %s | time = %s | nGammas = %s | %srel = %s | LS_resNormRel = %s | resNormRel = %s', ...
    %    iterStr, lambdaStr, rpad_num(tLambdaRound,5), rpad_num(dbinfo_lambda.num_gammas,4), uTau, rpad_num(tailRel_curr,num_disp_length), rpad_num(energy_P0_curr/norm(y),num_disp_length), rpad_num(resNormRel_curr,num_disp_length));
    
        qprintln(vl, 1, '%s%s) %s | %s | %s | %s | %s | %s | %s | %s', ...
            iterChar, ...
            pad_str(sprintf('%g',i), lengths.iterStr, 'l', ' '), ...
            rpad_num(lambda_curr / lambdaSparse, lengths.lambdaRel, '%.3g'), ...
            rpad_num(tLambdaRound, lengths.time), ...
            rpad_num(dbinfo_lambda.num_gammas, lengths.nGammas), ...
            rpad_num(dbinfo_lambda.num_ws, lengths.nws), ...
			rpad_num(dbinfo_lambda.nIter_w_solver, lengths.ngrads), ...
            rpad_num(energy_P0_curr/norm(y), lengths.LS_resNormRel, '%.7g'), ...
            rpad_num(nonSparsity_curr, lengths.nonSparsity, '%.7g'), ...
            rpad_num(resNormRel_curr, lengths.resNormRel, '%.7g'));
    
    nReports = 0;
    
    if params.monitor_violations
        %% Report violations
        % This funciton reports violations that are measured in relative terms
        reportViolation = @(viol_curr, viol_thresh, viol_name) ...
            qprintln((vl>=3) && (viol_curr > viol_thresh), ...
            '%s%s Violation: %s  [ %g rel ]', normalIterChar, violationStr, viol_name, viol_curr);
        
        % This funciton reports violations that are measured in absolute terms
        reportViolation_abs = @(viol_curr, viol_thresh, viol_name) ...
            qprintln((vl>=3) && (viol_curr > viol_thresh), ...
            '%s%s Violation: %s  [ %g abs ]', normalIterChar, violationStr, viol_name, viol_curr);
        
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
        
        viol_curr = dbinfo_lambda.violationMax_wSol_fista_inferior_to_yalmip;
        viol_thresh =  params.reportThresh_wSol_fista_inferior_to_yalmip;
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
end

tElapsed = toc(tStart);

% Return final solution
x_sol = xProj_best;


%% Final report
qprintln(vl,1);

qprintln(vl,2, '%s Breaking.', sol.stop_message);
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
I_best = find([db.LS_resNorm] == min([db.LS_resNorm]));

sol.lambdaBestIdx_min = min(I_best);
sol.lambdaBestIdx_max = max(I_best);

sol.lambdaBest_min = min([db(I_best).lambda]);
sol.lambdaBest_max = max([db(I_best).lambda]);

sol.lambdaRelBest_min = min([db(I_best).lambdaRel]);
sol.lambdaRelBest_max = max([db(I_best).lambdaRel]);


%% Return violations
if params.monitor_violations
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
        viol.violationMax_wSol_fista_inferior_to_yalmip = violationMax_wSol_fista_inferior_to_yalmip;
    else
        viol.violationMax_wSol_mosek_inferior_to_yalmip = nan;
        viol.violationMax_wSol_gurobi_inferior_to_yalmip = nan;
        viol.violationMax_wSol_quadprog_inferior_to_yalmip = nan;
        viol.violationMax_wSol_fista_inferior_to_yalmip = nan;
    end
else
    viol = [];
end

if params.monitor_violations
    sol.viol = viol;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'stepVec_exp.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'stepVec_pwlin.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'stepVec_tan.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = stepVec_tan(a, b, nVals)
%function v = stepVec_tan(a, b, nVals)
%
% Returns a vector v of nVals values in [a,b] such that arctan(v/b) is
% equi-spaced on the interval [arctan(a/b), arctan(1)]. In practice, the
% result is that the values are about twice as dense around point a as they
% are around point b.
v = b * tan(atan(a/b) + (0:(nVals-1))*(atan(1)-atan(a/b))/(nVals-1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'tailNorm.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = tailNorm(x,k)
%function out = tailNorm(x,k)
%
% Tells how far a vector x is from being k-sparse, in terms of l1-distance.
out = norm(x - truncVec(x,k),1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'trimmedLasso.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = trimmedLasso(x,k)
out = sort(abs(x),'ascend');
out = sum(out(1:numel(x)-k));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'truncVec.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_trunc, supp] = truncVec(x,k)
%function [x_trunc, supp] = truncVec(x,k)
%
% Truncates the vector x to the nearest vector with cardinality <= k.
%
% x_trunc - The truncated vector. Ssme size as x, but with the n-k
%           smallest-magnitude entries set to zero.
%
% supp - The support of the truncated vector. A list of the indices of the
%        k largest-magnitude entries.

n = numel(x);
x_trunc = x;
[~,J] = sort(abs(x));
x_trunc(J(1:n-k)) = 0;
supp = J(n-k+1:end);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Input file: 'update_min.m'  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

