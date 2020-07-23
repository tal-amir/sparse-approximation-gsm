% Usage example of the GSM sparse approximation solver.
% Tries to recover a k-sparse signal x0 from a noisy sample y = A*x0 + y.

addpath(fullfile(pwd,'./gsm'));

%% Prepare problem setting

% Generates a (n x d) matrix A, a k-sparse signal x0, and a noisy
% sample y = A*x0 + e, and tries to recover x0 from y.
n = 100; d = 5000;
k = 20;

% Relative noise level, set to 5% noise.
nu = 0.05;

% Fix random seed
rng(12345);

% Generate a random Gaussian dictionary with normalized columns
A = randn([n,d]);
A = A ./ repmat(sqrt(sum(A.^2,1)),[n,1]);

% Gaussian k-sparse signal x0
x0 = zeros(d,1);
S = randperm(d,k);
x0(S) = randn([k,1]);

% Noisy undersampling
y = A*x0;
e = randn([n,1]);
e = e / norm(e) * norm(y) * nu;
y = y+e;

fprintf('\nGenerating k-sparse signal x0 in R^d and a noisy undersample y = A*x0 + e in R^n.\n');
fprintf('n=%d, d=%d, k=%d, noise level nu = norm(e)/norm(A*x0) = %g\n', n, d, k, nu);


%% Call the solver

[x_sol, sol] = sparse_approx_gsm_v1_21(A, y, k);

% Alternative examples:
% =====================

% Use the thorough profile instead of the normal one:
% [x_sol, sol] = sparse_approx_gsm_v1_21(A, y, k, 'profile', 'thorough');

% Use power 1 for the residual instead of 2. This requires Mosek or Yalmip.
% [x_sol, sol] = sparse_approx_gsm_v1_21(A, y, k, 'residualPower', 1);

% Use the thorough profile but disable postprocessing by OMP:
% [x_sol, sol] = sparse_approx_gsm_v1_21(A, y, k, 'profile', 'thorough', 'postprocess_by_omp', false);

% An alternative way to pass parameters:
% params = {'profile', 'thorough', 'postprocess_by_omp', false};
% [x_sol, sol] = sparse_approx_gsm_v1_21(A, y, k, params);


%% Evaluate solution and report

% Relative residual norm
relResNorm = norm(A*x_sol-y) / norm(A*x0-y);

fprintf('\n');
fprintf('Residual norm, relative to x0: %g ', relResNorm);

if relResNorm <= 1
    fprintf('<= 1. Optimization success.\n');
else
    fprintf('> 1. Suboptimal solution.\n');
end

% Relative recovery error
relRecErr = norm(x_sol-x0,1) / norm(x0,1);

fprintf('Relative recovery error:       %g ', relRecErr);

if relRecErr <= 2*nu
    fprintf('<= 2*nu. Recovery success.\n');
else
    fprintf('> 2*nu. Recovery failure.\n');
end

fprintf('Support precision:             %.2f%%\n', 100*nnz(x0.*x_sol)/k);
fprintf('\n');


