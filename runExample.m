% Simple usage example of the GSM sparse approximation solver.
% Tries to recover a k-sparse signal x0 from a noisy sample y=A*x0+y.
%
% This script requires the Mosek optimization solver:
%     https://www.mosek.com/downloads/

addpath('./utils');

% Generates a (n x d) matrix A, a k-sparse signal x0, and a noisy
% sample y = A*x0 + e, and tries to recover x0 from y.
n = 25; d = 100;
k = 8;

% Relative noise level. Set to 5% noise.
nu = 0.05;

% Fix random seed
rng(12345);


% Generate a gaussian dictionary with normalized columns
A = randn([n,d]);
A = A ./ repmat(sqrt(sum(A.^2,1)),[n,1]);

% Gaussian k-sparse signal x0
x0 = zeros(d,1);
S = randperm(d,k);
x0(S) = randn([k,1]);

% Noisy sample
y = A*x0;
e = randn([n,1]);
e = e / norm(e) * norm(y) * nu;
y = y+e;

% Basic usage
%[x_sol, sol] = sparse_approx_gsm_v1_10(A,y,k);

% Power-1 variant:
%[x_sol, sol] = sparse_approx_gsm_v1_10(A, y, k, 'residualPower', 1);

% Controlling the number of lambda values. Default: 50
[x_sol, sol] = sparse_approx_gsm_v1_10(A, y, k, 'nLambdaVals', 10);

relResNorm = norm(A*x_sol-y) / norm(A*x0-y);

fprintf('\n');
fprintf('Residual norm, relative to x0: %g ', relResNorm);

if relResNorm <= 1
    fprintf('<= 1. Optimization success.\n');
else
    fprintf('> 1. Suboptimal solution.\n');
end

fprintf('Relative recovery error:       %.2f%%\n', 100*norm(x_sol-x0,1) / norm(x0,1));
fprintf('Support precision:             %.2f%%\n', 100*nnz(x0.*x_sol)/k);
fprintf('\n');


