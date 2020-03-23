function x_sol = solve_P_lambda_gsm(A,y,k,lambda_rel,lambda_rel_base)

lambda_bar = sqrt(max(sum(A.^2, 1)))*norm(y);
lambda = lambda_rel*lambda_bar;

if ~exist('lambda_rel_base','var')
    lambda_rel_base = [];
end

if ~isempty(lambda_rel_base)
    if lambda_rel < lambda_rel_base
        error('lambda_base must be lower or equal to lambda');
    end
    
    lambda_base = lambda_rel_base*lambda_bar;
end



%% Solve without init
params = {'profile', 'thorough', 'verbosity',0, ...
    'lambdaVals', lambda};

[~, sol] = sparse_approx_gsm_v1_10(A, y, k, params);

x_sol = sol.db(end).x;


%% Solve with init
if isempty(lambda_rel_base) || (lambda_rel == lambda_rel_base)
    return;
end

params = {'profile', 'thorough', 'verbosity',0, ...
    'lambdaVals', [lambda_base,lambda], ...
    'init_x_from_previous_lambda', true, ...
    'full_homotopy_with_init', true};

[~, sol2] = sparse_approx_gsm_v1_10(A, y, k, params);

x_sol_2 = sol2.db(end).x;

% Choose better solution
Fk = @(x) 0.5*norm(A*x-y)^2 + lambda*trimmedLasso(x,k);

if Fk(x_sol_2) < Fk(x_sol)
    x_sol = x_sol_2;
end

end



function out = trimmedLasso(x,k)
s = sort(abs(x));
out = sum(s(1:(numel(x)-k)));
end
