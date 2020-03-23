function x_sol = solve_P_lambda_tls(A, y, k, tls_method, eta_vals, lambda_rel, lambda_rel_base)

if ~any(strcmp(tls_method,{'dcp','admm'}))
    error('tel_method must be ''dcp'' or ''admm''');
end

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

% Calculate objective function
Fk = @(x) 0.5*norm(A*x-y)^2 + lambda*trimmedLasso(x,k);

obj_best = inf;

for i_eta = 1:numel(eta_vals)
    eta = eta_vals(i_eta);
    
    %% Solve without init
    [~, sol] = solve_trimmed_lasso(A, y, k, ...
        'method', tls_method, ...
        'lambdaVals', lambda, ...
        'eta', eta, ...
        'x_init', [], ...
        'verbosity', 0);
    
    x_curr = sol.x_all{end};
    
    obj_curr = Fk(x_curr);
    
    if obj_curr < obj_best
        obj_best = obj_curr;
        x_sol = x_curr;
    end
    
    %% Solve with init
    if isempty(lambda_rel_base) || (lambda_rel == lambda_rel_base)
        return;
    end
    
    [~, sol2] = solve_trimmed_lasso(A, y, k, ...
        'method', tls_method, ...
        'lambdaVals', [lambda_base,lambda], ...
        'eta', eta, ...
        'propagate_solutions_through_lambdas', true, ...
        'x_init', [], ...
        'verbosity', 0);
    
    x_curr = sol2.x_all{end};
    
    % Choose better solution
    
    obj_curr = Fk(x_curr);
    
    if obj_curr < obj_best
        obj_best = obj_curr;
        x_sol = x_curr;
    end
end

end



function out = trimmedLasso(x,k)
s = sort(abs(x));
out = sum(s(1:(numel(x)-k)));
end
