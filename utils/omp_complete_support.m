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
