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
