function out = nonSparsityAbs(x,k)
%function out = nonSparsityAbs(x,k)
%
% Tells how far a vector x is from being k-sparse, in terms of average
% magnitude per entry.

n = numel(x);
out = norm(x - truncVec(x,k),1) / (n-k);
end
