function out = nonSparsityRel(x,k)
%function out = nonSparsityRel(x,k)
%
% Tells how far a vector x is from being k-sparse, in terms of relative
% l1-distance.

n = numel(x);
out = norm(x - truncVec(x,k),1) / norm(truncVec(x,k),1) * (k/(n-k));
end
