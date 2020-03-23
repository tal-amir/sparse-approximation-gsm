function out = tailNorm(x,k)
%function out = tailNorm(x,k)
%
% Tells how far a vector x is from being k-sparse, in terms of l1-distance.
out = norm(x - truncVec(x,k),1);
end
