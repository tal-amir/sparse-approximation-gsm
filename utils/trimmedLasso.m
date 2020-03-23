function out = trimmedLasso(x,k)
s = sort(abs(x));
out = sum(s(1:(numel(x)-k)));
end
