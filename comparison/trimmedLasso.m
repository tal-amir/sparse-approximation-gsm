function out = trimmedLasso(x,k)
out = sort(abs(x),'ascend');
out = sum(out(1:numel(x)-k));
end

