function s = namevals2struct(namevals)
%function s = namevals2struct(v)
%
% This function takes a cell array that contains name/value pairs and
% returns a struct defined by these names and values.
%
% Example:
% >> s = namevals2struct({'a',5,'b',6,'c',[],'d',[1,2,3]});
%
% s = 
%
%    a: 5
%    b: 6
%    c: []
%    d: [1 2 3]

if numel(namevals) == 0
    s = struct();
    return
elseif numel(namevals) == 1
    namevals = namevals{1};
end

if mod(numel(namevals),2) ~= 0
    error('v must contain name/value pairs');
end

n = numel(namevals)/2;

names = namevals(1:2:2*n-1);
values = namevals(2:2:2*n);

if ~all(cellfun(@isstr, names))
    error('v must contain name/value pairs');
end

if numel(unique(names)) ~= n
    error('Parameter names must not repeat');
end

if any(cellfun(@isempty, names))
    error('Parameter names cannot be empty strings');
end

s = cell2struct(values,names,2);

end
