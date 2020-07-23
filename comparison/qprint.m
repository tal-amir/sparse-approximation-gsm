% Usage: qprint(qp, varargin)
%        qprint(vl, vl_thresh, varargin)
% 
% Prints output only if qp is a logical that equals true, or vl and
% vl_thresh are numbers such that vl >= vl_thresh.
%
% Examples: qprint(true, 'Results: %d, %d\n', result1, result2);
%           qprint(vl, 2, 'Here is some verbose report: %g %g %g', number1, number2, number3);
function printed = qprint(varargin)
printed = false;

if nargin == 0
    error('Not enough arguments');
end

if (nargin == 1) && iscell(varargin{1})
    args = varargin{1};
else
    args = varargin;
end

if islogical(args{1}) && isscalar(args{1})
    qp = args{1};
    args = args(2:end);
elseif (numel(args) >= 2) && all(isnumeric([args{1} args{2}])) && isscalar(args{1}) && isscalar(args{2})
    qp = args{1} >= args{2};
    args = args(3:end);
else
    error('Invalid input format');
end

if ~isempty(args) && ~isstr(args{1})
    error('Invalid input format');
end

if ~qp
    return
end

% If we are here, we can print
printed = true;

if numel(args) == 0
    return
end

commandStr = 'fprintf(';
for i= 1:numel(args)
    commandStr = [commandStr, sprintf('args{%d}', i)];
    
    if i < numel(args)
        commandStr = [commandStr, ', '];
    else
        commandStr = [commandStr, ');'];
    end
end

eval(commandStr);
end



