% Usage: qprintln(qp, varargin)
%        qprintln(vl, vl_thresh, varargin)
% 
% Prints output only if qp is a logical that equals true, or vl and
% vl_thresh are numbers such that vl >= vl_thresh.
% Unlike qprint, this function prints an additional '\n' at the end.
%
% Examples: qprintln(true, 'Results: %d, %d\n', result1, result2);
%           qprintln(vl, 2, 'Here is some verbose report: %g %g %g', number1, number2, number3);
%           qprintln(true); % Just jumps to a new line
function printed = qprintln(varargin)
printed = qprint(varargin);
if printed
    fprintf('\n');
end
end


% This function checks if all the entries in x are round numbers.
function out = isround(x)
out = all(round(x(:)) == x(:));
end


% Returns a string of the format HH:MM:SS.FFF describing the time given in
% t. t should be given in seconds.
function sOut = getTimeStr(t)

if isnan(t)
    sOut = 'nan';
    return
end

sOut = sprintf('%s', datestr(t/24/60/60,'HH:MM:SS.FFF'));

if strcmp(sOut(1:3), '00:')
    sOut = sOut(4:end);
end

end



