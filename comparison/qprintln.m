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



