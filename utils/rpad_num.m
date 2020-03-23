function sOut = rpad_num(x, totLen, formatStr, padChar)
%function sOut = rpad_num(x, totLen, formatStr, padChar)
%
% This function takes a number x and a number totLen and outputs x in
% string format, padded from the right in order to reach the length totLen.
%
% padChar is an optional argument that controls the character used for
% padding. Default: ' '
%
% formatStr is the format string (as in printf). Default: '%g'


if ~exist('padChar','var') || isempty(padChar)
    padChar = ' ';
end

if ~exist('formatStr','var') || isempty(formatStr)
    formatStr = '%g';
end

sIn = num2str(x, formatStr);

padLen = max(0, totLen - numel(sIn));

padStr = repmat(padChar, [1, padLen]);
sOut = [sIn, padStr];
end



