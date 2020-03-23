function sOut = getTimeStr(t)
% Returns a string of the format HH:MM:SS.FFF describing the time given in
% t. t should be given in seconds.

if isnan(t)
    sOut = 'nan';
    return
end

sOut = sprintf('%s', datestr(t/24/60/60,'HH:MM:SS.FFF'));

if strcmp(sOut(1:3), '00:')
    sOut = sOut(4:end);
end

end

