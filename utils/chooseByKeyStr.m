function out = chooseByKeyStr(key, varargin)
%function out = chooseByKeyStr(key, varargin)
%
% Chooses a value according to a given key string.
%
% Example: thresh = chooseByKeyStr(profileName, 'fast', 0.01, 'normal', 0.5, 'thorough', 0.9)
%          Sets 'thresh' to be 0.01, 0.5 or 0.9, depending on whether
%          profileName is 'fast', 'normal' or 'thorough' respectively.
found = false;

for i=1:numel(varargin)/2
    if strcmp(key, varargin{2*i-1})
        if found
            error('Key ''%s'' appears in arguments more than once', key);
        end
        
        out = varargin{2*i};
        found = true;
    end
end

if ~found
    error('Key ''%s'' is not supplied in arguments', key);
end
end
