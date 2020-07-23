function s_out = addDefaultFields(s_in, s_defaults, nonExistantFieldAction)
%function s_out = addDefaultFields(s_in, s_defaults, nonExistantFieldAction)
%
% This function takes a struct s_in, and a 'default value' struct
% s_defaults, and for each field that is not set in s_in, sets it to the
% appropriate value given in s_default. The result is returned in s_out.
%
% If s_in contains fields that are not given default values in s_defaults,
% an error is thrown.
%
% If s_in is set to [], s_defaults is returned.
%
% nonExistantFieldAction - (optional) Determines the action taken when
%                          s_in contains a field that does not exist in
%                          s_default. Possible values:
%                          'allow'   - Such fields are added to s_out
%                          'discard' - Such fields are discarded from s_out
%                          'error'   - Return an error message
%                          

if ~exist('nonExistantFieldAction', 'var') || isempty(nonExistantFieldAction)
    nonExistantFieldAction = 'error';
end

nonExistantFieldAction = lower(nonExistantFieldAction);

if ~ismember(nonExistantFieldAction, {'allow', 'discard', 'error'})
    error('Invalid option ''%s'' for argument nonExistantFieldAction.\nValid options: ''allow'', ''discard'', ''error''');
end

if isempty(s_in)
    s_out = s_defaults;
    return
end

userDefinedFieldNames = fieldnames(s_in);
isFieldNameValid = false(size(userDefinedFieldNames));

defaultFieldNames = fieldnames(s_defaults);
ndefs = numel(defaultFieldNames);

s_out = s_in;


for i=1:ndefs
    currName = defaultFieldNames{i};
    currVal  = getfield(s_defaults, currName);
    
    inds = find(strcmp(currName, userDefinedFieldNames));
    
    if ~isempty(inds)
        isFieldNameValid(inds) = 1;
    else
        s_out = setfield(s_out, currName, currVal);
    end
end

if (sum(isFieldNameValid == 0) > 0)
    badFieldNums = find(isFieldNameValid == 0);
    
    if strcmp(nonExistantFieldAction, 'error')
    if numel(badFieldNums) == 1
        error('Bad field name ''%s''', userDefinedFieldNames{badFieldNums});
    else
        badFieldNames = '';
        for i=1:numel(badFieldNums)-1
            badFieldNames = cat(2, badFieldNames, userDefinedFieldNames{badFieldNums(i)}, ', ');
        end
        
        badFieldNames = cat(2, badFieldNames, userDefinedFieldNames{badFieldNums(numel(badFieldNums))});
        
        error('Bad field names: %s', badFieldNames);
    end
    
    elseif strcmp(nonExistantFieldAction, 'discard')
        s_out = rmfield(s_out, userDefinedFieldNames(badFieldNums));
    end    
end

end

