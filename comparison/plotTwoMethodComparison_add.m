function h = plotTwoMethodComparison_add(h, titleStr, objx, labelx, objy, labely, colorVals, sizeVals, varargin)
%function h = plotTwoMethodComparison_add(h, titleStr, objx, labelx, objy, labely, colorVals, sizeVals, varargin)

if isempty(h)
    h = figure(); hold on;
else
    figure(h); hold on;
end

%% Set default parameters
dp = struct();

dp.markerStyle = 'o';
dp.xscale = 'lin';
dp.yscale = 'lin';
dp.FontName = 'Helvetica';
dp.axisLabelFontSize = 12;
dp.titleFontSize = 15;
dp.colorbarFontSize = 12;
dp.useLatex_xlabel = true;
dp.useLatex_ylabel = true;
dp.sizeRange = [4,50];
dp.add_lines = true;

params = processParams(varargin, dp);

I_success = find(objx <= 1);
I_fail = find(objx > 1);

if isempty(sizeVals) || (numel(unique(sizeVals)) == 1)
    sizeVals = 0.5*(max(params.sizeRange)+min(params.sizeRange));
else
    sizeVals = sizeVals- min(sizeVals);
    sizeVals = sizeVals / max(sizeVals);
    sizeVals = sizeVals*(max(params.sizeRange)-min(params.sizeRange)) + min(params.sizeRange);
end

scatter(objx, objy, sizeVals, colorVals, params.markerStyle); hold on;

%I = I_fail;
%scatter(objx(I), objy(I), sizeVals(I), colorVals(I), params.markerStyle);
%drawnow

%I = I_success; 
%scatter(objx(I), objy(I), sizeVals(I), colorVals(I), params.markerStyle);

if strcmp(params.xscale, 'log')
    set(gca, 'xscale', 'log');
end

if strcmp(params.yscale, 'log')
    set(gca, 'yscale', 'log');
end

vmin = min(min(objx),min(objy));
vmax = max(max(objx), max(objy));

%p1 = [min(obj1); min(obj2)];
%p2 = [max(obj1); max(obj2)];
%nPoints = 1000;
%points = exp(log(p1)*ones(1,nPoints) + (log(p2)-log(p1))*(0:nPoints-1)/(nPoints-1));
%loglog(points(1,:), points(2,:), 'k');

if params.add_lines
    plot([vmin, vmax], [vmin, vmax], 'k');
    plot([vmin, vmax], [1, 1], 'k');
    plot([1,1], [vmin, vmax], 'k');
end

xlim([min(min(objx),min(objy)), max(max(objx),max(objy))]);
ylim([min(min(objx),min(objy)), max(max(objx),max(objy))]);

title(titleStr, 'interpreter', 'latex', 'FontName', params.FontName, 'FontSize', params.titleFontSize);
if params.useLatex_xlabel
    xlabel(labelx, 'interpreter', 'latex', 'FontName', params.FontName, 'FontSize', params.axisLabelFontSize);
else
    xlabel(labelx, 'FontName', params.FontName, 'FontSize', params.axisLabelFontSize);
end

if params.useLatex_ylabel
    ylabel(labely, 'interpreter', 'latex', 'FontName', params.FontName, 'FontSize', params.axisLabelFontSize);
else
    ylabel(labely, 'FontName', params.FontName, 'FontSize', params.axisLabelFontSize);
end

end


function params = processParams(paramsArgList, defaultParams)
% This function takes the user-defined parameter overrides, verifies them
% and returns a struct that contraints the complete set of parameters.

% Convert cell array of name / value pairs to a struct
if numel(paramsArgList) == 0
    params = struct();    
elseif numel(paramsArgList) == 1
    params = paramsArgList{1};
    
    if isempty(params)
        params = struct();
    end
else
    params = namevals2struct(paramsArgList);
end

% Add all missing parameters
params = addDefaultFields(params, defaultParams);

%% Post-process and verify parameters
if ~ismember(params.xscale, {'log', 'lin'})
    error('Parameter ''xscale'' should be ''log'' or ''lin''');
end

if ~ismember(params.yscale, {'log', 'lin'})
    error('Parameter ''yscale'' should be ''log'' or ''lin''');
end

end

