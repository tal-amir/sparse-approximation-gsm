% This script compares GSM, DC-Programming and ADMM in solving a
% trimmed-lasso penalized problem
%
% (P_lambda) min_x ||Ax-y||_2^2 + lambda tau_k(x)
%
% This script requires the Mosek optimization solver:
%     https://www.mosek.com/downloads/
% and the Yalmip modeling toolbox:
%     https://yalmip.github.io/download/

addpath('./utils');
addpath('./comparison');

% Generates a (n x d) matrix A, a k-sparse signal x0, and a noisy
% sample y = A*x0 + e, and tries to recover x0 from y.
n = 60; d = 300;

k_vals = [10,20,25];
markers = {'o', 's', 'p'};
markerColors = [0,0,0.5; 0.1700, 0.5917, 0.1383; 0.902, 0, 0];

% lambda_rel = lambda / lambda_bar
lambda_rel = 0.2;

% Base lambda_rel. Obtained solution is used as init with lambda_rel.
lambda_rel_base = 1e-5;

% eta used in Trimmed Lasso
eta_vals = [1e-2];
% eta_vals = [1e-5, 1e-2]; Takes more time

% Relative noise level. Set to 5% noise.
nu = 0.01;

nTests = 200;

% Display live-updated results in figures
display_figures = true;

% Detect Mosek
if isempty(which('mosekopt'))
    error('This script requires the Mosek optimization solver. https://www.mosek.com/downloads/');
end

% Detect Yalmip
if isempty(which('yalmip'))
    error('This script requires the Yalmip modeling toolbox. https://yalmip.github.io/download/');
end

% Display settings
figSize = [480,360];
markerSize = 50;
legendFontSize = 12;
legendLocation = 'east';

label_gsm = '$$\mathrm{F}_{\lambda}\left(\hat{\bf x}_{\mathrm{GSM}}\right) / \mathrm{F}_{\lambda}\left({\bf x}_{0}\right)$$';
label_dcp = '$$\mathrm{F}_{\lambda}\left(\hat{\bf x}_{\mathrm{DCP}}\right) / \mathrm{F}_{\lambda}\left({\bf x}_{0}\right)$$';
label_admm = '$$\mathrm{F}_{\lambda}\left(\hat{\bf x}_{\mathrm{ADMM}}\right) / \mathrm{F}_{\lambda}\left({\bf x}_{0}\right)$$';


% Fix random seed
rng(12345);


% Report
nn = 8214;
s0 = 8320;
s2 = 8322;
s1 = 8321;
sk = char(8342);
sup2 = char(178);
ofx = '(x)';
ofx0 = ['(x',s0,')'];

uTruncx = [char(928), char(8342), '(x)'];
uProjx = 'Proj(x)';
uLambda = char(955);
uTau = char(964);

fprintf('\nComparing GSM, DC-Programming and ADMM in minimizing %s\n\n', ['F', sk, ofx]);

fprintf(['F', sk, ofx, ' = 0.5*', nn, 'Ax-y', nn, s2, sup2, ' + ', uLambda, '*', uTau, sk, ofx, '\n']);
fprintf(['normalied_objective(x) = ', 'F', sk, ofx, ' / ', 'F', sk, ofx0, '\n']);

fprintf('\nMatrix size: %d x %d\n', n,d);

k_vals_str = sprintf('%g,',k_vals);
fprintf('Sparsity values: %s\n', k_vals_str(1:end-1));

fprintf('Noise level: %g%%\n', 100*nu);

fprintf('\nStarting %d tests. ',nTests);
fprintf('Results are saved in ''result'' struct array.\n');
fprintf('Reporting normalized objectives:\n');

t_all = tic;

result = [];

if display_figures
    close all
    
    screenSize = get(groot,'Screensize'); screenSize = screenSize(3:4);
    dFig = [figSize(1)/2, 0] * 1.2;
    pos_dcp = [round(screenSize/2 - figSize/2 - dFig), figSize];
    pos_admm = [round(screenSize/2 - figSize/2 + dFig), figSize];
    
    f_dcp = figure('Name', 'DCP', 'Position', pos_dcp);
    title('GSM vs. DC-Programming'); hold on;
    
    f_admm = figure('Name', 'ADMM', 'Position', pos_admm); hold on;
    title('GSM vs. ADMM'); hold on;
end

for i=1:nTests
    ik = mod(i-1, numel(k_vals)) + 1;
    k = k_vals(ik);
    
    % Generate a gaussian dictionary with normalized columns
    A = randn([n,d]);
    A = A ./ repmat(sqrt(sum(A.^2,1)),[n,1]);
    
    % Gaussian k-sparse signal x0
    x0 = zeros(d,1);
    S = randperm(d,k);
    x0(S) = randn([k,1]);
    
    % Noisy sample
    y = A*x0;
    e = randn([n,1]);
    e = e / norm(e) * norm(y) * nu;
    y = y+e;
    
    % Objective function
    lambda_bar = sqrt(max(sum(A.^2,1)))*norm(y);
    Fk = @(x) 0.5*norm(A*x-y)^2 + lambda_rel*lambda_bar*trimmedLasso(x,k);
    
    t_curr = tic;
    
    x_dcp  = solve_P_lambda_tls(A, y, k, 'dcp', eta_vals, lambda_rel, lambda_rel_base);
    x_admm = solve_P_lambda_tls(A, y, k, 'admm', eta_vals, lambda_rel, lambda_rel_base);
    x_gsm  = solve_P_lambda_gsm(A, y, k, lambda_rel, lambda_rel_base);
    
    t_curr = toc(t_curr);
    
    obj_x0 = Fk(x0);
    nobj_gsm  = Fk(x_gsm)  / obj_x0;
    nobj_dcp  = Fk(x_dcp)  / obj_x0;
    nobj_admm = Fk(x_admm) / obj_x0;
    
    fprintf('%3d) k=%d  t[s]=%d  GSM: %g  DC-Programming: %g  ADMM: %g\n', i, k, round(t_curr), nobj_gsm, nobj_dcp, nobj_admm);
    
    result_curr = struct();
    
    result_curr.k = k;
    result_curr.nobj_gsm = nobj_gsm;
    result_curr.nobj_dcp = nobj_dcp;
    result_curr.nobj_admm = nobj_admm;
    
    result = [result, result_curr];
    
    if display_figures
        % DCP figure
        clf(f_dcp);
        legend_strings = {};
        
        for iik = 1:min(numel(k_vals),i)
            kc = k_vals(iik);
            markerStyle = markers{iik};
            markerColor = markerColors(iik,:);
            add_lines = (iik == min(numel(k_vals),i));
            legend_strings{iik} = sprintf('$$k = %d$$', kc);
            
            
            x_curr = [result([result.k] == kc).nobj_gsm];
            y_curr = [result([result.k] == kc).nobj_dcp];
            
            f_dcp = plotTwoMethodComparison_add(f_dcp, 'GSM vs. DC-Programming', x_curr, label_gsm, y_curr, label_dcp, markerColor, [], 'sizeRange', [markerSize, markerSize], 'markerStyle', markerStyle, 'xscale', 'log', 'yscale', 'log', 'add_lines', false);
        end
        
        x_all = [result.nobj_gsm];
        y_all = [result.nobj_dcp];
        
        vmin = min(min(x_all),min(y_all));
        vmax = max(max(x_all),max(y_all));
        
        xlim([vmin, vmax]);
        ylim([vmin, vmax]);
        
        [~,icons] = legend(legend_strings, 'location', legendLocation, 'interpreter', 'latex', 'FontSize', legendFontSize);
        icons = findobj(icons,'type','patch');
        set(icons,'MarkerSize', sqrt(markerSize));
        
        plot([vmin, vmax], [vmin, vmax], 'k');
        plot([vmin, vmax], [1, 1], 'k');
        plot([1,1], [vmin, vmax], 'k');
        
        % ADMM figure
        clf(f_admm);
        legend_strings = {};
        
        for iik = 1:min(numel(k_vals),i)
            kc = k_vals(iik);
            markerStyle = markers{iik};
            markerColor = markerColors(iik,:);
            add_lines = (iik == min(numel(k_vals),i));
            legend_strings{iik} = sprintf('$$k = %d$$', kc);
            
            
            x_curr = [result([result.k] == kc).nobj_gsm];
            y_curr = [result([result.k] == kc).nobj_admm];
            
            f_admm = plotTwoMethodComparison_add(f_admm, 'GSM vs. ADMM', x_curr, label_gsm, y_curr, label_admm, markerColor, [], 'sizeRange', [markerSize, markerSize], 'markerStyle', markerStyle, 'xscale', 'log', 'yscale', 'log', 'add_lines', false);
        end
        
        x_all = [result.nobj_gsm];
        y_all = [result.nobj_admm];
        
        vmin = min(min(x_all),min(y_all));
        vmax = max(max(x_all),max(y_all));
        
        xlim([vmin, vmax]);
        ylim([vmin, vmax]);
        
        [~,icons] = legend(legend_strings, 'location', legendLocation, 'interpreter', 'latex', 'FontSize', legendFontSize);
        icons = findobj(icons,'type','patch');
        set(icons,'MarkerSize', sqrt(markerSize));
        
        plot([vmin, vmax], [vmin, vmax], 'k');
        plot([vmin, vmax], [1, 1], 'k');
        plot([1,1], [vmin, vmax], 'k');        
    end
    
    drawnow
end

t_all = toc(t_all);
fprintf('\nDone. Time elapsed [sec.]: %d\n', round(t_all));


