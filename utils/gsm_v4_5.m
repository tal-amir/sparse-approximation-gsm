function [mu, theta] = gsm_v4_5(z,k,gamma, isParallel, neglect_theta, use_mex)
%% Initialize variables
d = numel(z);

if ~exist('isParallel', 'var') || isempty(isParallel)
    isParallel = false;
end

if ~exist('neglect_theta', 'var') || isempty(neglect_theta)
    neglect_theta = false;
end

if ~exist('use_mex', 'var') || isempty(use_mex)
    use_mex = true;
end

%% Verify parameters
if (k > d) || (k < 0)
    error('k must be between 0 and numel(z)');
end

[mu, theta] = gsm_main(z,k,gamma, isParallel, neglect_theta, use_mex);

end

function [mu, theta] = gsm_main(z,k,gamma, isParallel, neglect_theta, use_mex)
d = numel(z);

[m_orig, n_orig] = size(z);
z = reshape(z, [1,d]);

%% Handle special cases
if k == 0
    mu = 0;
    theta = zeros(size(z));
elseif gamma == 0
    mu = (k/d)*sum(z);
    theta = ones(size(z)) * (k/d);
elseif gamma < 0
    [mu, theta] = gsm_main(-z, k, -gamma, isParallel, neglect_theta, use_mex);
    mu = -mu;
elseif gamma == inf
    [a, I_eq] = select(z, k);
    [~, I_gr] = find(z > a);

    mu = sum(z(I_gr)) + a*(k-numel(I_gr));
    
    theta = zeros(size(z));
    theta(I_gr) = 1;
    theta(I_eq) = (k - numel(I_gr)) / numel(I_eq);
elseif 2*k > d
    [mu, theta] = gsm_main(-z, d-k, gamma, isParallel, neglect_theta, use_mex);
    mu = sum(z) + mu;
    theta = 1-theta;
else
    % If none of the special cases takes place, run the main code
    if use_mex        
        khat = max(k,2*k-2);
        [m_orig, n_orig] = size(z);
        z = z(:);
        [z, I_sort, I_inv] = partial_sort(z, khat);        
        [mu, theta] = gsm_v4_5_mex(z, double(k), gamma);        
        theta = theta(I_inv);
        theta = reshape(theta, [m_orig, n_orig]);
    else
        [mu, theta] = gsm_reduced_case(z, k, gamma, isParallel, neglect_theta);
    end
end

theta = max(0, min(1, theta));

theta = reshape(theta, [m_orig, n_orig]);
end


function [mu, theta, b] = gsm_reduced_case(z,k,gamma, isParallel, neglect_theta)
%function [mu, theta, b] = gsm_reduced_case(z,k,gamma, isParallel, neglect_theta)
%
% TODO: Write description
% In this function assume that 0 < gamma < inf and k <= numel(z)/2
base = 1;
d = numel(z);

khat = max(k,2*k-2);

[z, I_sort, I_inv] = partial_sort(z, khat);
I1 = 1:khat;
I2 = (khat+1):d;

z1 = z(I1);
z2 = z(I2);

if ~isParallel
    [b, b2] = calc_b_consecutive(z, khat, gamma);
        
    b1 = calc_b_consecutive(z1, khat, gamma);
    %b2 = calc_b_consecutive(z2, khat, gamma);
else
    [b, b1, b2] = calc_b_parallel(z, khat, gamma);
    %b1 = calc_b_parallel(z1, khat, gamma);
    %b2 = calc_b_parallel(z2, khat, gamma);
end

mu = sum(z(1:k)) + b(k+base)/gamma;

if neglect_theta
    theta = [];
    return
end

%theta = calc_theta_right_test(z,k,gamma,I2,b);
theta = calc_theta_right(z,k,gamma,I2,b);

%for i=1:d
%    theta(i) = calc_theta_i(z, k, gamma, i, blarge);
%end

if isParallel
    for i=I1
        theta_hat_i = calc_theta_hat_i(z1, khat, gamma, i, b1);
        %theta_hat_i = calc_theta_i_back(z1,k,gamma, i, b1);
        theta(i) = calc_theta_i_left(z,k,khat,gamma,b, b1, b2, theta_hat_i);
    end
else
    for i=I1
        theta_hat_i = calc_theta_hat_i(z1, khat, gamma, i, b1);
        %theta_hat_i = calc_theta_i_back(z1,k,gamma, i, b1);
        theta(i) = calc_theta_i_left(z,k,khat,gamma,b, b1, b2, theta_hat_i);
    end
end


theta = theta(I_inv);

theta = reshape(theta,size(z));
end



function [b, b2] = calc_b_consecutive(z, s, gamma)
d = numel(z);

if s == 0
    b = 0;
    return
end

base = 1;

b2 = [];

% Initialize for r=d+1
btilde = nan(s+1,1);
btilde(0+base) = 0;

b = zeros(s+1,1);

vtilde = nan(s+1,1);
vtilde(0+base) = inf;
vtilde(1+base) = -inf;

v = nan(s+1,1);
v(0+base) = inf;
v((2:s)+base) = -inf;

for r=d:-1:1   
    % q=0 is handled implicitly
    for q=1:min(s, d-r)
        v(q+base) = max(min(z(r), vtilde(q-1+base)), vtilde(q+base));                     
        xi = gamma*(z(r)-vtilde(q+base));                
        eta = (btilde(q-1+base) - btilde(q+base)) + xi;
        
        if eta <= log((d-r-q+1)/q)
            b(q+base) = btilde(q+base)   - subplus( xi) + log1p(q/(d-r+1)*expm1(eta));
        else
            b(q+base) = btilde(q-1+base) - subplus(-xi) + log1p((d-r+1-q)/(d-r+1)*expm1(-eta));
        end                
        
        if isnan(b(q+base))
            fprintf('nan!\n');
        end
    end
    
    % q = d-r+1
    if s >= d-r+1
        v(d-r+1+base) = min(vtilde(d-r+base), z(r));
        b(d-r+1+base) = 0;
    end
    
    if r == s+1
        b2 = b;
    end
        
    btilde = b;
    vtilde = v;
end

b = btilde;

end




function [b, b1, b2] = calc_b_parallel(z, k, gamma)
d = numel(z);

assert(k <= d);

p = ceil(d/k);

if p == 1
    b = calc_b_consecutive(z, k, gamma);
    b1 = b;
    b2 = [];
    return
end

zvecs = cell(1,p);

zsvecs = cell(1,p);
bvecs = nan(k+1,p);
dvec = nan(1,p);

for i=1:p
    zvecs{i} = z((i-1)*k+1 : min(i*k,d));
    zsvecs{i} = get_order_stats(zvecs{i},k);
    
    bvecs(:, i) = calc_b_consecutive(zvecs{i}, k, gamma);
    dvec(i) = numel(zvecs{i});
    
    if any(isnan(bvecs(:, i)))
        fprintf('nan!\n');
    end
end

b1 = bvecs(:,1);
zs1 = zsvecs{1};
d1 = dvec(1);

bvecs = bvecs(:,2:end);
zsvecs = zsvecs(2:end);
dvec = dvec(2:end);

while numel(dvec) > 1
    [bhat, zhats] = merge(bvecs(:,1), bvecs(:,2), zsvecs{1}, zsvecs{2}, dvec(1), dvec(2), gamma);    
    
    dhat = dvec(1) + dvec(2);   
    
    bvecs = [bvecs(:,3:end), bhat];
    zsvecs = [zsvecs(:,3:end), zhats];
    dvec = [dvec(3:end), dhat];
end

b2 = bvecs;
zs2 = zsvecs{1};
d2 = dvec;

b = merge(b1, b2, zs1, zs2, d1, d2, gamma);

end



function theta = calc_theta_right(z,k,gamma,I,b)
base = 1;
d = numel(z);

theta = nan(1,d);

theta(I) = exp(gamma*(z(I)-z(1)) - b(base+1)) / d;
%theta(I) = exp(gamma*(z(I)-z(1))) / sum(exp(gamma*(z-z(1))));
theta(I) = max(0,min(1,theta(I)));

facs_curr = nan(1,numel(I));

for q=2:k
    theta(I) = (q / (d-q+1)) * exp(gamma*(z(I)-z(q)) + (b(base+q-1) - b(base+q))) .* (1-theta(I));
    
    %facs = (q / (d-q+1)) * exp(gamma*(z(I)-z(q)) + (b(base+q-1) - b(base+q)));    
    %theta(I) = facs - (facs.*theta(I));
    
    %max(abs((q / (d-q+1)) * exp(gamma*(z(I)-z(q)) + (b(base+q-1) - b(base+q)))))
    %max(abs((1-theta(I))))
    
    theta(I) = max(0,min(1,theta(I)));
    
    facs_prev = facs_curr;
    facs_curr = log(q / (d-q+1)) + gamma*(z(I)-z(q)) + b(base+q-1) - b(base+q);
    
    %if max(exp(facs_curr(I_small))) > (1+1e-7)*q
    %    error('prob #1: q=%d, val=%g\n', q, max(exp(facs_curr(I_small))));
    %end
    
    %rel_logerr = max(facs_prev(:)-facs_curr(:)) / max(abs(facs_prev));
    
    %if rel_logerr > 1e-15
        %error('prob #2: q=%d, val=%g\n', q, rel_logerr);
    %end
    
end
    
end



function theta_hat_i = calc_theta_hat_i(z,k,gamma, i, bz_hat)
base = 1;
d = numel(z); 

theta_hat_i = nan(k,1);

theta_hat_i(1) = exp(gamma*(z(i)-z(1)) - bz_hat(base+1))/d;
%theta_hat_i(1) = exp(gamma*(z(i)-z(1)))/sum(exp(gamma*(z-z(1))));
 
if theta_hat_i(1) > 1
    %warning('This should not happen');
end

etas = nan(k,1);

q_hat = k+1;

for q=2:k
    eta = log(q/(d-q+1)) + gamma*(z(i)-z(q)) + (bz_hat(base+q-1) - bz_hat(base+q));
    eta2 = gamma*(z(i)-z(q)) + (bz_hat(base+q-1) - bz_hat(base+q));
    
    if eta <= 0
        %theta_hat_i(q) = exp(eta)*(1-theta_hat_i(q-1));
        theta_hat_i(q) = (q/(d-q+1) * exp(eta2)) *(1-theta_hat_i(q-1));
        etas(q) = eta;
    else
        q_hat = q;
        break;
    end
end

theta_hat_i(k) = 1;

had_prob = false;

for q = k-1:-1:q_hat
    %eta = (d-q+1)/q*exp(bz_hat(base+q) - gamma_z_hat_i - bz_hat(base+q-1));
    %eta = (d-q)/(q+1)*exp(bz_hat(base+q+1) - gamma_z_hat_i - bz_hat(base+q));
    eta = log((q+1)/(d-q)) + gamma*(z(i)-z(q+1)) + (bz_hat(base+q) - bz_hat(base+q+1));
    eta2 = gamma*(z(i)-z(q+1)) + (bz_hat(base+q) - bz_hat(base+q+1));
    
    etas(q) = eta;
    
    if eta < 0      
        had_prob = true;
        %error('This should not happen');
    end
    
    theta_hat_i(q) = 1 - (((d-q)/(q+1))*exp(-eta2))*(theta_hat_i(q+1));
end

end



function theta_i = calc_theta_i_left(z,k,khat,gamma,bz, bz_hat, bz_check, theta_z_hat_i)
base = 1;

d = numel(z);

n1 = khat;
n2 = d-khat;

z1s = z(1:khat);
z2s = get_order_stats(z((khat+1):d), khat);

% delta_{q,t}(z1,z2) for t=0,...,k
delta = init_delta(k);

[alpha, log_alpha] = init_alpha(k);


theta_i = 0;

for q=1:k
    [alpha, log_alpha] = update_alpha(n1, n2, q, alpha, log_alpha);
    delta = update_delta(delta, n1, n2, q, z1s, z2s, z1s);
end

clear q

ta = max(1, k-n2);
tb = min(k, n1);

for t=ta:tb
    theta_i = theta_i + ...
        theta_z_hat_i(t) * ...
        exp(log_alpha(t+base) - gamma*delta(t+base) + bz_hat(t+base) + bz_check(k-t+base) - bz(k+base));
end

end



function [z_sort, I_sort, I_inv] = partial_sort(z, k)

if false && exist('maxk','builtin')
    [~, I_max] = maxk(z,k);        
    [~, I_min] = mink(z,numel(z)-k);        
else
    [~, I_sort] = sort(z,'descend');
    I_sort(2*k+1:end) = sort(I_sort(2*k+1:end));
end

z_sort = z(I_sort);

I_inv = nan(size(I_sort));
I_inv(I_sort) = 1:numel(z);
end


function [a, I] = select(z, k)
%function [a, I] = select(z, k)
%
% For a vector z in R^d and an integer k, returns the k'th largest element
% of z and a list of the indices of Z that contain it.
%
% a - z_(k) - the k'th largest element of z. If k == 0 or k > d, returns
%             inf and -inf respectively.
%
% I - The indices of z which contain the value a

if k <= 0
    a = inf;
elseif k > numel(z)
    a = -inf;
else
    zs = sort(z,'descend');
    a = zs(k);
end

I = find(z == a);
end


function zs = get_order_stats(z,k)
%function zs = get_order_stats(z,k)
%
% Calculates the k first order statistics of the vector z, padding with
% -inf if k > numel(z).
zs = sort(z,'descend');
if numel(zs) > k
    zs = zs(1:k);
else
    zs = [zs(:); -inf(k-numel(zs),1)]';
end
end


function [bhat, zhats] = merge(b1, b2, z1s, z2s, n1, n2, gamma)
base = 1;

assert(numel(z1s) == numel(z2s));
k = numel(z1s);

% Calculate order statistics of concatenated vector zhat
zhats = calc_zhat_sort(z1s, z2s);

% Initialize for q=0
% alpha^{n1,n2}_{q,t} for t=0,...,k
% ell are the logarithms of alpha
[alpha, log_alpha] = init_alpha(k);

% beta_{q,t}(z1,z2) for t=0,...,k
beta = nan(k+1,1);

% b_{q,gamma}(zhat) for q=0,...,k
bhat = nan(k+1,1);

% delta_{q,t}(z1,z2) for t=0,...,k
delta = init_delta(k);

% b_{0,gamma}(zhat) is zero by definition
bhat(0+base) = 0;

for q=1:k
    ta = max(0, q-n2);
    tb = min(q, n1);
    
    [alpha, log_alpha] = update_alpha(n1, n2, q, alpha, log_alpha);
    
    %for t=ta:tb
    %    log_alpha(t+base) = lognchoosek(n1,t) + lognchoosek(n2,q-t) - lognchoosek(n1+n2,q);
    %end
    
    delta = update_delta(delta, n1, n2, q, z1s, z2s, zhats);
    
    if ta > tb
        bhat(q+base) = 0;
        continue;
    end
    
    maxval = -inf;
    
    inds = tb:-1:ta;
    
    for t = inds
        beta(t+base) = b1(t+base) + b2(q-t+base) - gamma*delta(t+base);
        
        if beta(t+base) >= maxval
            %maxval = beta(t+base) + ell(t+base);
            maxval = beta(t+base);
            t_max = t;
        end
    end
    
    if any(isnan(beta(inds+base)))
        fprintf('nan\n');
    end

    bhat(q+base) = logmeanexp(alpha(inds+base), log_alpha(inds+base), beta(inds+base));

    if any(isnan(bhat(inds+base)))
        fprintf('nan\n');
    end    
end
end


function out = logmeanexp(alpha, log_alpha, beta)
%function out = logmeanexp(alpha, log_alpha, beta)
%
% Calculates log(sum(alpha .* exp(beta))) in a numerically safe way, where
% alpha = [alpha_1, ..., alpha_r]
% beta  = [beta_1, ..., beta_r]
% and log_alpha = log(alpha)
% such that all(alpha > 0) and sum(alpha) = 1.

if ~exist('log_alpha', 'var')
    log_alpha = [];
end

beta_max = max(beta);
S = sum(alpha .* expm1(beta-beta_max));

if S >= -0.5
    % Use first method
    out = log1p(S) + beta_max;
else
    % If preferable, use second method
    S = sum(alpha .* exp(beta-beta_max));
    %fprintf('S = %f\n', S);
    
    if (S >= 1e-30) || isempty(log_alpha)
        out = log(S) + beta_max;
    else
        v = beta+log_alpha;
        [v_max, i_max] = max(v);
        I = 1:numel(v);
        out = log1p(sum(exp(v(I~=i_max) - v_max))) + v_max;
    end    
end

end


function [alpha, log_alpha] = init_alpha(k)
%function [alpha, log_alpha] = init_alpha(k)
%
% Calculates the binomial coefficients alpha^{n1,n2}_{q,t} for t=0,1,...,k
% and their logarithms, where q=0 and 1 <= k <= n1+n2
base = 1;

% The case q=0
alpha = zeros(k+1,1);
alpha(0+base) = 1;

log_alpha = -inf(k+1,1);
log_alpha(0+base) = 0;
end


function [alpha, log_alpha] = update_alpha(n1, n2, q, alpha, log_alpha)
%function [alpha, log_alpha] = update_alpha(n1, n2, q, alpha, log_alpha)
%
% Calculates the binomial coefficients alpha^{n1,n2}_{q,t} for t=0,1,...,k
% and their logarithms, where 1 <= q <= k <= n1+n2
%
% Takes as input alpha^{n1,n2}_{q-1,t} for t=0,1,...,k, and the respective
% logarithms.
%
% If the logarithms are not required, log_alpha can be left empty in order
% to save time.

base = 1;

k = numel(alpha)-1;

assert(1 <= q);
assert(q <= k);
assert(k <= n1+n2);

% Numerical threshold. If a coefficient is above its threshold, we
% calculate its logarithm directly. This threshold ensures that it is
% only negligibly affected by numerical underflow. Coefficients below this
% threshold are calculated recursively.
if isa(alpha,'double')
    realmin_curr = eps(realmin('double'));
    eps_curr = eps(double(1));
else
    realmin_curr = eps(realmin('single'));
    eps_curr = eps(single(1));
end

log_thresh = (log(realmin_curr) - log(eps_curr)) - log(2);

% Range of t values
ta = max(0, q-n2);
tb = min(q, n1);

% Calculate alpha recursively, marching backwards from t = tb
for t = tb:-1:max(ta,1)    
    if alpha(t+base) >= alpha(t-1+base)        
        alpha(t+base) = alpha(t+base) + ( alpha(t-1+base) + (n1-t)*(alpha(t-1+base)-alpha(t+base)) ) / (n1+n2-q+1);
    else
        alpha(t+base) = alpha(t-1+base) + ( alpha(t+base) + (n2-q+t)*(alpha(t+base)-alpha(t-1+base)) ) / (n1+n2-q+1);        
    end
    
    assert(max(alpha) <= 1);
end

if ta == 0
    alpha(0+base) = ((n2-q+1)/(n1+n2-q+1)) * alpha(0+base);    
else
    alpha(ta-1+base) = 0;
end

if tb < k
    alpha(tb+1+base) = 0;
end

% If log_alpha is empty, the user does not require log_alpha and we leave
% it empty in order to save computation time.
if isempty(log_alpha)
    return
end

% Index t of the mode of alpha_{q,0},...,alpha_{q,k}
t_mode = floor((n1+1)*(q+1)/(n1+n2+2));
assert(t_mode >= ta);
assert(t_mode <= tb);

log_alpha(t_mode+base) = log(alpha(t_mode+base));

% Calculate log_alpha recursively, marching backwards from modal t
for t = (t_mode-1):-1:ta
    fac = ((t+1)*(n2-(q-t)+1))/((n1-t)*(q-t));
    assert(fac <= 1);
    
    if (alpha(t+base) > 0) && (log_alpha(t+base) >= log_thresh)
        % If there is no risk of underflow, calculate the log directly
        log_alpha(t+base) = log(alpha(t+base));
    elseif (fac <= 0.5)
        % Otherwise, calculate the log recursively
        log_alpha(t+base) = log_alpha(t+1+base) + log(fac);
    else
        % If the multiplied factor is closer to 1 than to 0, use log1p
        log_alpha(t+base) = log_alpha(t+1+base) + ...
            log1p( ( (n2+1)*(t+1) - (n1+1)*(q-t) ) / ( (n1-t)*(q-t) ));
    end    
end

if ta > 0
    log_alpha(ta-1+base) = -inf;
end

% Calculate log_alpha recursively, marching forward from modal t
for t = (t_mode+1):tb
    fac = ((n1-t+1)*(q-t+1))/(t*(n2-q+t));
    assert(fac <= 1);
    
    if (alpha(t+base) > 0) && (log_alpha(t+base) >= log_thresh)
        % If there is no risk of underflow, calculate the log directly
        log_alpha(t+base) = log(alpha(t+base));
    elseif fac <= 0.5
        % Otherwise, calculate the log recursively
        log_alpha(t+base) = log_alpha(t-1+base) + log(fac);
    else
        % If the multiplied factor is closer to 1 than to 0, use log1p
        log_alpha(t+base) = log_alpha(t-1+base) + ...
            log1p( ((n1+1)*(q-t+1) - t*(n2+1)) / (t*(n2-q+t)) );
    end    
end

if tb < k
    log_alpha(tb+1+base) = -inf;
end

end


function zhats = calc_zhat_sort(z1s, z2s)
%function zhats = calc_zhat_sort(z1s, z2s)
%
% Given the k first order statistics (z1s(1),...,z1s(k)) and
% (z2s(1),...,z2s(k)) of vectors z1 and z2 respectively, calculates the k
% first order statistics (zhats(1),...,zhats(k)) of the concatenated vector
% zhat = [z1, z2]. If any of the vectors has less than k entries, the order
% statistics are padded by -inf at the end.

k = numel(z2s);
assert(numel(z2s) == k);

zhats = nan(1,k);

il = 1;
ir = 1;

% Merge the order statistics
for ih = 1:k
    if (il <= k) && (ir <= k) && (z1s(il) >= z2s(ir))
        zhats(ih) = z1s(il);
        il = il + 1;
    elseif ir <= k
        zhats(ih) = z2s(ir);
        ir = ir + 1;
    else
        zhats(ih) = z1s(il);
        il = il + 1;
    end
end

assert(il+ir == k+2);
assert(issorted(-zhats));
end


function delta = init_delta(k)
%function delta = init_delta(k)
%
% Calculates delta_{q,t}(zhat) for t=0,1,...,k
% where zhat = [z1, z2], q=0 and k >= 1

delta = zeros(k+1,1);
end


function delta = update_delta(delta, nleft, nright, q, zleft_sort, zright_sort, zhat_sort)
%function delta = update_delta(delta, nleft, nright, q, zleft_sort, zright_sort, zhat_sort)
%
% Calculates delta_{q,t}(zhat) for t=0,1,...,k
% where zhat = [z1, z2], and 1 <= q <= k
%
% Takes as input delta_{q-1,t}(zhat) for t=0,1,...,k
base = 1;

k = numel(delta);
assert(k >= q);

ta = max(0, q-nright);
tb = min(q, nleft);

for t=tb:-1:max(ta,1)
    if zhat_sort(q) >= zleft_sort(t)
        delta(t+base) = delta(t-1+base) + (zhat_sort(q) - zleft_sort(t));
    else
        delta(t+base) = delta(t+base) + (zhat_sort(q) - zright_sort(q-t));
        assert((zhat_sort(q) - zright_sort(q-t)) >= 0);
    end
end

if ta == 0
    delta(0+base) = delta(0+base) + (zhat_sort(q) - zright_sort(q));
    assert( (zhat_sort(q) - zright_sort(q)) >= 0 );
else
    delta(ta-1+base) = 0;
end

end

