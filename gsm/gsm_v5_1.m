%function [mu, theta, info] = gsm_v5_1(z, k, gamma, {optional:} use_mex_code)
%
% Generalized Soft-Min
% Tal Amir, Ronen Basri, Boaz Nadler
% Weizmann Institute of Science
%
% Contact: tal.amir@weizmann.ac.il
%
% Version 5.1, June 2020
%
% Given a vector z in R^d, 0 <= k <= d and gamma in [-inf, inf], calculates the 
% Generalized Soft-Min function mu_{k,gamma}(z) and its gradient, denoted by 
% theta_{k,gamma}(z). 
%
% Based on the paper [1]. All notations below are defined in the paper.
% 
% Input: z - A d-dimensional vector of double or single class
%        k - An integer between 0 and d
%        gamma - A real number in the interval [-inf, inf]
%        use_mex_code - (optional) A boolean that tells whether to use the faster mex code 
%                       or the slower matlab implementation. Defaults to true if left blank.           
%
% Output: mu     The function mu_{k,gamma}(z) 
%         theta  The vector function theta_{k,gamma}(z), of the same size and shape as z
%         info   A struct containing the following fields:
%                * k_base, gamma_base - In most cases, calculation is done by a redution 
%                   to the case gamma >= 0, 0 <= k <= d/2. When using a reduction, these
%                   are the parameters k,gamma used in the reduction. 
%                * dl,dr - Integers dl = max(1,2*k-2), dr = d-dl.
%                  z is sorted and split to two vectors zl, zr of sizes dl and dr
%                  respectively.
%                * Il, Ir - The vectors zl, zr can be calculated by zl = z(Il) and 
%                  zr = z(Ir).
%                * bz  - b_{q,gamma}(z)  for q=0,...,k.
%                * bzl - b_{q,gamma}(zl) for q=0,...,dl.
%                * bzr - b_{q,gamma}(zr) for q=0,...,min(dr,k), where dr=d-dl.
%                * delta - Delta_{k,t}(zl,zr) for t=0,...,k.
%                * alpha, log_alpha - The coefficients alpha^{dl,dr}_{k,t} and their 
%                  logarithms, for t=0,..k.
%
% [1] Tal Amir, Ronen Basri, Boas Nadler - The Trimmed Lasso: Sparse
%     recovery guarantees and practical optimization by the Generalized
%     Soft-Min Penalty.
%
% Note: If z is of class 'single', all calculations are done by single-precision floating
%       floating point arithmetic. To use this feature, <use_mex_code> must be set to false.
%       Another option to use single precision is to recompile the gsm_v#_#_mex.c source
%       with the appropriate flags with the <gsm_intermediate_numerical_type> defined as 1
%       and <gsm_use_matlab_arithmetic> defined as true.
function [mu, theta, info] = gsm_v5_1(z, k, gamma, use_mex_code)
% Normally z = [zl, zr] is sorted such that min(zl) >= max(zr). Turn this flag to do a 
% full increasing sort on z instead. This increses the runtime complexity from O(d*k) to
% O(d*k + k*log(k)), but may practically reduce the running time if many entries of z are 
% equal.
full_sort = true;

% If set to true, theta is calculated by an alternative O(d^2)-time algorithm, which may 
% be more accurate. This setting is only applicable if <use_mex_code> = false.
slow_mode = false;

if ~exist('use_mex_code', 'var') || isempty(use_mex_code)
    use_mex_code = true;
end

verify_parameters(z, k, gamma);

if isa(z, 'single')
    k = single(k);
    gamma = single(gamma);
    numcast = @(x) single(x);
elseif isa(z, 'double')
    k = double(k);
    gamma = double(gamma);
    numcast = @(x) double(x);
end

[mu, theta, info] = alg8(numcast, z, k, gamma, use_mex_code, full_sort, slow_mode);
end


function verify_parameters(z, k, gamma)
if ~isreal(z) || any(isnan(z)) || any(isinf(z))
    error('z must be a real vector with no NaNs or infs');
end

if ~isa(z, 'double') && ~isa(z, 'single')
    error('z must be a vector of class ''single'' or ''double''');
end

if numel(z) < 2
    error('z must contain at least two entries');
end

d = numel(z);

if ~isscalar(k) || (k < 0) || (k > d) || isnan(k) || (round(k) ~= k)
    error('k must be an integer between 0 and numel(z)');
end

if ~isscalar(gamma) || ~isreal(gamma) || isnan(gamma)
    error('gamma must be a number in [-inf, inf]');
end

end


% Main function for calculating mu_{k,gamma}(z) and theta_{k,gamma}(z) for 0 <= k <= d
% and -inf <= gamma <= inf. Based on Algorithm 8 in [1].
% This function handles special cases k=0,d and gamma=0,+-inf separately, and otherwise 
% uses a reduction to 1 <= k <= d/2, 0 < gamma < inf.
function [mu, theta, info] = alg8(numcast, z, k, gamma, use_mex_code, full_sort, slow_mode)
d = numcast(numel(z));

info = struct();
info.version = '5.1';
info.k_base = [];
info.gamma_base = [];
info.dl = [];
info.dr = [];
info.Il = [];
info.Ir = [];
info.bz = [];
info.bzl = [];
info.bzr = [];
info.delta = [];
info.alpha = [];
info.log_alpha = [];

if k == 0
    mu = numcast(0);
    theta = numcast(zeros(size(z)));
    
    info.k_base = numcast(0);
    info.gamma_base = gamma;   
elseif gamma == 0
    mu = numcast((k/d)*sum(z));
    theta = numcast((k/d)*ones(size(z)));

    info.k_base = numcast(k);
    info.gamma_base = numcast(0);   
elseif gamma < 0
    [mu, theta, info] = alg8(numcast, -z, k, -gamma, use_mex_code, full_sort, slow_mode);
    mu = -mu;
elseif gamma == inf
    a = min(maxk(z,k));
    I_eq = find(z == a);
    I_gr = find(z > a);

    mu = sum(z(I_gr)) + numcast(a*(k-numel(I_gr)));
    
    theta = numcast(zeros(size(z)));
    theta(I_gr) = numcast(1);
    theta(I_eq) = numcast((k - numel(I_gr)) / numel(I_eq));
    
    info.k_base = k;
    info.gamma_base = numcast(inf);
elseif 2*k > d
    [mu, theta, info] = alg8(numcast, -z, d-k, gamma, use_mex_code, full_sort, slow_mode);
    mu = mu + sum(z);
    theta = 1-theta;
else
    % Here 1 <= k <= d/2 and 0 < gamma < inf
    [mu, theta, info] = calc_gsm_main(numcast, z, k, gamma, use_mex_code, full_sort, slow_mode);
end
end


% This function calculates the GSM under the assumption 1 <= k <= d/2, 0 < gamma < inf.
function [mu, theta, info] = calc_gsm_main(numcast, z, k, gamma, use_mex_code, full_sort, slow_mode)
d = numel(z);
dl = max(1,2*k-2);
dr = d-dl;

size_orig = size(z);
z = reshape(z, [1,d]);

% Sort z
[I_sort, I_unsort] = find_sort_perm(z, dl, full_sort);
z = z(I_sort);

Il = I_sort(1:dl);
Ir = I_sort((dl+1):d);

if use_mex_code
    if slow_mode
        warning('<slow_mode> = true is only applicable with <use_mex_code> = false');
    end
    
    [mu, theta, dl, bz, bzl, bzr, delta, alpha, log_alpha] = gsm_v5_1_mex(z, k, gamma);
else
    [mu, theta, bz, bzl, bzr, delta, alpha, log_alpha] = calc_gsm_reduction(numcast, z, k, gamma, slow_mode);
end

% Unsort theta to match the original order of z
theta = theta(I_unsort);
theta = reshape(theta, size_orig);

% Return info
info = struct();
info.version = '5.1';
info.k_base = k;
info.gamma_base = gamma;
info.dl = dl;
info.dr = dr;
info.Il = Il;
info.Ir = Ir;
info.bz = bz;
info.bzl = bzl;
info.bzr = bzr;
info.delta = delta;
info.alpha = alpha;
info.log_alpha = log_alpha;
end


function [mu, theta, bz, bzl, bzr, delta, alpha, log_alpha] = calc_gsm_reduction(numcast, z, k, gamma, slow_mode)
% This is the main function that calculates mu and theta in the reduced case
% 1 <= k <= d/2 and 0 < gamma < inf.
base = 1;
dl = max(2*k-2,1);
d = numel(z);
dr = d-dl;

zl = z(1:dl);

%% Step 1
% Calculate mu_{k,g}(z); b_{q,gamma}(z), z_(q) for q=0,...,k; and b_{q,gamma}(zr),
% zr_{q} for q=0,...,min(k,dr)
[mu, bz, z_ord, bzr, zr_ord] = alg3(numcast, z,  k, gamma, k, dr); 

% Calculate b_{q,gamma}(zl), zl_(q) for q=0,...,dl
[~, bzl, zl_ord] =             alg3(numcast, zl, [], gamma, dl, []);

%% Step 2
[delta, alpha, log_alpha] = alg6(numcast, k, dl, dr, z_ord, zl_ord, zr_ord);

theta = numcast(nan(size(z)));

if slow_mode
    [~, bz2, z_ord2] = alg3(numcast, z, k, gamma, d, []);
    
    for i=d:-1:1
        if (i < d) && (z(i) == z(i+1))
            theta(i) = theta(i+1);
        elseif (i > dl) || (gamma*(z(i)-z_ord(k+base)) + bz(k-1+base) - bz(k+base) <= log((d-k+1)/k))
            theta(i) = alg4(numcast, z, k, gamma, bz, z_ord, i);
        else            
            theta_zl_i = alg5(numcast, z(i), d, k, gamma, bz2, z_ord2);
            theta(i) = theta_zl_i(k+base);
        end
    end
    
    return
end

%C = test_coeffs(numcast, k, gamma, bzl, bzr, delta, alpha, log_alpha, bz, dl, dr, z_ord, zl_ord, zr_ord);
%figure; imshow(exp(C/10),[]);
%max(abs(sum(exp(C),2)-1))

stability_thresh = z_ord(k+base) + ( (bz(k+base) - bz(k-1+base)) + log(numcast(d-k+1)/numcast(k)) ) / gamma;

for i=d:-1:1
    if (i < d) && (z(i) == z(i+1))
        theta(i) = theta(i+1);
    elseif (i > dl) || (z(i) <= stability_thresh)
        % Stability of Algorithm 4 is guaranteed for i > dl, and for any other i for
        % which z_i is below the stability threshold
        theta(i) = alg4(numcast, z, k, gamma, bz, z_ord, i);
    else
        % For the rest of the indices i, calculate theta_{k,gamma}^i(z) from
        % theta_{q,gamma}^i(zl) for q=0,...,k
        theta_zl_i = alg5(numcast, z(i), dl, k, gamma, bzl, zl_ord);
        theta(i) = alg7(numcast, dr, k, gamma, theta_zl_i, bz(k+base), bzl, bzr, delta, alpha, log_alpha);
    end
end
end


function [mukg, bz, z_ord, bzr, zr_ord] = alg3(numcast, z, k, gamma, s, dr)
% Calculates mu_{k,gamma}(z) and b_{q,gamma}(z) for q=0,...,s.
% Based on Algorithm 3 from [1]. 
% Input: z = [z_1,...,z_d]
%        1 <= k <= s <= d (Optional. Can be set to [].)
%        0 < gamma < inf
%        dr s.t. 1 <= dr <= d. (Optional. Can be set to 0 or []).
%
% Output: mu    = mu_{k,gamma}(z). If k is [], mu = [].
%         bz    = [b_{0,gamma}(z), ..., b_{s,gamma}(z)]  (see [1] for definition) 
%         z_ord = [z_(0), ..., z_(s)], where z_(i) is the i-th largest entry 
%             of z and z_(0) = inf.
%
% Optional output: (Not returned if dr = 0 or [])
%         bzr = [b_{0,gamma}(zr), ..., b_{s2,gamma}(zr)], where z = [zl, zr], zr is of
%           size dr, and s2 = min(s, dr).
%         zr_ord = [zr_(0), zr_(1), ..., zr_(s2)]
base = 1;

if isempty(dr) || (dr == 0)
    bzr = [];
    zr_ord = [];
end 

d = numel(z);

% NaNs here signify uninitialized values that should not be used.
b = numcast(nan(s+1,1));
btilde = numcast(nan(size(b)));

v = numcast(nan(s+1,1));
vtilde = numcast(nan(size(v)));

% Throughout the loop, b[0], btilde[0] should be 0 and v[0], vtilde[0] should be inf.

% Initialization. Used by iteration r=d
vtilde(0+base) = inf;

% Only required in the C++ code, where instead of updating btilde=b at the end of each 
% iteration, we simply switch the pointers of the two vectors.
%btilde(0+base) = 0; 

% Part of the calculation for iteration r=d
v(0+base) = inf;
b(0+base) = 0;

for r=d:-1:1
    % - Iteration r requires vtilde(q) and btilde(q) for q=0,...,min(d-r,s)
    %   and updates v(q) and b(q) for q=1,...,min(d-r+1,s). 
    % - At the end of iteration r, v[q] and b[q] are updated for q=0,...,s.
    
    % q=0 is handled implicitly, by keeping v[0]=vtilde[0]=inf and b[0]=btilde[0]=0.
    for q=1:nmin(s,d-r)
        v(q+base) = nmax(nmin(z(r), vtilde(q-1+base)), vtilde(q+base));
        xi = gamma*(z(r)-vtilde(q+base));
        eta = (btilde(q+base) - btilde(q-1+base)) - xi;
        
        if eta <= 0
            % Note that if z is fully sorted in nonincreasing order, eta is guaranteed to
            % be nonpositive, and subminus(xi) = 0.
            log1p_term = log1p(numcast((d-r-q+1)/(d-r+1))*expm1(eta));
            b(q+base) = (btilde(q-1+base) - subplus(-xi)) + log1p_term;
        else
            log1p_term = log1p(numcast(q/(d-r+1))*expm1(-eta));
            b(q+base) = (btilde(q+base) - subplus(xi)) + log1p_term;
        end
        
        % b[q] should be non-positive anyway, so clip any possible negativity caused
        % by numerical error.
        b(q+base) = nmin(b(q+base),numcast(0));        
    end
    
    % Handle q=d-r+1
    if s >= d-r+1
        v(d-r+1+base) = nmin(vtilde(d-r+base), z(r));
        b(d-r+1+base) = numcast(0);
    end
    
    if ~isempty(dr) && (r == d-dr+1)
        % Return z_{q}^{r_extra} for q=0,1,...,s and b_{k,gamma}^{r_extra}(z)
        s2 = nmin(s, dr);
        bzr = b((0+base) : (s2+base));
        zr_ord = v((0+base) : (s2+base));
    end
    
    btilde = b;
    vtilde = v;
end

bz = btilde;
z_ord = vtilde;

if ~isempty(k)
    mukg = btilde(k+base)/gamma + sum(vtilde((k+base):-1:(1+base)));
else
    mukg = [];
end
end



function theta_kg_i = alg4(numcast, z, k, gamma, bz, z_ord, i)
% Calculates theta_{k,gamma}^i(z) by forward recursion.
% Algorithm 4 in [1].
%
% Input: z = [z_1, ..., z_d], 1 <= k,i <= d, 0 < gamma < inf
%        b = [b_{0,gamma}(z), ..., b_{k,gamma}(z)]
%        z_ord = [z_(0), ..., z_(k)]
%
% Output: theta_{k,gamma}^i(z)
base = 1;
one = numcast(1);
d = numel(z);

xi = numcast(0);

for q=1:k
    factor_curr = numcast(q/(d-q+1)) * exp( gsm_fma(gamma, z(i)-z_ord(q+base), bz(q-1+base) - bz(q+base)));
    xi = nmin(one, factor_curr * (one-xi));
end

theta_kg_i = nmin(one, nmax(numcast(0), xi));
end


function theta_zl_i = alg5(numcast, zl_i, dl, k, gamma, bzl, zl_ord)
% Algorithm 5 in [1]
%
% Input: zl = [zl_1, ..., zl_dl], 1 <= i,k <= dl, 0 < gamma < inf
%        bzl = b_{q,gamma}(zl) for q=0,...,dl
%        zl_ord = zl_(q) for q=0,...,dl
%
% Output: theta_{q,gamma}^i(zl) for q=0,...,k

base = 1;
one = numcast(1);

theta_zl_i = nan(dl+1,1);

theta_zl_i(0+base) = 0;
theta_zl_i(dl+base) = 1;

theta_zl_i = numcast(theta_zl_i);

qhat = k+1;

for q=1:k
    eta = gamma*(zl_i-zl_ord(q+base)) + (bzl(q-1+base) - bzl(q+base));
    frac_term = numcast(q/(dl-q+1));
    
    if eta <= -log(frac_term)
        theta_zl_i(q+base) = nmin(one, frac_term * exp(eta)) * (one - theta_zl_i(q-1+base));
    else
        qhat = q;
        break
    end    
end

if qhat <= k
    for q = (dl-1):-1:qhat
        eta = (bzl(q+1+base) - bzl(q+base)) - gamma*(zl_i-zl_ord(q+1+base));
        frac_term = numcast((dl-q)/(q+1));
        theta_zl_i(q+base) = max(numcast(0), gsm_fma(-frac_term*exp(eta), theta_zl_i(q+1+base), one) );
    end
end

theta_zl_i = theta_zl_i((0+base) : (k+base));
end


function [delta, alpha, log_alpha] = alg6(numcast, k, dl, dr, z_ord, zl_ord, zr_ord)
% Calculates the entries Delta_{k,t}(zl,zr) and the coefficients alpha_{k,t}^{dl,dr} for
% t=0,...,k. Based on Algorithm 6 in [1].
% Input:
%   z = [zl, zr] in R^d where zl in R^dl, zr in R^dr and dl,dr >= 1
%   (z, zl and zr are not required explicitly)
%   1 <= k <= d
%   z_ord = [z_(0), ..., z_(k)]
%   zl_ord = [zl_(0), ..., zl_(k)]
%   zr_ord = [zr_(0), ..., zr_(min(k,dr))]
%
% Output:
%    delta:     Delta_{k,t}(zl,zr) for t=0,...,k
%    alpha:     alpha_{k,t}^{dl,dr} for t=0,...,k
%    log_alpha: log(alpha_{k,t}^{dl,dr}) for t=0,...,k
%
% alpha are returned as normal nonzero floating-point numbers, or zero
% (i.e. alpha does not contain nonzero subnormal values.)
base = 1;

if isa(numcast(1),'double')
    gsm_realmin = realmin('double');
else
    gsm_realmin = realmin('single');
end

d = dl+dr;

% Initialize for q=0
delta = nan(1, k+1);
delta(0+base) = 0;
delta = numcast(delta);

alpha = nan(1, k+1);
alpha(0+base) = 1;
alpha = numcast(alpha);

% t_mode is the t for which alpha^{dl,dr}_{q,t} is maximal
t_mode = 0;

for q=1:k
    % Range of t values where alpha_{q,t} and delta_{q,t} are not identically zero
    ta = nmax(0,q-dr);
    tb = nmin(q,nmin(k,dl));
    
    % Calculate alpha{q, t} at t=t_mode
    t_mode_prev = t_mode;
    t_mode = numcast( idivide( uint64(dl+1)*uint64(q+1), uint64(d+2) ) );

    % t_mode always equals t_mode_prev or t_mode_prev+1.  
    % Calculate alpha_{q, t_mode} from alpha_{q-1, t_mode_prev}.
    if t_mode == t_mode_prev
        fac = numcast(((dr+t_mode-q+1)*q)/((d-q+1)*(q-t_mode)));
        alpha(t_mode+base) = fac * alpha(t_mode_prev+base);
    elseif t_mode == t_mode_prev+1
        fac = numcast(((dl-t_mode+1)*q)/((d-q+1)*t_mode));
        alpha(t_mode+base) = fac * alpha(t_mode_prev+base);
    else
        error('This should not happen');
    end

    % Calculate delta{q,t} for t=ta,...,tb    
    for t=tb:-1:nmax(ta,1)        
        %alpha(t+base) = numcast((dr-q+t+1)/(d-q+1))*alpha(t+base) + numcast((dl-t+1)/(d-q+1))*alpha(t-1+base);

        if z_ord(q+base) >= zl_ord(t+base)
            delta(t+base) = delta(t-1+base) + (z_ord(q+base)-zl_ord(t+base));
        else
            delta(t+base) = delta(t+base) + (z_ord(q+base)-zr_ord(q-t+base));
        end       
    end
    
    if ta == 0
        delta(0+base) = delta(0+base) + (z_ord(q+base)-zr_ord(q+base));
    else
        delta(ta-1+base) = numcast(0);
    end
end

log_alpha = nan(1,k+1);
log_alpha = numcast(log_alpha);

%% Propagate knowledge of alpha_{q,t} and log_alpha_{q,t} from t_mode to the other t's
% Note that alpha_{q,t_mode} is a normal positive floating point number, since it is 
% greater or equal to 1/dl (since sum(alphas) = 1).

t_mode = numcast( idivide( uint64(dl+1)*uint64(k+1), uint64(d+2) ) );
log_alpha(t_mode+base) = log(alpha(t_mode+base));

% Range of t values
ta = max(0, k-dr);
tb = min(k, dl);

t = 0;
while t <= k
    if t == ta
        t = tb+1;
        continue;
    end
    
    delta(t+base) = numcast(0);
    alpha(t+base) = numcast(0);
    log_alpha(t+base) = numcast(-inf);
    
    t = t+1;
end

for t = (t_mode-1):-1:ta    
    fac = numcast(((t+1)*(dr-(k-t)+1))/((dl-t)*(k-t)));
    alpha(t+base) = fac * alpha(t+1+base);
    
    if (alpha(t+base) >= gsm_realmin)
        % If alpha(t+base) is a normal floating point number, calculate its log directly.
        log_alpha(t+base) = log(alpha(t+base));
    elseif fac <= 0.5
        % ...otherwise, calculate the log recursively.
        log_alpha(t+base) = log_alpha(t+1+base) + log(fac);
        
        % For safety, we do not allow subnormal numbers in the alphas. Instead, we use 
        % zeros.
        alpha(t+base) = 0;
    else
        % If the multiplied factor is closer to 1 than to 0, use log1p
        facm1 = numcast(((dr+1)*(t+1) - (dl+1)*(k-t)) / ((dl-t)*(k-t)));
        log_alpha(t+base) = log_alpha(t+1+base) + log1p(facm1);
        alpha(t+base) = 0;
    end
end

for t = (t_mode+1):tb     
    fac = numcast(((dl-t+1)*(k-t+1))/(t*(dr-k+t)));
    alpha(t+base) = fac * alpha(t-1+base);

    if (alpha(t+base) >= gsm_realmin)
        log_alpha(t+base) = log(alpha(t+base));
    elseif fac <= 0.5
        log_alpha(t+base) = log_alpha(t-1+base) + log(fac);
        alpha(t+base) = 0;
    else
        facm1 = numcast(((dl+1)*(k-t+1) - (dr+1)*t) / (t*(dr-k+t)));
        log_alpha(t+base) = log_alpha(t-1+base) + log1p( facm1 );
        alpha(t+base) = 0;
    end
end

end


function thetakgi = alg7(numcast, dr, k, gamma, theta_zl_i, bkg_z, bzl, bzr, delta, alpha, log_alpha)
base = 1;

qa = nmax(1, k-dr);
qb = k;

xi = numcast(0);

for q=qb:-1:qa
    alpha_curr = alpha(q+base);
    arg_curr = ((bzl(q+base) + bzr(k-q+base)) - bkg_z) - gamma*delta(q+base);
    exp_curr = exp(arg_curr);
        
    if isa(numcast(1),'double') 
        realmin_curr = realmin;
        realmax_curr = realmax;
    else
        realmin_curr = realmin('single');
        realmax_curr = realmax('single');
    end
    
    % We need to avoid 0*inf (or any <subnormal number> * inf). Any other case, including
    % underflows, is ok.
    %
    % In the following two cases, 0*inf is guaranteed not to take place:
    % (recall that alpha_curr is always <= 1)
    %
    % 1. arg_curr <= 0. Then alpha_curr * exp(arg_curr) is surely ok.
    % 2. Both alpha_curr and exp_curr are normal floating point numbers. Since alpha_curr 
    %    <= 1, the product alpha_curr * exp_curr is fine (even if underflows).
    %
    % Note that the way we calculated alpha in Alg. 6 guarantees that it is either normal
    % or zero, since we flushed subnormal values to zero.
    if (arg_curr <= 0) || ((alpha_curr > numcast(0)) && (exp_curr < realmax_curr))
        coeff_curr = nmin(numcast(1), alpha_curr * exp_curr);
    else
        coeff_curr = exp(nmin(numcast(0), log_alpha(q+base) + arg_curr));
    end
    
    xi = gsm_fma(coeff_curr, theta_zl_i(q+base), xi);    
end

thetakgi = nmax(numcast(0),min(numcast(1),xi));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                   Utilities                                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [I_sort, I_unsort] = find_sort_perm(z,dl,full_sort)
d = numel(z);

if full_sort
    [~, I_sort] = sort(z,'descend');
else
    bval = min(maxk(z,dl));
    I1 = find(z > bval);
    I2 = find(z == bval);
    I3 = find(z < bval);
    I_sort = [I1, I2, I3];
end

I_unsort = nan(size(I_sort));
I_unsort(I_sort) = 1:d;
end


% Safe min function
function out = nmin(x,y)
if isnan(x) || isnan(y)
    out = cast(nan, class(x));
else
    out = min(x,y);
end
end

% Safe max function
function out = nmax(x,y)
if isnan(x) || isnan(y)
    out = cast(nan, class(x));
else
    out = max(x,y);
end
end


% Fused multiply-add. This may speed calculation and increase precision. It is not used
% in the C code because it caused a slowdown for some unknown reason, so it is also 
% disabled here to keep the C and Matlab codes equivalent.
function out = gsm_fma(x,y,a)
out = x*y + a;
%out = fma(x,y,a);
end



