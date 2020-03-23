// calcgsm_mex.cpp
#include "matrix.h"
#include <stdlib.h>
#include <mex.h>
#include <math.h>

#define double_precision true

#define max(x,y) ((x) >= (y) ? (x) : (y))
#define min(x,y) ((x) <= (y) ? (x) : (y))
#define subplus(x) (max((x),0))
#define subminus(x) (max(-(x),0))
#define trim(x) (max((num_type)0,min((num_type)1,(x))))
#define malloc_cmd(s) (mxMalloc(s))
#define memfree_cmd(p) (mxFree(p))

#define nan (mxGetNaN())
#define inf (mxGetInf())

typedef mwIndex index_type;
typedef mwSize  size_type;

#if double_precision
typedef double num_type;
#else
typedef float num_type;
#endif

typedef struct
{
    mwSize k;
    mwSize n;
    num_type gamma;
    
    const num_type* z;
    num_type* mu;
    num_type* theta;
} argStruct;

argStruct readArguments(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

void calc_gsm_reduced_case(const num_type* z, size_type n, size_type k, num_type gamma, num_type* mu, num_type* theta);

void calc_b(const num_type* z, size_type n, size_type s, size_type k, num_type gamma, num_type* mu, num_type* b, num_type* z_ord, num_type* bright, num_type* zright_ord);
void calc_theta(const num_type* z, index_type n, index_type s, index_type k, num_type gamma, const num_type *b, const num_type *z_ord, const num_type *bright, const num_type *zright_ord, const num_type* bleft, const num_type* zleft_ord, num_type* theta);
double calc_theta_i_direct(const num_type z_i, size_type n, index_type k, num_type gamma, const num_type *b, const num_type *z_ord);
inline void calc_theta_tilde_i(const num_type z_i, size_type s, num_type gamma, const num_type *bleft, const num_type *zleft_ord, num_type* theta_tilde_i);
num_type convert_to_theta_i(const num_type* theta_tilde_i, size_type nleft, index_type nright, index_type k, num_type gamma, const num_type* b, const num_type *bleft, const num_type *zleft_ord, const num_type* bright, const num_type *zright_ord, const num_type* log_alpha, const num_type* delta);

void init_alpha(num_type* alpha, num_type* log_alpha, index_type s);
void update_alpha(num_type* alpha, num_type* log_alpha, index_type s, size_type n1, size_type n2, index_type k, index_type q);
void init_delta(num_type* delta, index_type s);
void update_delta(num_type* delta, index_type s, size_type nleft, size_type nright, index_type q, const num_type* zleft_ord, const num_type* zright_ord, const num_type* zhat_ord);

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    argStruct args = readArguments(nlhs, plhs, nrhs, prhs);   
    calc_gsm_reduced_case(args.z, args.n, args.k, args.gamma, args.mu, args.theta);    
}


void calc_gsm_reduced_case(const num_type* z, size_type n, size_type k, num_type gamma, num_type* mu, num_type* theta)
{
    size_type khat = max(2*k-2, k);

    num_type* b      = (num_type*) malloc_cmd(khat*sizeof(num_type));
    num_type* bright = (num_type*) malloc_cmd(khat*sizeof(num_type));
    num_type* bleft  = (num_type*) malloc_cmd(khat*sizeof(num_type));
    
    num_type* z_ord      = (num_type*) malloc_cmd((khat+1)*sizeof(num_type));
    num_type* zright_ord = (num_type*) malloc_cmd((khat+1)*sizeof(num_type));
   
    calc_b(z, n, khat, k, gamma, mu, b, z_ord, bright, zright_ord);
    calc_b(z, khat, khat, 0, gamma, NULL, bleft, NULL, NULL, NULL);
    calc_theta(z, n, khat, k, gamma, b, z_ord, bright, zright_ord, bleft, z_ord, theta);
       
    memfree_cmd(b);
    memfree_cmd(bright);
    memfree_cmd(bleft);

    memfree_cmd(z_ord);
    memfree_cmd(zright_ord);
}


// This is the main function
void calc_b(const num_type* z, size_type n, size_type s, size_type k, num_type gamma, num_type* mu, num_type* b, num_type* z_ord, num_type* bright, num_type* zright_ord)
{
    num_type* b1 = (num_type*) malloc_cmd((s+1)*sizeof(num_type));
    num_type* b2 = (num_type*) malloc_cmd((s+1)*sizeof(num_type));
    num_type* v1 = (num_type*) malloc_cmd((s+1)*sizeof(num_type));
    num_type* v2 = (num_type*) malloc_cmd((s+1)*sizeof(num_type));
    
    num_type *br, *vr, *brp1, *vrp1;
    
    int vecSwitch = 1;
    
    // Update pointers
    br   = (vecSwitch == 1 ? b1 : b2);
    brp1 = (vecSwitch == 1 ? b2 : b1);
    vr   = (vecSwitch == 1 ? v1 : v2);
    vrp1 = (vecSwitch == 1 ? v2 : v1);

    
    // Initialize partially (for some of the entries) brp1 = br^{d+1}, vrp1 = vr^{d+1} 
    brp1[0] = 0;
    vrp1[0] = inf;
    vrp1[1] = -inf;        

    // Initialize partially (for some of the entries) br = b^d, vr = v^d 
    for (index_type q=0; q <= s; ++q)
    {
        br[q] = 0;
        vr[q] = -inf;
    }
    
    vr[0] = inf;

    // Offset for treating 1-based vector z
    const index_type ofs = 1;
    
    vecSwitch = -vecSwitch;    
    
    // Main loop
    for (index_type r=n; r >= 1; --r)
    {        
        // Update pointers
        vecSwitch = -vecSwitch;
        
        br      = (vecSwitch == 1 ? b1 : b2);
        brp1    = (vecSwitch == 1 ? b2 : b1);
        vr      = (vecSwitch == 1 ? v1 : v2);
        vrp1    = (vecSwitch == 1 ? v2 : v1);   

        // q=0 is handled implicitly
        for (index_type q = 1; q <= min(s, n-r); ++q)
        {
            vr[q] = max(min(z[r-ofs], vrp1[q-1]), vrp1[q]);                     
            num_type xi = gamma*(z[r-ofs]-vrp1[q]);                
            num_type eta = (brp1[q-1] - brp1[q]) + xi;
            
            if (eta <= log((num_type)(n-r-q+1) / (num_type)q))
            {
                br[q] = brp1[q]   -  subplus(xi) + log1p((num_type)q/(num_type)(n-r+1)*expm1(eta));
            }
            else
            {
                br[q] = brp1[q-1] - subminus(xi) + log1p((num_type)(n-r+1-q)/(num_type)(n-r+1)*expm1(-eta));
            }

        }
                    
        // q = n-r+1
        if (s >= n-r+1)
        {
            vr[n-r+1] = min(vrp1[n-r], z[r-ofs]);
            br[n-r+1] = 0;
        }
            
        if (r == s+1)
        {            
            if (bright != NULL)
            {
                // Return mu_{k,gamma}^{s+1}(z)
                for (index_type q = 0; q <= s; ++q)
                {
                    bright[q] = br[q];
                }
            }

            if (zright_ord != NULL)
            {
                // Return z_{t}^{s+1} for q=0,1,...,s
                for (index_type q = 0; q <= s; ++q)
                {
                    zright_ord[q] = vr[q];
                }
            }
        }        
    }

    // Return b_{k,gamma}(z)
    for (index_type q=0; q <= s; ++q)
    {
        b[q] = br[q];
    }

    // Return z_(q) for q=0,1,...,s
    if (z_ord != NULL)
    {
        for (index_type q = 0; q <= s; ++q)
        {
            z_ord[q] = vr[q];
        }
    }
    
    // Return mu_{k,gamma}(z)
    if (mu != NULL)
    {
        *mu = 0;
        for (index_type q=k; q >= 1; --q)
        {
            *mu += vr[q];
        }
        
        *mu += b[k]/gamma;
    }    

    // Release temporary variables
    memfree_cmd(b1);
    memfree_cmd(b2);
    memfree_cmd(v1);
    memfree_cmd(v2);    
}


void calc_theta(const num_type* z, index_type n, index_type s, index_type k, num_type gamma, const num_type *b, const num_type *z_ord, const num_type *bright, const num_type *zright_ord, const num_type* bleft, const num_type* zleft_ord, num_type* theta)
{
    const index_type ofs = 1;
    
    num_type* theta_tilde_i = (num_type*) malloc_cmd(s*sizeof(num_type));
    
    for (index_type i=n; i >= s+1; --i)
    {
        if (z[i-ofs] == z[i+1-ofs])
        {
            theta[i-ofs] = theta[i+1-ofs];
        }
        else
        {
            theta[i-ofs] = calc_theta_i_direct(z[i-ofs], n, k, gamma, b, z_ord);
        }
    }

    
    // Calculate alpha_k and delta_k   
    const size_type nleft  = s;
    const size_type nright = n-s;

    num_type* alpha = (num_type*) malloc_cmd((k+1)*sizeof(num_type));
    num_type* log_alpha = (num_type*) malloc_cmd((k+1)*sizeof(num_type));
    num_type* delta = (num_type*) malloc_cmd((k+1)*sizeof(num_type));
    
    init_alpha(alpha, log_alpha, k);
    init_delta(delta, k);
    
    for (index_type q=1; q <= k; ++q)
    {
        update_alpha(alpha, log_alpha, k, nleft, nright, k, q);
        update_delta(delta, k, nleft, nright, q, zleft_ord, zright_ord, zleft_ord);
    }
    
    // Complete theta
    for (index_type i=s; i >= 1; --i)
    {
        if (z[i-ofs] == z[i+1-ofs])
        {
            theta[i-ofs] = theta[i+1-ofs];
        }
        else
        {
            calc_theta_tilde_i(z[i-ofs], s, gamma, bleft, zleft_ord, theta_tilde_i);
            theta[i-ofs] = convert_to_theta_i(theta_tilde_i, nleft, nright, k, gamma, b, bleft, zleft_ord, bright, zright_ord, log_alpha, delta);
        }
    }

    memfree_cmd(alpha);
    memfree_cmd(delta);
    
    memfree_cmd(theta_tilde_i);
}


inline double calc_theta_i_direct(const num_type z_i, size_type n, index_type k, num_type gamma, const num_type *b, const num_type *z_ord)
{        
    num_type result = exp(gamma*(z_i-z_ord[1]) - b[1]) / (num_type)n;
    result = trim(result);
    
    for (index_type q=2; q <=k; ++q)
    {
        result = (((num_type)q) / ((num_type)(n-q+1))) * exp(gamma*(z_i-z_ord[q]) + (b[q-1] - b[q])) * (1-result);
        result = trim(result);
    }
    
    return result;
}
     

inline void calc_theta_tilde_i(const num_type z_i, size_type s, num_type gamma, const num_type *bleft, const num_type *zleft_ord, num_type* theta_tilde_i)
{
    const index_type ofs = 1;
    
    theta_tilde_i[1-ofs] = trim( exp(gamma*(z_i - zleft_ord[1]) - bleft[1]) / (num_type)s );
    
    index_type q_hat = s+1;

    for (index_type q=2; q <= s; ++q)
    {
        num_type eta = gamma*(z_i - zleft_ord[q]) + (bleft[q-1] - bleft[q]);    
        num_type frac_term = ((num_type)q)/((num_type)(s-q+1));
        
        if (eta <= -log(frac_term))
        {
            theta_tilde_i[q-ofs] = trim( (frac_term * exp(eta)) * (((num_type)1)-theta_tilde_i[q-1-ofs]) );
        }
        else
        {
            q_hat = q;
            break;
        }
    }
    
    theta_tilde_i[s-ofs] = 1;
       
    for (index_type q = s-1; q >= q_hat; --q)
    {
        num_type eta = gamma*(z_i - zleft_ord[q+1]) + (bleft[q] - bleft[q+1]);    
        num_type frac_term = ((num_type)(s-q))/((num_type)(q+1));

        theta_tilde_i[q-ofs] = trim( ((num_type)1) - (frac_term*exp(-eta))*(theta_tilde_i[q+1-ofs]) );
    }    
}


num_type convert_to_theta_i(const num_type* theta_tilde_i, size_type nleft, index_type nright, index_type k, num_type gamma, const num_type* b, const num_type *bleft, const num_type *zleft_ord, const num_type* bright, const num_type *zright_ord, const num_type* log_alpha, const num_type* delta)
{
    const index_type ofs = 1;
    
    const index_type ta = max(nright+1, k) - nright; //max(1, k-nright);
    const index_type tb = min(k, nleft);

    num_type result = 0;
    
    for (index_type t=tb; t >= ta; --t)
    {
        result +=
            theta_tilde_i[t-ofs] *
            exp(log_alpha[t] -gamma*delta[t] + bleft[t] + bright[k-t] - b[k]);
    }
    
    return result;
}
        
        
        
        
// Read and verify input arguments
argStruct readArguments(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Number of input arguments
    if (nrhs != 3)
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:nrhs",
               "Three input arguments are required");
    }

    // z
    if (!mxIsClass(prhs[0], "double") || mxIsSparse(prhs[1]) || mxIsComplex(prhs[0]) || (mxGetNumberOfElements(prhs[0]) < 1))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:k",
               "z should be a nonempty matrix of class double");
    }
            
    const size_type n = mxGetNumberOfElements(prhs[0]);
    const num_type* z = mxGetPr(prhs[0]);
    
    const index_type ofs = 1;
    
    // k
    if ((!mxIsDouble(prhs[1])) || mxIsComplex(prhs[1]) || (mxGetNumberOfElements(prhs[1]) != 1))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:k",
               "k should be a double scalar");
    }

    double k_double = *mxGetPr(prhs[1]);

    if (mxIsNaN(k_double) || mxIsInf(k_double))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:nan",
                "k should be a finite number");
    }
    
    index_type k = (index_type)k_double;
             
    if ((double)k != k_double)
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:k",
               "k must be an integer (represented as num_type)");
    }
    
    if ((k > n-1) || (k < 1))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:k",
               "k must be between 1 and numel(z) - 1");
    }
    
    const index_type khat = max(2*k-2,k);
    
    for (index_type i=1; i<=n; ++i)
    {
        if (mxIsNaN(z[i-ofs]))
        {
            mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:nan",
                   "z should not contain NaNs");
        }

        if (mxIsInf(z[i-ofs]))
        {
            mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:inf",
                   "z should not contain infs");
        }

        if (( (i < khat) && (z[i-ofs] < z[khat-ofs]) ) || ( (i > khat) && (z[i-ofs] > z[khat-ofs]) ))
        {
            mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:inf",
                   "z must be sorted such that z[i] >= z[khat] >= z[j] for all i <= khat <= j, where khat = max(2*k-2,1)");
        }
    }

    // gamma
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || (mxGetNumberOfElements(prhs[2]) != 1))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:gamma",
               "gamma should be a scalar");
    }
    
    const num_type gamma = mxGetScalar(prhs[2]);
    
    if (mxIsNaN(gamma))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:nan",
                "gamma should not contain NaNs");
    }
    
    if ((gamma < 0) || mxIsInf(gamma))
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:gamma",
               "gamma should be nonnegative and finite");
    }

    
    // Verify output arguments
    
    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt("SoftMinTransform:calcLogSt_mex:nlhs",
               "Two output arguments are required");
    }
    
    // Initialize outputs
    
    // mu output argument
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);   
    num_type * mu = mxGetPr(plhs[0]);

    // theta output argument
    plhs[1] = mxCreateDoubleMatrix(1, n, mxREAL);
    num_type * theta = mxGetPr(plhs[1]);
    
    // Return output
    argStruct args;
    args.z = z;
    args.k = k;
    args.n = n;
    args.gamma = gamma;
    args.theta = theta;
    args.mu = mu;
    
    return args;
}




//function out = logmeanexp(alpha, log_alpha, beta)
//
// Calculates log(sum(alpha .* exp(beta))) in a numerically safe way, where
// alpha = [alpha_1, ..., alpha_s]
// beta  = [beta_1, ..., beta_s]
// and log_alpha = log(alpha)
// such that all(alpha > 0) and sum(alpha) = 1.
num_type logmeanexp(const num_type* alpha, const num_type* beta, size_type s)
{
    num_type beta_max = -inf;
    
    for (index_type t=0; t<s; ++t)
    {
        if (beta[t] > beta_max)
            beta_max = beta[t];
    }

    num_type S = 0;
    
    for (index_type t=0; t<s; ++t)
    {
        S += alpha[t] * expm1(beta[t]-beta_max);
    }
    
    if (S >= -0.5)
    {
        // Use first method of calculation
        return log1p(S) + beta_max;
    }
    else
    {
        // If preferable, use second method of calculation

        S = 0;
        
        for (index_type t=0; t<s; ++t)
        {
            S += alpha[t] * exp(beta[t]-beta_max);
        }
        
        return log(S) + beta_max;
        
    }
}


//function [alpha, log_alpha] = init_alpha(s)
//
// Calculates the binomial coefficients alpha^{n1,n2}_{q,t} for t=0,1,...,s
// and their logarithms, where q=0 and 1 <= s <= n1+n2
inline void init_alpha(num_type* alpha, num_type* log_alpha, index_type s)
{
    alpha[0] = 1;
    log_alpha[0] = 0;
    
    for (index_type t=1; t <= s; ++t)
    {
        alpha[t] = 0;
        log_alpha[t] = -inf;
    }    
}


//function [alpha, log_alpha] = update_alpha(n1, n2, q, alpha, log_alpha)
//
// Calculates the binomial coefficients alpha^{n1,n2}_{q,t} for t=0,1,...,s
// and their logarithms, where 1 <= q <= s <= n1+n2
//
// Takes as input alpha^{n1,n2}_{q-1,t} for t=0,1,...,s, and the respective
// logarithms.
//
// If the logarithms are not required, log_alpha can be left empty in order
// to save time.

inline void update_alpha(num_type* alpha, num_type* log_alpha, index_type s, size_type n1, size_type n2, index_type k, index_type q)
{
    //assert(1 <= q);
    //assert(q <= s);
    //assert(s <= n1+n2);
    
    // Range of t values
    index_type ta = max(n2,q) - n2; // A safe way to denote max(0, q-n2) with unsigned integers
    index_type tb = min(q, n1);
    
    // Calculate alpha recursively, marching backwards from t = tb
    for (index_type t = tb; t >= max(ta,1); --t)
    {
        if (alpha[t] >= alpha[t-1])
        {
            alpha[t] += ( alpha[t-1] + ((num_type)(n1-t))*(alpha[t-1]-alpha[t]) ) / ((num_type)(n1+n2-q+1));
        }
        else
        {
            alpha[t] = alpha[t-1] + ( alpha[t] + ((num_type)(n2-q+t))*(alpha[t]-alpha[t-1]) ) / ((num_type)(n1+n2-q+1));
        }
    }
    
    
    if (ta == 0)
    {
        alpha[0] *= ((num_type)(n2-q+1))/((num_type)(n1+n2-q+1));
    }
    else
    {
        alpha[ta-1] = 0;
    }
    
    
    if (tb < s)
    {
        alpha[tb+1] = 0;
    }
    
    // Calculate log_alpha
    #if double_precision
        const num_type log_thresh = -709;
    #else
        const num_type log_thresh = -88;
    #endif
            
    // Index t of the mode of alpha_{q,0},...,alpha_{q,k}
    index_type t_mode = ((n1+1)*(q+1))/(n1+n2+2);

    log_alpha[t_mode] = log(alpha[t_mode]);

    // Calculate log_alpha recursively, marching backwards from modal t
    for (index_type t = t_mode-1; t >= ta; --t)
    {
        num_type fac = (((num_type)t+(num_type)1)*((num_type)n2-((num_type)q-(num_type)t)+(num_type)1))/(((num_type)n1-(num_type)t)*((num_type)q-(num_type)t));
    
        if ((alpha[t] > 0) && (log_alpha[t] >= log_thresh))
        {
            // If there is no risk of underflow, calculate the log directly
            log_alpha[t] = log(alpha[t]);
        }
        else if (fac <= 0.5)
        {
            // Otherwise, calculate the log recursively
            log_alpha[t] = log_alpha[t+1] + log(fac);
        }
        else
        {
            // If the multiplied factor is closer to 1 than to 0, use log1p
            log_alpha[t] = log_alpha[t+1] + 
                log1p( ( ((num_type)n2+(num_type)1)*((num_type)t+(num_type)1) - ((num_type)n1+(num_type)1)*((num_type)q-(num_type)t) ) / ( ((num_type)n1-(num_type)t)*((num_type)q-(num_type)t) ));
        }
    }

    if (ta > 0)
    {
        log_alpha[ta-1] = -inf;
    }

    // Calculate log_alpha recursively, marching forward from modal t
    for (index_type t = t_mode+1; t <= tb; ++t)
    {
        num_type fac = ((num_type)(n1-t+1)*(num_type)(q-t+1))/((num_type)t*(num_type)(n2-q+t));
        // assert(fac <= 1);
    
        if ((alpha[t] > 0) && (log_alpha[t] >= log_thresh))
        {
            // If there is no risk of underflow, calculate the log directly
            log_alpha[t] = log(alpha[t]);
        }
        else if (fac <= 0.5)
        {
            // Otherwise, calculate the log recursively
            log_alpha[t] = log_alpha[t-1] + log(fac);
        }
        else
        {
            // If the multiplied factor is closer to 1 than to 0, use log1p
            log_alpha[t] = log_alpha[t-1] + 
                log1p( ((num_type)(n1+1)*(num_type)(q-t+1) - (num_type)t*(num_type)(n2+1)) / ((num_type)t*(num_type)(n2-q+t)) );
        }
    }

    if (tb < k)
    {
        log_alpha[tb+1] = -inf;
    }
}



//function delta = init_delta(s)
//
// Calculates delta_{q,t}(zhat) for t=0,1,...,s
// where zhat = [z1, z2], q=0 and s >= 1
inline void init_delta(num_type* delta, index_type s)
{
    for (index_type t=0; t <= s; ++t)
    {
        delta[t] = 0;
    }
}


//function delta = update_delta(delta, nleft, nright, q, zleft_ord, zright_ord, zhat_ord)
//
// Calculates delta_{q,t}(zhat) for t=0,1,...,s
// where zhat = [z1, z2], and 1 <= q <= s
//
// Takes as input delta_{q-1,t}(zhat) for t=0,1,...,s
inline void update_delta(num_type* delta, index_type s, size_type nleft, size_type nright, index_type q, const num_type* zleft_ord, const num_type* zright_ord, const num_type* zhat_ord)
{
    const index_type ta = max(nright, q) - nright; // A safe way to denote max(0, q-nright) with unsigned integers
    const index_type tb = min(q, nleft);

    for (index_type t=tb; t >= max(ta,1); --t)
    {
        if (zhat_ord[q] >= zleft_ord[t])
        {
            delta[t] = delta[t-1] + (zhat_ord[q] - zleft_ord[t]);
        }
        else
        {
            delta[t] += zhat_ord[q] - zright_ord[q-t];
        }
    }

    if (ta == 0)
    {
        delta[0] += zhat_ord[q] - zright_ord[q];
    }
    else
    {
        delta[ta-1] = 0;
    }
}

        