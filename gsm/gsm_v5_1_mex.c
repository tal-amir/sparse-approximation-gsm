// gsm_v5_1_mex.cpp
#define gsm_mex_filename "gsm_v5_1_mex"

// Generalized Soft-Min
// Tal Amir, Ronen Basri, Boaz Nadler
// Weizmann Institute of Science
//
// Version 5.1, June 2020
//
// Contact: tal.amir@weizmann.ac.il
//
// This file is part of the matlab GSM software, and is only meant to be called
// from the main matlab file. To build, call make.m.
//
// This c mex program calculates the Generalized Soft-Min function mu_{k,gamma}(z) and
// its gradient, theta_{k,gamma}(z), in the special case 1 <= k <= d/2, 0 < gamma < inf.
//
// Based on the paper [1]. All notations below are defined in the paper.
//
// [1] Tal Amir, Ronen Basri, Boas Nadler - The Trimmed Lasso: Sparse
//     recovery guarantees and practical optimization by the Generalized
//     Soft-Min Penalty.


// This sets the numeric type used in all intermediate calculations. 
// 1: float (single precision, 32bit)
// 2: double (double precision 64bit)
// 3: long double (On Windows systems, extended double precision, 80bit)
// 4: quadruple precision 128bit, using the <quadmath.h> c library. Yields extremely 
//    accurate results, but incurs up to 40-fold slowdown.
//
// This type is independent of the input and output type of the 
// matlab interface, since data conversions are used.
#ifndef gsm_intermediate_numerical_type
#define gsm_intermediate_numerical_type 2
#endif

// Determines what numerical class to use for input and output of the mex program.
// If set to true, the program expects all input (and returns all output) in class 'single'.
// Otherwise, 'double' is used for input and output.
// This has no effect on the intermediate numerical type used for calculations.
#ifndef gsm_single_precision_matlab_args
#define gsm_single_precision_matlab_args false
#endif

// Set to true to use the dedicated fused multiply-add function to calculate expressions
// of the form x*y + a. 
// For some unknown reason, it causes a slowdown on some architectures when 
// compiled with MinGW.
#define gsm_use_fma false

// Used only for debugging purposes. If set to true, all calculations of log / exp / 
// log1p / expm1 are done by matlab functions (in double precision).  
#define gsm_use_matlab_arithmetic false


//#include "matrix.h"
#include <mex.h>

#if gsm_intermediate_numerical_type == 4
#include <quadmath.h>
#else
#include <math.h>
#include <float.h>
#endif

// gsm_num is the numerical type to use for intermediate calculations.
// The definitions below are accompanying constants and functions suited for that type.
//
// Note: It is possible to add more options to the list, e.g. using faster or more
//       accurate mathematical libraries than <math.h>. 
//       All operations on mat_num variables, other than +-*/, are done using these
//       definitions.
#if gsm_intermediate_numerical_type == 1
// Use float for intermediate calculations
typedef float gsm_num;
#define gsm_exp_wrapper(x) expf((x))
#define gsm_log_wrapper(x) logf((x))
#define gsm_expm1_wrapper(x) expm1f((x))
#define gsm_log1p_wrapper(x) log1pf((x))
#define gsm_fma_wrapper(x,y,a) fmaf((x),(y),(a))
#define gsm_max(x,y) fmaxf((x),(y))
#define gsm_min(x,y) fminf((x),(y))
const gsm_num gsm_nan = NAN;
const gsm_num gsm_inf = HUGE_VALF;
const gsm_num gsm_realmin = FLT_MIN;
const gsm_num gsm_realmax = FLT_MAX;
const gsm_num gsm_realeps = FLT_EPSILON;

// Use double for intermediate calculations
#elif gsm_intermediate_numerical_type == 2
typedef double gsm_num;
#define gsm_exp_wrapper(x) exp((x))
#define gsm_log_wrapper(x) log((x))
#define gsm_expm1_wrapper(x) expm1((x))
#define gsm_log1p_wrapper(x) log1p((x))
#define gsm_fma_wrapper(x,y,a) fma((x),(y),(a))
#define gsm_max(x,y) fmax((x),(y))
#define gsm_min(x,y) fmin((x),(y))
const gsm_num gsm_nan = NAN;
const gsm_num gsm_inf = HUGE_VAL;
const gsm_num gsm_realmin = DBL_MIN;
const gsm_num gsm_realmax = DBL_MAX;
const gsm_num gsm_realeps = DBL_EPSILON;

// Use long double for intermediate calculations
#elif gsm_intermediate_numerical_type == 3
typedef long double gsm_num;
#define gsm_exp_wrapper(x) expl((x))
#define gsm_log_wrapper(x) logl((x))
#define gsm_expm1_wrapper(x) expm1l((x))
#define gsm_log1p_wrapper(x) log1pl((x))
#define gsm_fma_wrapper(x,y,a) fmal((x),(y),(a))
#define gsm_max(x,y) fmaxl((x),(y))
#define gsm_min(x,y) fminl((x),(y))
const gsm_num gsm_nan = NAN;
const gsm_num gsm_inf = HUGE_VALL;
const gsm_num gsm_realmin = LDBL_MIN;
const gsm_num gsm_realmax = LDBL_MAX;
const gsm_num gsm_realeps = LDBL_EPSILON;

// Use quadruple precision floating-point for intermediate calculations
#elif gsm_intermediate_numerical_type == 4
typedef __float128 gsm_num;
#define gsm_exp_wrapper(x) expq((x))
#define gsm_log_wrapper(x) logq((x))
#define gsm_expm1_wrapper(x) expm1q((x))
#define gsm_log1p_wrapper(x) log1pq((x))
#define gsm_fma_wrapper(x,y,a) fmaq((x),(y),(a))
#define gsm_max(x,y) fmaxq((x),(y))
#define gsm_min(x,y) fminq((x),(y))
#define gsm_nan nanq(NULL)
#define gsm_inf __extension__ HUGE_VALQ
#define gsm_realmin __extension__ FLT128_MIN
#define gsm_realmax __extension__ FLT128_MAX
#define gsm_realeps __extension__ FLT128_EPSILON

#endif

#if (gsm_intermediate_numerical_type >= 3) && gsm_use_matlab_arithmetic
#error "<use_matlab_arithmetic> cannot be true when using more than double precision for intermediate calculations" 
#endif

#if gsm_use_fma
#define gsm_fma(x,y,a) gsm_fma_wrapper((x),(y),(a))
#else
#define gsm_fma(x,y,a) ((x)*(y) + (a))
#endif

// mat_num is the numerical type used for input and output of the matlab program.
#if gsm_single_precision_matlab_args
typedef mxSingle mat_num;
#define mex_create_matrix(m,n) mxCreateNumericMatrix((m), (n), mxSINGLE_CLASS, mxREAL)
#define mex_is_correct_numerical_class(p) (mxIsSingle(p))
#define matlab_numeric_class_name "single"
#else
typedef mxDouble mat_num;
#define mex_create_matrix(m,n) mxCreateNumericMatrix((m), (n), mxDOUBLE_CLASS, mxREAL)
#define mex_is_correct_numerical_class(p) (mxIsDouble(p))
#define matlab_numeric_class_name "double"
#endif

// Converts an mxArray pointer to a numeric-array pointer
#define mex_get_data_pointer(arr_ptr) (((arr_ptr) == NULL) ? NULL : (mat_num*)mxGetData(arr_ptr))

// Data types used to represent sizes and indices, respectively.
// size_type must be large enough to represent the dimension d of the input vector z,
// and (2*k)^2. 
// index_type can be signed or unsigned.
typedef mwSize  size_type;
typedef mwIndex index_type;

// Generic max and min
#define max(x,y) ((x) >= (y) ? (x) : (y))
#define min(x,y) ((x) <= (y) ? (x) : (y))

// Max and min for gsm_num type
#define gsm_subplus(x) gsm_max((x), gsm_zero)
#define gsm_subminus(x) gsm_max(-(x), gsm_zero)

// Ensures the value is between 0 and 1
#define gsm_trim(x) gsm_max(gsm_min((x), gsm_one), gsm_zero)

// Cast from mat_num to gsm_num and backwards
#define gsm_cast(x) ((gsm_num)(x))
#define gsm_uncast(x) ((mat_num)(x))

#define gsm_zero gsm_cast(0)
#define gsm_one gsm_cast(1)

// Memory allocation and deallocation commands.
// s - size in bytes
#define malloc_gsm_void(s) (mxMalloc(s))
#define memfree_gsm(p) (mxFree(p))

// Allocation of an n-dimensional numerical array
#define malloc_gsm_num(n) ((gsm_num*)malloc_gsm_void((n)*sizeof(gsm_num)))

// Copy the contents of a mat_num array to / from a gsm_num array of size n.
void copy_mat2gsm(const mat_num* arr_src, gsm_num* arr_tgt, size_type n);
void copy_gsm2mat(const gsm_num* arr_src, mat_num* arr_tgt, size_type n);

// Offset used to access the 0-based array z as if it is 1-based.
const index_type ofs = 1;

// A struct containing input and output arguments passed from matab
typedef struct
{
    mwSize d;
    mwSize k;
    mat_num gamma;
    const mat_num* z;
    
    const mxArray* z_arr;
    
    mxArray** mu_arr;
    mxArray** theta_arr;
    mxArray** dl_arr;
    mxArray** bz_arr;
    mxArray** bzl_arr;
    mxArray** bzr_arr;
    mxArray** delta_arr;
    mxArray** alpha_arr;
    mxArray** log_alpha_arr;
} matArgsType;


// ======= Main functions =========

matArgsType readArguments(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

void calc_gsm_reduction(
        /* Input:  */ const gsm_num* z, size_type d, size_type k, gsm_num gamma,
        /* Output: */ gsm_num* mu_ptr, gsm_num* theta, size_type* dl_ptr, gsm_num* bz, gsm_num* bzl, gsm_num* bzr, gsm_num* delta, gsm_num* alpha, gsm_num* log_alpha);

void calc_theta(const gsm_num* z, size_type d, size_type k, gsm_num gamma, size_type dl, const gsm_num *bz, const gsm_num *z_ord, const gsm_num* bzl, const gsm_num* zl_ord, const gsm_num *bzr, const gsm_num* delta, const gsm_num* alpha, const gsm_num* log_alpha, gsm_num* theta);

void alg3(const gsm_num* z, size_type d, size_type k, gsm_num gamma, size_type s, gsm_num* mu, gsm_num* bz, gsm_num* z_ord, size_type dr, gsm_num* bzr, gsm_num* zr_ord);

gsm_num alg4(const gsm_num z_i, size_type d, size_type k, gsm_num gamma, const gsm_num *b, const gsm_num *z_ord);

void alg5(const gsm_num zl_i, size_type dl, size_type k, gsm_num gamma, const gsm_num *bzl, const gsm_num *zl_ord, 
        gsm_num* theta_zl_i);

void alg6(size_type k, size_type dl, size_type dr, const gsm_num* z_ord, const gsm_num* zl_ord, const gsm_num* zr_ord, gsm_num* delta, gsm_num* alpha, gsm_num* log_alpha);

gsm_num alg7(index_type dr, size_type k, gsm_num gamma, const gsm_num* theta_zl_i, const gsm_num* bz, const gsm_num *bzl, const gsm_num* bzr, const gsm_num* delta, const gsm_num* alpha, const gsm_num* log_alpha);


// ======= Helper functions used for operations on gsm_num type =========

gsm_num gsm_exp(gsm_num x);
gsm_num gsm_log(gsm_num x);
gsm_num gsm_expm1(gsm_num x);
gsm_num gsm_log1p(gsm_num x);


// ======================================================================//
// Main mex function
// ======================================================================//

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    matArgsType args = readArguments(nlhs, plhs, nrhs, prhs);
    
    // Process input arguments and allocate memory for output
    size_type d = args.d;
    size_type k = args.k;
    gsm_num gamma = (gsm_num)args.gamma;
    
    gsm_num* z     = malloc_gsm_num(d);
    copy_mat2gsm(mex_get_data_pointer(args.z_arr), z, d);
    
    gsm_num mu;
    
    gsm_num* theta = malloc_gsm_num(d);
    size_type dl;
    
    gsm_num* bz  = malloc_gsm_num(k+1);
    gsm_num* bzl = malloc_gsm_num(max(2*k-2,1)+1);
    gsm_num* bzr = malloc_gsm_num(k+1);
    
    gsm_num* delta = malloc_gsm_num(k+1);
    gsm_num* alpha = malloc_gsm_num(k+1);
    gsm_num* log_alpha = malloc_gsm_num(k+1);
    
    // Perform main calculation
    calc_gsm_reduction(z, d, k, gamma, &mu, theta, &dl, bz, bzl, bzr, delta, alpha, log_alpha);
    
    // Return output
    size_type dr = d-dl;
    
    copy_gsm2mat(&mu, mex_get_data_pointer(*args.mu_arr = mex_create_matrix(1, 1)), 1);
    copy_gsm2mat(theta, mex_get_data_pointer(*args.theta_arr = mex_create_matrix(1, d)), d);
    
    *mex_get_data_pointer(*args.dl_arr = mex_create_matrix(1, 1)) = (mat_num)dl;
    
    copy_gsm2mat(bz, mex_get_data_pointer(*args.bz_arr = mex_create_matrix(k+1, 1)), k+1);
    copy_gsm2mat(bzl, mex_get_data_pointer(*args.bzl_arr = mex_create_matrix(dl+1, 1)), dl+1);
    copy_gsm2mat(bzr, mex_get_data_pointer(*args.bzr_arr = mex_create_matrix(min(k,dr)+1, 1)), min(k,dr)+1);
    copy_gsm2mat(delta, mex_get_data_pointer(*args.delta_arr = mex_create_matrix(1, k+1)), k+1);
    copy_gsm2mat(alpha, mex_get_data_pointer(*args.alpha_arr = mex_create_matrix(1, k+1)), k+1);
    copy_gsm2mat(log_alpha, mex_get_data_pointer(*args.log_alpha_arr = mex_create_matrix(1, k+1)), k+1);
    
    memfree_gsm(z);
    memfree_gsm(theta);
    memfree_gsm(bz);
    memfree_gsm(bzl);
    memfree_gsm(bzr);
    memfree_gsm(delta);
    memfree_gsm(alpha);
    memfree_gsm(log_alpha);
}


// ======================================================================//
// Read and verify input arguments
// ======================================================================//
matArgsType readArguments(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Number of input arguments
    if (nrhs != 3)
    {
        mexPrintf("\nThis program should only be called from the main matlab file.\n");
        mexPrintf("Usage: [mu, theta, dl, bz, bzl, bzr, delta, alpha, log_alpha] = %s(z, k, gamma)\n", gsm_mex_filename);
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:nrhs",
                "Three input arguments are required");
    }
    
    if ((nlhs < 2) || (nlhs > 9))
    {
        mexPrintf("\nThis program should only be called from the main matlab file.\n");
        mexPrintf("Usage: [mu, theta, dl, bz, bzl, bzr, delta, alpha, log_alpha] = %s(z, k, gamma)\n", gsm_mex_filename);
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:nlhs",
                "Two to nine output arguments are required");
    }
    
    const mxArray* z_arr     = prhs[0];
    const mxArray* k_arr     = prhs[1];
    const mxArray* gamma_arr = prhs[2];
    
    mxArray** mu_arr    = &plhs[0];
    mxArray** theta_arr = &plhs[1];
    mxArray** dl_arr    = (nlhs <= 2 ? NULL : &plhs[2]);
    mxArray** bz_arr    = (nlhs <= 3 ? NULL : &plhs[3]);
    mxArray** bzl_arr   = (nlhs <= 4 ? NULL : &plhs[4]);
    mxArray** bzr_arr   = (nlhs <= 5 ? NULL : &plhs[5]);
    mxArray** delta_arr = (nlhs <= 6 ? NULL : &plhs[6]);
    mxArray** alpha_arr = (nlhs <= 7 ? NULL : &plhs[7]);
    mxArray** log_alpha_arr = (nlhs <= 8 ? NULL : &plhs[8]);
    
    
    // Verify z
    if (!mex_is_correct_numerical_class(z_arr) || mxIsSparse(z_arr) || mxIsComplex(z_arr) || (mxGetNumberOfElements(z_arr) < 2))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:z",
                "z should be a nonempty matrix of class %s", matlab_numeric_class_name);
    }
    
    const size_type d = mxGetNumberOfElements(z_arr);
    const mat_num* z = mex_get_data_pointer(z_arr);
    
    // k
    if ((!mex_is_correct_numerical_class(k_arr)) || mxIsSparse(k_arr) ||
            mxIsComplex(k_arr) || (mxGetNumberOfElements(k_arr) != 1))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:k",
                "k must be a scalar of class %s", matlab_numeric_class_name);
    }
    
    mat_num k_num = *mex_get_data_pointer(k_arr);
    
    if (mxIsNaN(k_num) || mxIsInf(k_num))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:k",
                "k should not be inf or nan");
    }
    
    size_type k = (index_type)k_num;
    
    if ((mat_num)k != k_num)
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:k",
                "the value of k must be an integer (of class %s)", matlab_numeric_class_name);
    }
    
    if ((k > d-1) || (k < 1))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:k",
                "k must be between 1 and numel(z) - 1");
    }
    
    const size_type dl = max(2*k-2,1);
    
    for (index_type i=1; i<=d; ++i)
    {
        if (mxIsNaN(z[i-ofs]))
        {
            mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:nan",
                    "z should not contain NaNs");
        }
        
        if (mxIsInf(z[i-ofs]))
        {
            mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:inf",
                    "z should not contain infs");
        }
    }
    
    mat_num zl_min = z[1-ofs];
    
    for (index_type i=2; i<=dl; ++i)
    {
        zl_min = min(zl_min, z[i-ofs]);
    }
    
    for (index_type i=dl+1; i<=d; ++i)
    {
        if (z[i-ofs] > zl_min)
        {
            mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:sort",
                    "z must be sorted such that z[i] >= z[j] for all 1 <= i <= 2*k-2 <= j <= d");
        }
    }
    
    // gamma
    if (!mex_is_correct_numerical_class(gamma_arr) || mxIsComplex(gamma_arr) ||
            (mxGetNumberOfElements(gamma_arr) != 1))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:gamma",
                "gamma should be a scalar of class %s", matlab_numeric_class_name);
    }
    
    const mat_num gamma = (mat_num)mxGetScalar(gamma_arr);
    
    if (mxIsNaN(gamma))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:nan",
                "gamma should not be NaN");
    }
    
    if ((gamma <= (mat_num)0) || mxIsInf(gamma))
    {
        mexErrMsgIdAndTxt("GSM_V5_1_MEX:readArguments:gamma",
                "gamma should be nonnegative and finite");
    }
    
    
    matArgsType args;
    
    args.z = z;
    args.d = d;
    args.k = k;
    args.gamma = gamma;
    
    args.z_arr = z_arr;
    
    args.mu_arr = mu_arr;
    args.theta_arr = theta_arr;
    args.dl_arr = dl_arr;
    args.bz_arr = bz_arr;
    args.bzl_arr = bzl_arr;
    args.bzr_arr = bzr_arr;
    args.delta_arr = delta_arr;
    args.alpha_arr = alpha_arr;
    args.log_alpha_arr = log_alpha_arr;
    
    return args;
}



// ======================================================================//
// Algorithm functions
// ======================================================================//


// ======================================================================//
// Main function for calculating the GSM in the reduced case
// 0 < gamma < inf, 1 <= k <= d/2 and assuming that z is sorted such that z_i >= z_j
// for i = 1,...,dl, j = dl+1,...,d, with dl = max(1,2*k-2).
//
// Output arguments:
// mu_ptr: Pointer to a gsm_num where the value of mu_{k,gamma{(z) will be returned.
// theta:  Array of d nums
// dl_ptr: Pointer to a size_type, where dl will be returned.
// bz:     Array of k+1 nums
// bzl:    Array of max(2*k-2,1)+1 nums
// bzr:    Array of k+1 nums
// delta, alpha, log_alpha: Arrays of k+1 nums
// ======================================================================//
void calc_gsm_reduction(
        /* Input:  */ const gsm_num* z, size_type d, size_type k, gsm_num gamma,
        /* Output: */ gsm_num* mu_ptr, gsm_num* theta, size_type* dl_ptr, gsm_num* bz, gsm_num* bzl, gsm_num* bzr, gsm_num* delta, gsm_num* alpha, gsm_num* log_alpha)
{
    size_type dl = max(2*k-2, 1);
    size_type dr = d - dl;
    
    // Output dl
    *dl_ptr = dl;
    
    gsm_num* zl_ord = malloc_gsm_num((dl+1));
    gsm_num* zr_ord = malloc_gsm_num((min(k,dr)+1));
    
    // Calculate mu_{k,gamma}(z), b_{q,gamma}(z) for q=0,...,k and b_{q,gamma}(zr) for
    // q=0,...,min(k,dr), and the order statistics zr_(0),...,zr_(min(k,dr))
    alg3(z, d,  k, gamma, k,  mu_ptr, bz,  NULL,   dr, bzr,  zr_ord);
    
    // Calculate b_{q,gamma}(zl) for q=0,...,dl and the order statistics zl_(0),...,zl_(dl)
    alg3(z, dl, 0, gamma, dl, NULL,   bzl, zl_ord, 0,  NULL, NULL);
    
    // Calculate delta_{k,q}(zl,zr) and alpha^{dl,dr}_{k,q} for q=0,...,k, as well as the
    // logarithms of the alpha.
    alg6(k, dl, dr, zl_ord, zl_ord, zr_ord, delta, alpha, log_alpha);
    
    // Finally, calculate the vector function theta_{k,gamma}(z)
    calc_theta(z, d, k, gamma, dl, bz, zl_ord, bzl, zl_ord, bzr, delta, alpha, log_alpha, theta);
    
    memfree_gsm(zl_ord);
    memfree_gsm(zr_ord);
}


// Main function to calculate theta_{k,gamma}(z)
// Input: z=[zl,zr] in R^d with d=dl+dr
//        k s.t. 1 <= k <= dl, 0 < gamma < inf
//        b_{q,gamma}(z) for q=0,...,k
//        b_{q,gamma}(zl) for q=0,...,dl
//        zl_(q) for q=0,...,dl
//        delta_{k,q}(zl,zr), alpha^{dl,dr}_{k,q} and log(alpha^{dl,dr}_{k,q}) for q=0,...,k
// It is assumed that min_i zl_i >= max_j zr_j, and that for all i > dl, calculation of 
// theta_{k,gamma}^i(z) by alg4() is stable. Namely, that for all those values of i,
// z[i] <= z_(k) + ( (b_{k,gamma}(z) - b_{k-1,gamma}(z)) + log((d-k+1)/k) ) / gamma
void calc_theta(
        /* Input:  */ const gsm_num* z, size_type d, size_type k, gsm_num gamma, size_type dl, const gsm_num *bz, const gsm_num *z_ord, const gsm_num* bzl, const gsm_num* zl_ord, const gsm_num *bzr, const gsm_num* delta, const gsm_num* alpha, const gsm_num* log_alpha,
        /* Ourput: */ gsm_num* theta)
{
    const size_type dr = d-dl;
    
    const gsm_num stability_thresh = z_ord[k] + ( (bz[k] - bz[k-1]) + gsm_log(gsm_cast(d-k+1)/gsm_cast(k)) ) / gamma;
    
    gsm_num* theta_zl_i = malloc_gsm_num(dl+1);
    
    for (index_type i=d; i >= 1; --i)
    {
        if ((i < d) && (z[i-ofs] == z[i+1-ofs]))
        {
            theta[i-ofs] = theta[i+1-ofs];
        }
        else if ((i > dl) || (z[i-ofs] <= stability_thresh))
        {
            // Stability of Algorithm 4 is guaranteed for i > dl, and for any other i for
            // which z_i is below the stability threshold
            theta[i-ofs] = alg4(z[i-ofs], d, k, gamma, bz, z_ord);
        }
        else
        {
            // For the rest of the indices i, calculate theta_{k,gamma}^i(z) from
            // theta_{q,gamma}^i(zl) for q=0,...,k
            alg5(z[i-ofs], dl, k, gamma, bzl, zl_ord, theta_zl_i);
            theta[i-ofs] = alg7(dr, k, gamma, theta_zl_i, bz, bzl, bzr, delta, alpha, log_alpha);
        }
    }
    
    memfree_gsm(theta_zl_i);
}


// Calculates mu_{k,gamma}(z) and b_{q,gamma}(z) for q=0,...,s.
// Based on Algorithm 3 from [1].
// Input: z = [z_1,...,z_d]
//        1 <= k <= s <= d
//        0 < gamma < inf
//        dr s.t. 1 <= dr <= d (Optional. Set to 0 to ignore.).
//
// Output: mu    = mu_{k,gamma}(z). Ignored if NULL
//         bz    = [b_{0,gamma}(z), ..., b_{s,gamma}(z)]
//         z_ord = [z_(0), ..., z_(s)], where z_(i) is the i-th largest entry
//             of z and z_(0) = inf.
//
// Optional output: (Ignored if dr = 0)
//         bzr = [b_{0,gamma}(zr), ..., b_{s2,gamma}(zr)], where z = [zl, zr], zr is of
//           size dr, and s2 = min(s, dr).
//         zr_ord = [zr_(0), zr_(1), ..., zr_(s2)]
void alg3(const gsm_num* z, size_type d, size_type k, gsm_num gamma, size_type s, gsm_num* mu, gsm_num* bz, gsm_num* z_ord, size_type dr, gsm_num* bzr, gsm_num* zr_ord)
{
    gsm_num* b1 = malloc_gsm_num(s+1);
    gsm_num* b2 = malloc_gsm_num(s+1);
    gsm_num* v1 = malloc_gsm_num(s+1);
    gsm_num* v2 = malloc_gsm_num(s+1);
    
    gsm_num *b, *v, *btilde, *vtilde;
    
    int vecSwitch = 1;
    
    // Update pointers
    b      = (vecSwitch == 1 ? b1 : b2);
    v      = (vecSwitch == 1 ? v1 : v2);
    btilde = (vecSwitch == 1 ? b2 : b1);
    vtilde = (vecSwitch == 1 ? v2 : v1);
    
    // Initialization
    // Throughout the loop, b[0], btilde[0] should be 0 and v[0], vtilde[0] should be inf.
    b[0]      = gsm_zero;
    btilde[0] = gsm_zero;
    
    v[0]      = gsm_inf;
    vtilde[0] = gsm_inf;
    
    // Main loop
    for (index_type r=d; r >= 1; --r)
    {
        // - Iteration r requires vtilde(q) and btilde(q) for q=0,...,min(d-r,s)
        //   and updates v(q) and b(q) for q=1,...,min(d-r+1,s).
        // - At the end of iteration r, v[q] and b[q] are updated for q=0,...,s.
        
        // q=0 is handled implicitly, by keeping v[0]=vtilde[0]=inf and b[0]=btilde[0]=0.
        for (index_type q=1; q <= min(s,d-r); ++q)
        {
            v[q] = gsm_max(gsm_min(z[r-ofs], vtilde[q-1]), vtilde[q]);
            
            gsm_num xi = gamma*(z[r-ofs]-vtilde[q]);
            
            // Alternative in case z is fully sorted in decreasing order
            //gsm_num xi = gamma*(z[r-ofs]-z[r+q-ofs]);
            
            gsm_num eta = (btilde[q] - btilde[q-1]) - xi;

            if (eta <= gsm_zero)
            {
                // Note that if z is fully sorted in nonincreasing order, eta is guaranteed to
                // be nonpositive, and subminus(xi) = 0.
                gsm_num log1p_term = gsm_log1p( (gsm_cast(d-r-q+1) / gsm_cast(d-r+1)) * gsm_expm1(eta));
                b[q] = (btilde[q-1] - gsm_subminus(xi)) + log1p_term;
            }
            else
            {
                gsm_num log1p_term = gsm_log1p( (gsm_cast(q) / gsm_cast(d-r+1)) * gsm_expm1(-eta));
                b[q] = (btilde[q] - gsm_subplus(xi)) + log1p_term;
            }
            
            // b[q] should be non-positive anyway, so clip any possible negativity caused
            // by numerical error.
            b[q] = gsm_min(b[q], gsm_zero);
        }
        
        // Handle q = d-r+1
        if (s >= d-r+1)
        {
            v[d-r+1] = gsm_min(vtilde[d-r], z[r-ofs]);
            b[d-r+1] = gsm_zero;
        }
        
        if (r == d-dr+1)
        {
            // Return z_{q}^{r_extra} for q=0,1,...,s and b_{k,gamma}^{r_extra}(z)
            for (index_type q = 0; q <= min(s, dr); ++q)
            {
                bzr[q] = b[q];
                zr_ord[q] = v[q];
            }
        }
        
        // Assign btilde := b, vtilde := v by switching pointers
        vecSwitch = -vecSwitch;
        
        b      = (vecSwitch == 1 ? b1 : b2);
        v      = (vecSwitch == 1 ? v1 : v2);
        btilde = (vecSwitch == 1 ? b2 : b1);
        vtilde = (vecSwitch == 1 ? v2 : v1);
    }
    
    // Return b_{k,gamma}(z)
    for (index_type q=0; q <= s; ++q)
    {
        bz[q] = btilde[q];
    }
    
    // Return z_(q) for q=0,1,...,s
    if (z_ord != NULL)
    {
        for (index_type q = 0; q <= s; ++q)
        {
            z_ord[q] = vtilde[q];
        }
    }
    
    // Return mu_{k,gamma}(z)
    if (mu != NULL)
    {
        *mu = gsm_zero;
        
        for (index_type q=k; q >= 1; --q)
        {
            *mu += vtilde[q];
            //*mu += z[q-ofs]; // This can be used if z is fully sorted in decreasing order
        }
        
        *mu += btilde[k] / gamma;
    }
    
    // Release temporary variables
    memfree_gsm(b1);
    memfree_gsm(b2);
    memfree_gsm(v1);
    memfree_gsm(v2);
}


// Calculates theta_{k,gamma}^i(z) by forward recursion.
// Algorithm 4 in [1].
//
// Input: z = [z_1, ..., z_d], 1 <= k,i <= d, 0 < gamma < inf
//        b = [b_{0,gamma}(z), ..., b_{k,gamma}(z)]
//        z_ord = [z_(0), ..., z_(k)]
//
// Output: theta_{k,gamma}^i(z)
inline gsm_num alg4(const gsm_num z_i, size_type d, size_type k, gsm_num gamma, const gsm_num *bz, const gsm_num *z_ord)
{
    gsm_num xi = gsm_zero;
    gsm_num factor_curr;
    
    for (index_type q=1; q <=k; ++q)
    {
        factor_curr = (gsm_cast(q) / gsm_cast(d-q+1)) * 
                gsm_exp( gsm_fma(gamma, z_i - z_ord[q], bz[q-1] - bz[q]) );
        
        xi = gsm_min(gsm_one, factor_curr * (gsm_one - xi));
    }
       
    return gsm_trim(xi);
}



// Algorithm 5 in [1]
//
// Input: zl = [zl_1, ..., zl_dl], 1 <= i <= dl, 0 < gamma < inf
//        bzl = b_{q,gamma}(zl) for q=0,...,dl
//        zl_ord = zl_(q) for q=0,...,dl
//
// Output: theta_{q,gamma}^i(zl) for q=0,...,k
void alg5(const gsm_num zl_i, size_type dl, size_type k, gsm_num gamma, const gsm_num *bzl, const gsm_num *zl_ord, 
        gsm_num* theta_zl_i)
{
    theta_zl_i[0]  = gsm_zero;
    theta_zl_i[dl] = gsm_one;
             
    index_type q_hat = k+1;
    
    for (index_type q=1; q <= k; ++q)
    {
        gsm_num eta = gamma*(zl_i - zl_ord[q]) + (bzl[q-1] - bzl[q]);
        gsm_num frac_term = gsm_cast(q) / gsm_cast(dl-q+1);
        
        if (eta <= -gsm_log(frac_term))
        {
            theta_zl_i[q] = gsm_min(gsm_one, frac_term * gsm_exp(eta)) * (gsm_one - theta_zl_i[q-1]);
        }
        else
        {
            q_hat = q;
            break;
        }
    }
       
    if (q_hat <= k)
    {
        for (index_type q = dl-1; q+1 >= q_hat+1; --q)
        {
            gsm_num eta = (bzl[q+1] - bzl[q]) - gamma*(zl_i - zl_ord[q+1]);
            gsm_num frac_term = gsm_cast(dl-q) / gsm_cast(q+1);
            
            theta_zl_i[q] = gsm_max(gsm_zero, gsm_fma( -frac_term*gsm_exp(eta), theta_zl_i[q+1], gsm_one ) );
        }
    }
}



// Calculates the entries Delta_{k,t}(zl,zr) and the coefficients alpha_{k,t}^{dl,dr} for
// t=0,...,k. Based on Algorithm 6 in [1].
// Input:
//   z = [zl, zr] in R^d where zl in R^dl, zr in R^dr and dl,dr >= 1
//   (z, zl and zr are not required explicitly)
//   1 <= k <= d
//   z_ord = [z_(0), ..., z_(k)]
//   zl_ord = [zl_(0), ..., zl_(k)]
//   zr_ord = [zr_(0), ..., zr_(min(k,dr))]
//
// Output:
//    delta:     Delta_{k,t}(zl,zr) for t=0,...,k
//    alpha:     alpha_{k,t}^{dl,dr} for t=0,...,k
//    log_alpha: log(alpha_{k,t}^{dl,dr}) for t=0,...,k
//
// alpha are returned as normal nonzero floating-point numbers, or zero
// (i.e. alpha does not contain nonzero subnormal values.)
inline void alg6(size_type k, size_type dl, size_type dr, const gsm_num* z_ord, const gsm_num* zl_ord, const gsm_num* zr_ord, gsm_num* delta, gsm_num* alpha, gsm_num* log_alpha)
{
    const size_type d = dl+dr;
    
    // Initialize for q=0
    delta[0] = gsm_zero;
    alpha[0] = gsm_one;
    
    // t_mode is the t for which alpha^{dl,dr}_{q,t} is maximal
    index_type t_mode_prev, t_mode = 0;
    
    for (index_type q=1; q<=k; ++q)
    {
        // Range of t values where alpha_{q,t} and delta_{q,t} are not identically zero
        index_type ta = max(dr,q) - dr; // A safe way to denote max(0, q-dr) with unsigned integers
        index_type tb = min(q, min(k,dl));
        
        // Calculate alpha{q, t} at t=t_mode
        t_mode_prev = t_mode;
        t_mode = ( (size_type)(dl+1)*((size_type)(q+1)) ) / (size_type)(d+2);
        
        // t_mode always equals t_mode_prev or t_mode_prev+1.
        // Calculate alpha_{q, t_mode} from alpha_{q-1, t_mode_prev}.
        if (t_mode == t_mode_prev)
        {
            gsm_num fac = gsm_cast((size_type)(dr+t_mode-q+1)*(size_type)q) /
                    gsm_cast((size_type)(d-q+1)*(size_type)(q-t_mode));
            
            alpha[t_mode] = fac * alpha[t_mode_prev];
        }
        else if (t_mode == t_mode_prev+1)
        {
            gsm_num fac = gsm_cast((size_type)(dl-t_mode+1)*(size_type)q) /
                    gsm_cast((size_type)(d-q+1)*(size_type)t_mode);
            alpha[t_mode] = fac * alpha[t_mode_prev];
        }
        else
        {
            // This should not happen.
        }
        
        // Calculate delta{q,t} for t=ta,...,tb
        for (index_type t = tb; t >= max(ta,1); --t)
        {
            if (z_ord[q] >= zl_ord[t])
            {
                delta[t] = delta[t-1] + (z_ord[q] - zl_ord[t]);
            }
            else
            {
                delta[t] = delta[t] + (z_ord[q] - zr_ord[q-t]);
            }
        }
        
        if (ta == 0)
        {
            delta[0] += z_ord[q] - zr_ord[q];
        }
        else
        {
            delta[ta-1] = gsm_zero;
        }
    }
    
    // Propagate knowledge of alpha_{q,t} and log_alpha_{q,t} from t_mode to the other t's.
    // Note that alpha_{q,t_mode} is a normal positive floating point number, since it is 
    // greater or equal to 1/dl (since sum(alphas) = 1).
    t_mode = ( (size_type)(dl+1)*(size_type)(k+1) ) / (size_type)(d+2);
    log_alpha[t_mode] = gsm_log(alpha[t_mode]);
    
    index_type ta = max(k,dr) - dr; // A safe way to denote max(0, k-dr) with unsigned integers
    index_type tb = min(k,dl);
    
    index_type t = 0;
    
    while (t <= k)
    {
        if (t == ta)
        {
            t = tb+1;
            continue;
        }
        
        delta[t] = gsm_zero;
        alpha[t] = gsm_zero;
        log_alpha[t] = -gsm_inf;
        ++t;
    }
    
    // x+1 >= y+1 in the stopping condition is a safety mechanism for
    // unsigned int indices. Otherwise, if ta == 0, then different problems
    // arise when t_mode > 0 and when t_mode == 0. This can lead to an
    // endless loop with t being "negative".
    for (t = t_mode-1; t + 1 >= ta + 1; --t)
    {
        gsm_num fac = gsm_cast((size_type)(t+1)*(size_type)(dr-(k-t)+1)) /
                gsm_cast((size_type)(dl-t)*(size_type)(k-t));
        
        alpha[t] = fac * alpha[t+1];
        
        if (alpha[t] >= gsm_realmin)
        {
            // If alpha[t] is a normal floating point number, calculate its log directly.
            log_alpha[t] = gsm_log(alpha[t]);
        }
        else if (fac <= gsm_cast(0.5L))
        {
            // ...otherwise, calculate the log recursively.
            log_alpha[t] = log_alpha[t+1] + gsm_log(fac);
            
            // For safety, we do not allow subnormal numbers in the alphas. Instead, we use
            // zeros.
            alpha[t] = gsm_zero;
        }
        else
        {
            // If the multiplied factor is closer to 1 than to 0, use log1p
            gsm_num facm1 = gsm_cast((size_type)(dr+1)*(size_type)(t+1) - (size_type)(dl+1)*(size_type)(k-t)) /
                    gsm_cast((size_type)(dl-t)*(size_type)(k-t));
            
            log_alpha[t] = log_alpha[t+1] + gsm_log1p(facm1);
            alpha[t] = gsm_zero;
        }
    }
    
    
    for (t = t_mode+1; t <= tb; ++t)
    {
        gsm_num fac = gsm_cast((size_type)(dl-t+1)*(size_type)(k-t+1)) /
                gsm_cast((size_type)t*(size_type)(dr-k+t));
        
        alpha[t] = fac * alpha[t-1];
        
        if (alpha[t] >= gsm_realmin)
        {
            // If alpha[t] is a normal floating point number, calculate its log directly.
            log_alpha[t] = gsm_log(alpha[t]);
        }
        else if (fac <= gsm_cast(0.5L))
        {
            log_alpha[t] = log_alpha[t-1] + gsm_log(fac);
            alpha[t] = gsm_zero;
        }
        else
        {
            gsm_num facm1 = gsm_cast((size_type)(dl+1)*(size_type)(k-t+1) - (size_type)(dr+1)*(size_type)(t)) /
                    gsm_cast((size_type)(t)*(size_type)(dr-k+t));
            
            log_alpha[t] = log_alpha[t-1] + gsm_log1p(facm1);
            alpha[t] = gsm_zero;
        }
    }
}



gsm_num alg7(index_type dr, size_type k, gsm_num gamma, const gsm_num* theta_zl_i, 
        const gsm_num* bz, const gsm_num *bzl, const gsm_num* bzr, 
        const gsm_num* delta, const gsm_num* alpha, const gsm_num* log_alpha)
{
    const index_type qa = max(dr+1, k) - dr; // = max(1, k-dr);
    const index_type qb = k;
    
    gsm_num xi = gsm_zero;
    
    for (index_type q=qb; q >= qa; --q)
    {
        gsm_num alpha_curr = alpha[q];
        gsm_num arg_curr = ((bzl[q] + bzr[k-q]) - bz[k]) - gamma*delta[q];
        gsm_num exp_curr = gsm_exp(arg_curr);  
        gsm_num coeff_curr;
        
        // We need to avoid 0*inf (or any <subnormal number> * inf). Any other case, including
        // underflows, is ok.
        //
        // In the following two cases, 0*inf is guaranteed not to take place:
        // (recall that alpha_curr is always <= 1)
        //
        // 1. arg_curr <= 0. Then alpha_curr * exp(arg_curr) is surely ok.
        // 2. Both alpha_curr and exp_curr are normal floating point numbers. Since alpha_curr
        //    <= 1, the product alpha_curr * exp_curr is fine (even if underflows).
        //
        // Note that the way we calculated alpha in Alg. 6 guarantees that it is either normal
        // or zero, since we flushed subnormal values to zero.
        if ( (arg_curr <= gsm_zero) || ((alpha_curr > gsm_zero) && (exp_curr < gsm_realmax)) )
        {
            coeff_curr = gsm_min(gsm_one, alpha_curr * exp_curr);
        }        
        else
        {
            coeff_curr = gsm_exp(gsm_min(gsm_zero, log_alpha[q] + arg_curr));            
        }
        
        xi = gsm_fma(coeff_curr, theta_zl_i[q], xi);
    }
    
    return gsm_trim(xi);
}


// ======================================================================//
// Helper functions
// ======================================================================//

inline gsm_num gsm_exp(gsm_num x)
{
#if gsm_use_matlab_arithmetic
    mxArray *p1, *p2;
    mat_num* in = mex_get_data_pointer(p1 = mex_create_matrix(1,1));
    
    *in = (mat_num)x;
    
    mexCallMATLAB(1,&p2,1, &p1, "exp");
    
    mat_num* out = mex_get_data_pointer(p2);
    gsm_num result = gsm_cast(*out);
    
    mxDestroyArray(p1);
    mxDestroyArray(p2);
    
    return result;
#else
    return gsm_exp_wrapper(x);
#endif
}

inline gsm_num gsm_log(gsm_num x)
{
#if gsm_use_matlab_arithmetic
    mxArray *p1, *p2;
    mat_num* in = mex_get_data_pointer(p1 = mex_create_matrix(1,1));
    
    *in = (mat_num)x;
    
    mexCallMATLAB(1,&p2,1, &p1, "log");
    
    mat_num* out = mex_get_data_pointer(p2);
    gsm_num result = gsm_cast(*out);
    
    mxDestroyArray(p1);
    mxDestroyArray(p2);
    
    return result;
#else
    return gsm_log_wrapper(x);
#endif
}


inline gsm_num gsm_expm1(gsm_num x)
{
#if gsm_use_matlab_arithmetic
    mxArray *p1, *p2;
    mat_num* in = mex_get_data_pointer(p1 = mex_create_matrix(1,1));
    
    *in = (mat_num)x;
    
    mexCallMATLAB(1,&p2,1, &p1, "expm1");
    
    mat_num* out = mex_get_data_pointer(p2);
    gsm_num result = gsm_cast(*out);
    
    mxDestroyArray(p1);
    mxDestroyArray(p2);
    
    return result;
#else
    return gsm_expm1_wrapper(x);
#endif
}

inline gsm_num gsm_log1p(gsm_num x)
{
#if gsm_use_matlab_arithmetic
    mxArray *p1, *p2;
    mat_num* in = mex_get_data_pointer(p1 = mex_create_matrix(1,1));
    
    *in = (mat_num)x;
    
    mexCallMATLAB(1,&p2,1, &p1, "log1p");
    
    mat_num* out = mex_get_data_pointer(p2);
    gsm_num result = gsm_cast(*out);
    
    mxDestroyArray(p1);
    mxDestroyArray(p2);
    
    return result;
#else
    return gsm_log1p_wrapper(x);
#endif
}


inline void copy_gsm2mat(const gsm_num* arr_src, mat_num* arr_tgt, size_type n)
{
    if (arr_tgt == NULL)
    {
        return;
    }
    
    for (index_type i=0; i < n; ++i)
    {
        arr_tgt[i] = gsm_uncast(arr_src[i]);
    }
}

inline void copy_mat2gsm(const mat_num* arr_src, gsm_num* arr_tgt, size_type n)
{
    if (arr_tgt == NULL)
    {
        return;
    }
    
    for (index_type i=0; i < n; ++i)
    {
        arr_tgt[i] = gsm_cast(arr_src[i]);
    }
}
