Sparse approximation by the Generalized Soft-Min penalty
========================================================

Version v1.10, 23-Mar-2020

Tal Amir, Ronen Basri, Boaz Nadler  
Weizmann Institute of Science  
eMail: tal.amir@weizmann.ac.il  

Based on the paper [1].

Given a vector y in R^n, matrix A in R^(n x d) and 1 < k < d, this function estimates a solution of the _sparse approximation_ or _best subset selection_ problem

(P0)         min_x ||A*x-y||_2 s.t. ||x||0 <= k.  

A typical use is:  
`>> [x_sol, sol] = sparse_approx_gsm(A,y,k,varargin);`

This program requires the Mosek optimization solver.  
https://www.mosek.com/downloads/

1. Tal Amir, Ronen Basri, Boaz Nadler (2020) - The Trimmed Lasso: Sparse Recovery Guarantees and Practical Optimization by the Generalized Soft-Min Penalty
2. Bertsimas, Copenhaver, Mazumder (2017) - The trimmed Lasso: Sparsity and Robustness  

Input arguments
---------------
`A` - Matrix of size n x d  
`y` - Column vector in R^n  
`k` - Target sparsity level  
`varargin` - name/value pairs of parameters. See documentation for details.

Output arguments
----------------
`x_sol` - Estimated solution vector to the sparse apprixmation problem  
`sol`   - A struct containing information about the solution.  
        See documentation for details.

Files
-----
`sparse_approx_gsm.m`     - Main Matlab function  
`sparse_approx_gsm.txt`   - Main documentation  
`README.md`               - Readme  

`runExample.m`            - A script with a simple usage example  
`runCompareTLS.m`         - A comparison between GSM and the DC-Programming and ADMM methods described in [2]. Requires the Yalmip modeling toolbox.  
                          
`./utils`                 - Files used by the main program. Required to be in the path.
`./comparison`            - Files used for comparing the performance of GSM to other methods.


