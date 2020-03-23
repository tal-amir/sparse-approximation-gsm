Sparse approximation by the Generalized Soft-Min penalty
========================================================

Version v1.10, 23-Mar-2020

Tal Amir, Ronen Basri, Boaz Nadler  
Weizmann Institute of Science  
eMail: tal.amir@weizmann.ac.il  

Based on the paper [1].

This program estimates a solution of the _sparse approximation_ or _best subset selection_ problem: Given a vector _y_, matrix _A_ and sparsity level _k_, find a vector _x_ that minimizes

**(P0)**         min _x_ ||_A_*_x_-_y_||2 s.t. ||_x_||0 <= k.  

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
`k` - Target sparsity level 1 < k < d
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


