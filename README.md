Sparse approximation by the Generalized Soft-Min penalty
========================================================

Version 1.10, 23-Mar-2020

Tal Amir, Ronen Basri, Boaz Nadler  
Weizmann Institute of Science  
EMail: tal.amir@weizmann.ac.il  

This program estimates a solution of the _sparse approximation_ or _best subset selection_ problem: Given a vector _y_, matrix _A_ and sparsity level _k_, find a vector _x_ that minimizes

(P0)         min _x_ ‖_A_*_x_-_y_‖₂ s.t. ‖_x_‖₀ ≤ _k_.  

The algorithm is based on [1].

A typical use is:  
`>> [x_sol, sol] = sparse_approx_gsm(A,y,k,...);`

This program requires the Mosek optimization solver.  
https://www.mosek.com/downloads/


Input arguments
---------------
`A` - Matrix of size n x d  
`y` - Column vector in R^n  
`k` - Target sparsity level 1 < k < d  
`...` - name/value pairs of parameters. See documentation.

Output arguments
----------------
`x_sol` - Estimated solution of (P0)  
`sol`   - A struct containing information about the solution. See documentation.

Files
-----
`sparse_approx_gsm_v1_10.m`    - Main Matlab function  
`sparse_approx_gsm_v1_10.txt`  - Main documentation  
`README.md`                    - This readme  

`runExample.m`     - A script with a simple usage example  
`runCompareTLS.m`  - A comparison between GSM and the DC-Programming and ADMM methods described in [2]. Requires the Yalmip modeling toolbox.  
                          
`./utils`       - Files used by the main program. Required to be in the path.  
`./comparison`  - Files used for comparing the performance of GSM to other methods.

References
----------
1. Tal Amir, Ronen Basri, Boaz Nadler (2020) - The Trimmed Lasso: Sparse Recovery Guarantees and Practical Optimization by the Generalized Soft-Min Penalty
2. Bertsimas, Copenhaver, Mazumder (2017) - The Trimmed Lasso: Sparsity and Robustness  
