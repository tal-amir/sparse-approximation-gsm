Sparse approximation by the Generalized Soft-Min penalty
========================================================

Version v1.22, 5-Nov-2020

Tal Amir, Ronen Basri, Boaz Nadler  
Weizmann Institute of Science  
EMail: tal.amir@weizmann.ac.il  

This program estimates a solution of the _sparse approximation_ or _best subset selection_ problem: Given a vector _y_, matrix _A_ and sparsity level _k_, find a vector _x_ that minimizes  
  
(P0)         min<sub>_x_</sub> ||_A_*_x_-_y_||<sub>2</sub> s.t. ||_x_||<sub>0</sub> <= _k_.  
  
The algorithm is based on [1].  

Basic usage:
`>> x_sol = sparse_approx_gsm(A,y,k);`

The `./utils` subdirectory should be added to the Matlab path:  
`>> addpath('./utils');`

For more details, see the main documentation.

Requirements
------------
**MATLAB**  
Matlab version 2018b is required, but the code my run on earlier versions.



Input & output arguments
------------------------
**Input**  
`A` - Matrix of size n x d  
`y` - Column vector in R^n  
`k` - Target sparsity level 1 < k < d  
  
**Output**  
`x_sol` - Estimated solution of (P0)  


Files
-----
`sparse_approx_gsm_v1_22.m`    - Main Matlab function  
`sparse_approx_gsm_v1_22.txt`  - Main documentation  
`README.md`                    - This readme  

`runExampleSmall.m`         
`runExampleMedium.m`        
`runExampleLarge.m`         - Usage examples with matrices of different sizes
                          
`./gsm`         - MEX C code to calculate the Generalized Soft-Min. Required in the Matlab path.  

References
----------
1. Tal Amir, Ronen Basri, Boaz Nadler (2020) - The Trimmed Lasso: Sparse Recovery Guarantees and Practical Optimization by the Generalized Soft-Min Penalty
2. Bertsimas, Copenhaver, Mazumder (2017) - The Trimmed Lasso: Sparsity and Robustness  
