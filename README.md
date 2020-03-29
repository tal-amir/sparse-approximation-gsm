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
`>> x_sol = sparse_approx_gsm(A,y,k);`


Requirements
------------
This program requires the _Mosek_ optimization solver.  
https://www.mosek.com/downloads/  
  
Mosek requires a **user licence**. A personal academic licence can be requested [here](https://www.mosek.com/license/request/personal-academic/), and is normally sent immediately by email.  
The attached `mosek.lic` file should be placed in a directory called `mosek` under the user's home directory. For example:  
* Windows: `c:\users\_userid_\mosek\mosek.lic`  
* Unix / Linux / OS X: `/home/_userid_/mosek/mosek.lic`  

Then, the `toolbox/r2015aom` directory should be added to the Matlab path. e.g.,
`>> addpath('C:\Program Files\Mosek\9.1\toolbox\r2015aom');`

The code that compares our method with other methods requires the _Yalmip_ modeling toolbox.  
Yalmip can be downloaded [here](https://yalmip.github.io/download/) and placed in an arbitrary folder.  
To add Yalmip to the path:  
`>> addpath(genpath('C:\Dropbox\Temp\YALMIP'));`


Input & output arguments
------------------------
**Input:**  
`A` - Matrix of size n x d  
`y` - Column vector in R^n  
`k` - Target sparsity level 1 < k < d  
  
**Output:**  
`x_sol` - Estimated solution of (P0)  


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
