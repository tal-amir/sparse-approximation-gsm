Sparse approximation by the Generalized Soft-Min penalty
========================================================

Version 1.21, 23-Jul-2020

Tal Amir, Ronen Basri, Boaz Nadler  
Weizmann Institute of Science  
EMail: tal.amir@weizmann.ac.il  

This program estimates a solution of the _sparse approximation_ or _best subset selection_ problem: Given a vector _y_, matrix _A_ and sparsity level _k_, find a vector _x_ that minimizes  
  
(P0)         min _x_ ‖_A_*_x_-_y_‖₂ s.t. ‖_x_‖₀ ≤ _k_.  
  
The algorithm is based on [1].  


Requirements
------------
**Matlab**  
This program supports Matlab 2018b and onward, but may work with earlier versions. 
  
To compare our method with other methods, the _Mosek_ optimization solver and _YALMIP_ modeling toolbox are required.  
 
**Mosek**  
https://www.mosek.com/downloads/  
  
Mosek requires a _user licence_. A personal academic licence can be requested [here](https://www.mosek.com/license/request/personal-academic/),  
and is normally sent immediately by email.  

The attached `mosek.lic` file should be placed in `<home>/mosek`, where  
`<home>` is the user's home directory on the computer. For example:  
* Windows: `c:\users\<userid>\mosek\mosek.lic`  
* Unix / Linux / OS X: `/home/<userid>/mosek/mosek.lic`  

Then add Mosek's `toolbox/r2015aom` subdirectory to the Matlab path. e.g.,  
`>> addpath('C:\Program Files\Mosek\9.1\toolbox\r2015aom');`

**YALMIP**  
YALMIP can be downloaded [here](https://yalmip.github.io/download/) and placed in an arbitrary folder.  

All subdirectories of YALMIP should be added to the Matlab path.  
For example, if YALMIP is extracted to `C:\YALMIP`,  
`>> addpath(genpath('C:\YALMIP'));`


Usage
-----
A typical use is:  
`>> x_sol = sparse_approx_gsm(A,y,k);`

For more details, see the usage example scripts or the main documentation.

**Input**  
`A` - Matrix of size n x d  
`y` - Column vector in R^n  
`k` - Target sparsity level 1 < k < d  
  
**Output**  
`x_sol` - Estimated solution of (P0)  


Files
-----
`sparse_approx_gsm_v1_21.m`    - Main Matlab function  
`sparse_approx_gsm_v1_21.txt`  - Main documentation  
`README.md`                    - This readme  

`runExample*.m`             - Script files with simple usage examples  
`runCompareTrimmedLasso.m`  - A comparison between GSM and the DC-Programming and ADMM methods described in [2].
                          
`./gsm`  - Required in the matlab path.  
`./comparison`  - Required only for comparing GSM with other methods. 

References
----------
1. Tal Amir, Ronen Basri, Boaz Nadler (2020) - The Trimmed Lasso: Sparse Recovery Guarantees and Practical Optimization by the Generalized Soft-Min Penalty
2. Bertsimas, Copenhaver, Mazumder (2017) - The Trimmed Lasso: Sparsity and Robustness  
