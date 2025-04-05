DR-SM v1.0
========

Matlab code for Dimensionality reduction as a surrogate model (DRSM) for high-dimensional UQ, from:

Kim, J., Yi, S. R., & Wang, Z. (2024). Dimensionality reduction can be used as a surrogate model for high-dimensional forward uncertainty quantification. arXiv preprint arXiv:2402.04582.
https://doi.org/10.48550/arXiv.2402.04582.

Included toolboxes:
- drtoolbox - https://lvdmaaten.github.io/drtoolbox/
- Netlab - http://www1.aston.ac.uk/ncrg/ (Minor: https://uk.mathworks.com/matlabcentral/fileexchange/2654-netlab)
- VHGPR - from: Lázaro-Gredilla, M., & Titsias, M. K. (2011). Variational Heteroscedastic Gaussian Process Regression.

Notes:
 - This is a "modular framework", allowing you to select a suitable dimensionality reduction algorithm and conditional distribution model, with appropriately tuned parameters. The accuracy of the surrogate method depends on the effectiveness of the chosen DR and conditional modeling approaches.
This version uses simple choices, such as PCA and KDE.
 - Some functions in the Netlab toolbox have been modified to support the conditional distribution model used in DRSM.
 - A multi-output version (addressing the remarks in Section 6.5 of the paper) is also included.
 - The scripts implement a stochastic surrogate model for predicting responses of linear elastic bar and 3D space truss structures.

How to Run:
 - For single-output example: main_DRSM_singleY.m
 - For multi-output example: main_DRSM_multiY.m
