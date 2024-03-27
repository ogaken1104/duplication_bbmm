# BBMM implementation

## Summary of use case
- term needed to calculate
    - loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{log}|\hat{K}_{XX}|$
    - gradient of loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{Tr}(\hat{K}_{XX}^{-1}\frac{d\hat{K}_{XX}}{d\theta})$←in gpytorch, this is comupted using "autograd" but in jax implementation we may should use analytical derivative (because of the different autograd scheme...?)
    - prediction of mean:  $\hat{K}_{XX}^{-1}\boldsymbol{y}$
    - prediction of std: $\hat{K}_{XX}^{-1}\boldsymbol{k}_{X\boldsymbol{x}^*}$
- use cases
    1. calculate all three terms: linear solve, log determinant, trace term
    2. calculate only linear solve

## Status
### progress
- implemented **calculate loss and derivative of loss** using BBMM
- confirmed that by using our implemented BBMM, we can calculate loss and derivative of loss **faster** than Cholesky, **for "simple" GP problem** (not Physics-Informed)
- precondition is implemented, but this doesn't work well so not tested carefully.
### problems
- in case of Physics-Informed problem, our implementation is **very slow than Cholesky**.
  - possible causes
    - for loop in block kernel matrices takes time
    - autograd of physics-informed kernel takes time
### To do
- elucidate why in case of Physics-informed problem, our implementation is very slow.
- implement **optimization of noise** hyper-parameter for diag linear operator
- (find **better preconditioning way**)

## results in master thesis (ogawa)
by running `tests/for_paper/check_time_for_sin.py`, we can obtain the results in the master thesis.

## Program Structue
- very important files to understand the program
  - **utils.calc_loss_dloss_linearop.py**: **most important one**, main calculation scheme
    - function `setup_loss_dloss_mpcg` returns a function (`loss_dloss_mpcg`) that calculates loss and derivative of loss using our re-implementation of BBMM
  - **test_calc_loss_sin_linearop.py**: test for calculating loss and derivative of loss for sin1d regression problem by **(1) our implementation, (2) Cholesky and (3) BBMM in GPytorch**
  - utils.conjugate_gradient.py
    - `mpcg_bbmm` implement the modified preconditioned conjugate gradient algorithm 
- functions
  - pivoted_chokesky_xxx.py: pivoted cholesky decompoisition using numpy or jax
- operators: class for enablling matrix-matrix multiplicatoin, mimicked from https://github.com/cornellius-gp/linear_operator
  - _linear_operator.py: base class for LinearOp
  - lazy_evaluated_kernel_tensor.py: class for covariance matrix
  - diag_linear_operator.py: class for diagonal matrix
  - root_linear_operator.py: class for root matrix
  - sum_linear_operator.py: class for sum of linear operators
  - psd_sum_linear_operator.py: class for sum of positive semi-definite linear operators
  - added_diag_linear_operator.py: class for adding diagonal matrix to linear operator
- utils: calculation scheme
  - calc_logdet
  - conjugate_gradient
  - preconditioner
  - calc_loss_dloss_linear_solve.py
  - calc_prediction_linear_solve.py
  - calc_loss_dloss.py
  - calc_prediction.py
  - ~~calc_trace~~

## Test
- compeleted
  - **tests/functions, tests/operators, tests/utils**, tests/loss_dloss, tests/for_paper
  - loss_dloss: test for calculating loss and derivative of loss for many problems
  - for_paper: test for obtaining the results in the master thesis
  - _mpcg_for_test: not latest no need to use
- how to test
    - run `pytest ./test`. then all test in ./test is done

<!--
### result of chekcing the component of loss
| problem setup                 | cond. number | linear_solve | log determinant | (trace term) | comment                                                             |
| ----------------------------- | ------------ | ------------ | --------------- | ------------ | ------------------------------------------------------------------- |
| max relative error            | --           | 1%           | 5%              | 5%           |                                                                     |
| random                        | 4.8          | .            | .               | .            |                                                                     |
| sin1d (large noise: 1e-03)    | 1.9e5        | .            | .               | .            |                                                                     |
| sin1d (small noise: 1e-06)    | 2.0e8        | .            | F               | .            | precondition didn't work probably because digonal is alredy maximum |
| sin1d x5 (small noise: 1e-06) | 1e6          | .            | .               | F            |                                                                     |

### result of optimizing trianing and prediction
| problem setup                | cond. # | $\theta$ bbmm | $\theta$ default | $\theta$ gpytorch | prediction accuracy       |
| ---------------------------- | ------- | ------------- | ---------------- | ----------------- | ------------------------- |
| sin x100(large eps)          | 7e2     | 0.14, 11.5    | 0.15, 10.8       | 0.19, 2.7         | OK                        |
| sin (large eps), 1000 points | 2e5     | 0.29, 0.14    | 0.32, 0.17       |                   | bbmm: 2e-3, default: 6e-5 |
| sin (small eps), 10 points   | 2e3     | 0.75, 2.1     | 0.73, 1.9        |                   | bbmm: 5e-5, default: 1e-4 |
| sin x100(small eps)          | 1e6     | 0.39, 2.39    | 0.13, 15.1       |                   | not perfect               |
-->

