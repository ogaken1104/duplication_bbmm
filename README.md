# BBMM implementation

## Summary of use case
- term needed to calculate
    - loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{log}|\hat{K}_{XX}|$
    - gradient of loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{Tr}(\hat{K}_{XX}^{-1}\frac{d\hat{K}_{XX}}{d\theta})$←in gpytorch, this is comupted using "autograd" but in jax implementation we may should use analytical derivative (because of the different autograd scheme)
    - prediction of mean:  $\hat{K}_{XX}^{-1}\boldsymbol{y}$
    - prediction of std: $\hat{K}_{XX}^{-1}\boldsymbol{k}_{X\boldsymbol{x}^*}$
- use cases
    1. calculate all three terms: linear solve, log determinant, trace term
    2. calculate only linear solve


## Todo
### must
- ~~calc whole log marginal likelihood and its derivative to check~~
- change preconditioner to use jax: very slow but successful
- implement inference by our BBMM
  - simple sin curve
    - ~~possible (if the condiiotn number is not too large ($\lt1e6$))~~
    - try to use analytical derivative
  - sin curve with its laplasian
    - possible
  - stokes eq in 2D
    - not converge (becuase of high cond. #)
  - try larger number of points (~10^5~7)

### should in future
- find better preconditioning way?
- elucidate why pivoted cholesky decompositoin by jax is slow than numpy x10.

## test
- what to test
    - each value of linear_solve, log determinant, trace term
- how to test
    - run `pytest ./test`. then all test in ./test is done

## Program Structue
- functions
  - pivoted_chokesky_xxx.py: pivoted cholesky decompoisition using numpy or jax
- operators: class for enablling matrix-matrix multiplicatoin
  - _linear_operator.py: base class
  - lazy_evaluated_kernel_tensor.py: class for covariance matrix
- utils: calculation scheme
  - calc_logdet
  - calc_trace
  - conjugate_gradient
  - preconditioner
  - mmm (do not use)

## desing of class
### linear_operator
- methods:
  - matmul, _diagonal, shape
### diag_linear_operator, root_linear_operator
- methods:
  - zero_mean_mvn_samples


### result of chekcing the component of loss
|problem setup|cond. number|linear_solve|log determinant|(trace term)|comment|
|--|--|--|--|--|--|
|max relative error|--|1%|5%|5%||
|random|4.8|.|.|.||
|sin1d (large noise: 1e-03)|1.9e5|.|.|.||
|sin1d (small noise: 1e-06)|2.0e8|.|F|.|precondition didn't work probably because digonal is alredy maximum|
|sin1d x5 (small noise: 1e-06)|1e6|.|.|F||

### result of optimizing trianing and prediction
|problem setup|cond. #|$\theta$ bbmm|$\theta$ default|$\theta$ gpytorch|prediction accuracy|
|--|--|--|--|--|--|
|sin x100(large eps)|7e2|0.14, 11.5|0.15, 10.8|0.19, 2.7|OK|
|sin (large eps), 1000 points|2e5|0.29, 0.14|0.32, 0.17||bbmm: 2e-3, default: 6e-5|
|sin (small eps), 10 points|2e3|0.75, 2.1|0.73, 1.9||bbmm: 5e-5, default: 1e-4|
|sin x100(small eps)|1e6|0.39, 2.39|0.13, 15.1||not perfect|

