# BBMM implementation

## Summary of use case
- term needed to calculate
    - loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{log}|\hat{K}_{XX}|, $
    - gradient of loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{Tr}(\hat{K}_{XX}^{-1}\frac{d\hat{K}_{XX}}{d\theta})$←in gpytorch, this is comupted using "autograd" but in jax implementation we may should use analytical derivative (because of the different autograd scheme)
    - prediction of mean:  $\hat{K}_{XX}^{-1}\boldsymbol{y}$
    - prediction of std: $\hat{K}_{XX}^{-1}\boldsymbol{k}_{X\boldsymbol{x}^*}$
- variation of calculation method
    1. calculate all three terms: linear solve, log determinant, trace term
    2. calculate only linear solve


## Todo
### must
- ~~develop a test code~~
- ~~make mmm_A function~~
    - ~~for K~~
    - ~~for $\frac{d\hat{K}_{XX}}{d\theta}$~~←computation is not efficient at this point.
    - ~~analyze the time complexity~~
- ~~modify mpcg algorithm to receive mmm function and check if we can solve~~
  - linear_cg.py and pivoted_cholesky.py
    - for pivoted cholesky, implementing linear_operator class may be needed
      - can obtain each row, _diagonal, shape, __getitem__, etc.
- ~~apply optimization stopping for alhpa, beta to obtain better $\mathrm{Tr}(\hat{K}_{XX}^{-1}\frac{d\hat{K}_{XX}}{d\theta})$~~
    - ~~has_convergedに基づいてalphaをzeroでmaskする(lax.select)~~
    - epsに基づいて,alpha, betaのzero divisionを避ける(lax.cond for each terms →lax .select)
      - implemented, but **almost no change**
- check if trace term is really calculated correctly
   - in gpytorch imprementation, probe_vector is generated from zero_mean_mvn_samples those variance is precond_lt $P=LL^t+\sigma^2I$
      - `probe_vectors = precond_lt.zero_mean_mvn_samples(num_random_probes)`
      - Is implementing this gives us better result?
- calc whole log marginal likelihood and its derivative to check
- implement inference by our BBMM
  - simple sin curve
  - stokes eq in 2D
  - larger number of points (~10^5~7)

### should

    

## test
- what to test
    - each value of linear_solve, log determinant, trace term
- how to test
    - run `pytest ./test`. then all test in ./test is done

## desing of class
### linear_operator
- methods:
  - matmul, _diagonal, shape


## memo
- we can make mmm firster by using "lax.switch" and "lax.cond" instead of "if" statement.
- trace termは実際には計算されず、自動微分が用いられている、、！→trace termを計算する必要はない
  - [参考](ttps://github.com/cornellius-gp/gpytorch/discussions/1949)