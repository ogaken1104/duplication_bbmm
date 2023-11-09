# BBMM implementation

## Summary of use case
- term needed to calculate
    - loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{log}|\hat{K}_{XX}|, $
    - gradient of loss: $\hat{K}_{XX}^{-1}\boldsymbol{y}, \mathrm{Tr}(\hat{K}_{XX}^{-1}\frac{d\hat{K}_{XX}}{d\theta})$
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
- apply optimization stopping for alhpa, beta to obtain better t_mat (cause of low accuracy of log determinat)
    - has_convergedに基づいてalpha, betaをzeroでmaskする(lax.cond for each val→lax.select)
      -  さらにそのmaskに基づいてalpha, betaで割り算をするときの計算を変える(lax.select)
   -  "has_converged"を計算することができるか確かめる、まずはnoprecondの場合から
- check if trace term is really calculated collectly
### should
- trace termの計算が本当に合っているかわからない
    - 検証する必要あり
    - ほぼ収束しているのに必要以上に計算を進めているから...？


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
we can make mmm firster by using "lax.switch" and "lax.cond" instead of "if" statement.