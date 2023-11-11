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
- check if logdet term is calculated correctly
  - ~~when precondition, error becomes large→is precond_log_det is accurate?~~
    - **why result changes when two ways of calc logdet in torch? this may be the key**
      - this will probably because the difference of random seed to generate random matrix
      - when giving the same t_mat, result was almost consistent, so it's ok
    - N<=800ではcholeskyを使って解く設定になっていたことが原因....！つまらないところで止まってしまっていた....！
  - ~~how to generate zs efficiently given preconditioner $P$~~
    - _pivoted_cholesky.py中の`self._precond_lt = PsdSumLinearOperator(RootLinearOperator(self._piv_chol_self), self._diag_tensor)`を実装でき、またlinear_operator classにzero_mean_mvn_samplesを実装できればよい
  - **preconditionerのrankを増やしても誤差が小さくならないのはかなり不可解**
    - これはおそらく，preconditionerを増やすことにより収束回数が小さくなり，その結果last_tridiag_iterが非常に小さくなっていることが原因であろう→max_tridiag_iterをある程度大きな値に設定しておくことが推奨される
      
- (check if trace term is really calculated correctly)
   - ~~in gpytorch imprementation, probe_vector is generated from zero_mean_mvn_samples those variance is precond_lt $P=LL^t+\sigma^2I$~~
      - `probe_vectors = precond_lt.zero_mean_mvn_samples(num_random_probes)`
      - Is implementing this gives us better result?
        - seems better
- calc whole log marginal likelihood and its derivative to check
- implement inference by our BBMM
  - simple sin curve
  - stokes eq in 2D
  - larger number of points (~10^5~7)

### should
- block covariance matrixに適したconditioningの方法はあるか？
    - スケールして、かつfの値を調整せずに、得たuを調整すればcondition numberを低く抑えられそう
    - Johnに調査を頼むことも考える
    

## test
- what to test
    - each value of linear_solve, log determinant, trace term
- how to test
    - run `pytest ./test`. then all test in ./test is done

## desing of class
### linear_operator
- methods:
  - matmul, _diagonal, shape
### precond_lt
- methods:
  - zero_mean_mvn_samples
### diag_linear_operator, rootlinearoperator
- methods:
  - zero_mean_mvn_samples



## memo
- we can make mmm firster by using "lax.switch" and "lax.cond" instead of "if" statement.
- trace termは実際には計算されず、自動微分が用いられている、、！→trace termを計算する必要はない
  - [参考](ttps://github.com/cornellius-gp/gpytorch/discussions/1949)