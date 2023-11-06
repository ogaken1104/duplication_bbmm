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
- make mmm_A function
    - ~~for K~~
    - for $\frac{d\hat{K}_{XX}}{d\theta}$
    - analyze the time complexity
- apply optimization stopping for alhpa, beta to obtain better t_mat (cause of low accuracy of log determinat)
    - ask john for sample code
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
