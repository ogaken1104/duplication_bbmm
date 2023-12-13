import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.pivoted_cholesky_numpy_mmm import pivoted_cholesky_numpy_mmm
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.utils.test_modules import generate_K

config.update("jax_enable_x64", True)


import gpytorch


## gpytorch.pivoted_choleskyとpivoted_choleskyが同じ働きをするか確認するテストコードを書いて
def test_pivoted_cholesky_dense():
    rank = 5
    # 1.1. 正定値行列の場合
    # 1.1.1. 正定値行列の場合
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    A_linear_op = DenseLinearOp(A.numpy())
    L = pivoted_cholesky_numpy_mmm(A_linear_op, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(A, rank=rank)
    assert jnp.allclose(L, L_torch.numpy())


def test_pivoted_cholesky_dense_random():
    ## 1.2. ランダムに生成された行列の場合
    rank = 50
    N = 1000
    K = generate_K(N)
    K_linear_op = DenseLinearOp(K)
    start_time = time.time()
    L = pivoted_cholesky_numpy_mmm(K_linear_op, max_iter=rank)
    end_time = time.time()
    print(f"random: {end_time-start_time:.2f}")
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(K)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())
