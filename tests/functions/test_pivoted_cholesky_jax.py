import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.pivoted_cholesky_jax import pivoted_cholesky_jax
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.utils.test_modules import generate_K

config.update("jax_enable_x64", True)


import gpytorch


## gpytorch.pivoted_choleskyとpivoted_choleskyが同じ働きをするか確認するテストコードを書いて
def _test_pivoted_cholesky_dense():
    rank = 5
    # 1.1. 正定値行列の場合
    # 1.1.1. 正定値行列の場合
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    A_linear_op = DenseLinearOp(A.numpy())
    L = pivoted_cholesky_jax(A_linear_op, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(A, rank=rank)
    assert jnp.allclose(L, L_torch.numpy())


def test_pivoted_cholesky_dense_random():
    ## 1.2. ランダムに生成された行列の場合
    rank = 50
    N = 1000
    K = generate_K(N)
    K_linear_op = DenseLinearOp(K)
    # compile
    L = pivoted_cholesky_jax(K_linear_op, max_iter=rank)
    start_time = time.time()
    # L = pivoted_cholesky_jax_mmm(K_linear_op, max_iter=rank)
    L = pivoted_cholesky_jax(K_linear_op, max_iter=rank)
    end_time = time.time()
    print(f"mmm: {end_time-start_time:.2f}")
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(K)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())

    L = pivoted_cholesky_jax(K, max_iter=rank)  # compile
    start_time = time.time()
    L = pivoted_cholesky_jax(K, max_iter=rank)
    end_time = time.time()
    print(f"not mmm: {end_time-start_time:.2f}")
    assert jnp.allclose(L, L_torch.numpy(), rtol=1e-2)
