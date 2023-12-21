import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.pivoted_cholesky_jax import pivoted_cholesky_jax
from bbmm.utils.test_modules import generate_K

config.update("jax_enable_x64", True)

import gpytorch


## gpytorch.pivoted_choleskyとpivoted_cholesky_jaxが同じ働きをするか確認するテストコードを書いて
def test_pivoted_cholesky():
    rank = 5
    # 1.1. 正定値行列の場合
    # 1.1.1. 正定値行列の場合
    A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    L = pivoted_cholesky_jax(A, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(A)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())


def test_pivoted_cholesky_random():
    ## 1.2. ランダムに生成された行列の場合
    rank = 50
    N = 1000
    K = generate_K(N)
    L = pivoted_cholesky_jax(K, max_iter=rank)  # compile
    start_time = time.time()
    L = pivoted_cholesky_jax(K, max_iter=rank)
    end_time = time.time()
    print(f"random: {end_time-start_time:.2f}")
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(K)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy(), rtol=1e-2)
