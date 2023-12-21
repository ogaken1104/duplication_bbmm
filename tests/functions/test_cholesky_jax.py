import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.cholesky_jax import cholesky_jax
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.utils import test_modules

config.update("jax_enable_x64", True)

rtol = 1e-5


def test_cholesky_jax_random_rank_50():
    rank = 50
    N = 1000
    ## 1.2. ランダムに生成された行列の場合
    K = test_modules.generate_K(N)
    assert test_modules.is_positive_definite(K), "K is not positive definite"
    K_linear_op = DenseLinearOp(K)
    L = cholesky_jax(K_linear_op, max_iter=rank)  # compile
    start_time = time.time()
    L = cholesky_jax(K_linear_op, max_iter=rank)
    end_time = time.time()
    print(f"mmm: {end_time-start_time:.2f}")
    L_linalg = jnp.linalg.cholesky(K)[:, :rank]
    assert jnp.allclose(L, L_linalg)

    L = cholesky_jax(K, max_iter=rank)  # compiling
    start_time = time.time()
    L = cholesky_jax(K, max_iter=rank)
    end_time = time.time()
    print(f"not mmm: {end_time-start_time:.2f}")
    assert test_modules.rel_error(L_linalg, L) < rtol, "L and L_linalg are not close"
