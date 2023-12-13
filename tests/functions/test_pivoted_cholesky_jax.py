import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.pivoted_cholesky_jax import pivoted_cholesky_jax

config.update("jax_enable_x64", True)

import gpytorch


def generate_K(N, seed=0, noise=1e-06):
    """
    generate positive definite symmetric matrix
    """
    # K = jax.random.normal(jax.random.PRNGKey(seed), (N, N))
    np.random.seed(0)
    K = np.random.normal((N, N))
    # K = K @ K.T + 30* jnp.eye(N) + noise*jnp.eye(N)
    # K = jnp.dot(K, K.T) + noise*jnp.eye(N)
    # K = jnp.dot(K, K.T) / N
    K = jnp.dot(K, K.T) / N
    # K += (noise+30)*jnp.eye(N) ## ??
    K += (5) * jnp.eye(N)
    K += (noise) * jnp.eye(N)
    if not is_positive_definite(K):
        raise Exception("K is not positive definite !")
    return K


def is_positive_definite(matrix):
    # 行列の固有値を計算
    eigenvalues = np.linalg.eigvals(matrix)

    # 全ての固有値が正であるかをチェック
    if np.all(eigenvalues > 0):
        return True
    else:
        return False


## gpytorch.pivoted_choleskyとpivoted_cholesky_jaxが同じ働きをするか確認するテストコードを書いて
def test_pivoted_cholesky():
    rank = 5
    N = 10
    # 1.1. 正定値行列の場合
    # 1.1.1. 正定値行列の場合
    A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    L = pivoted_cholesky_jax(A, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(A)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())


def test_pivoted_cholesky_random():
    ## 1.2. ランダムに生成された行列の場合
    rank = 5
    N = 60
    K = generate_K(N)
    start_time = time.time()
    L = pivoted_cholesky_jax(K, max_iter=rank)
    end_time = time.time()
    print(f"random: {end_time-start_time:.2f}")
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(K)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())
