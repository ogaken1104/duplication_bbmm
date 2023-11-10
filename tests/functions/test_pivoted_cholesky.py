import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.config import config

from bbmm.functions.pivoted_cholesky import pivoted_cholesky
from bbmm.operators.dense_linear_operator import DenseLinearOp

config.update("jax_enable_x64", True)

import gpytorch


def generate_K(N, seed=0, noise=1e-06):
    """
    generate positive definite symmetric matrix
    """
    K = jax.random.normal(jax.random.PRNGKey(seed), (N, N))
    # # np.random.seed(0)
    # K = np.random.normal((N, N))
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


## gpytorch.pivoted_choleskyとpivoted_choleskyが同じ働きをするか確認するテストコードを書いて
def test_pivoted_cholesky_dense():
    rank = 5
    N = 10
    # 1.1. 正定値行列の場合
    # 1.1.1. 正定値行列の場合
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    A_linear_op = DenseLinearOp(A.numpy())
    L = pivoted_cholesky(A_linear_op, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(A, rank=rank)
    assert jnp.allclose(L, L_torch.numpy())

    ## 1.2. ランダムに生成された行列の場合
    K = generate_K(N)
    K_linear_op = DenseLinearOp(K)
    L = pivoted_cholesky(K_linear_op, max_iter=rank)
    L_torch = gpytorch.pivoted_cholesky(torch.from_numpy(np.array(K)), rank=rank)
    assert jnp.allclose(L, L_torch.numpy())
