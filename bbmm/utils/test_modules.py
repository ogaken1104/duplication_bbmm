import time

import jax
import jax.numpy as jnp
import numpy as np


def generate_K(N, seed=0, noise=1e-06):
    """
    generate positive definite symmetric matrix
    """
    K = jax.random.normal(jax.random.PRNGKey(seed), (N, N))
    K = jnp.dot(K, K.T) / N
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


def check_cond(matrix):
    cond_num = jnp.linalg.cond(matrix)
    print(f"{cond_num:.2e}")
    return cond_num


def rel_error(true, pred):
    nonzero_index = jnp.where(true != 0.0)
    true = true[nonzero_index]
    pred = pred[nonzero_index]
    return jnp.mean(jnp.abs((true - pred) / true))


def rel_error_scaler(true, pred):
    if true == 0.0:
        return "truth value is zero"
    return jnp.mean(jnp.abs((true - pred) / true))
