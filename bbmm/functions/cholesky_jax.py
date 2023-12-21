from functools import partial
from typing import Optional

# import jax
import jax.numpy as jnp
import numpy as np

from bbmm.operators._linear_operator import LinearOp


def cholesky_jax(mat, max_iter=15):
    """
    mat: JAX NumPy array of N x N
    """
    # コレスキー分解を行うプログラム
    n = mat.shape[-1]
    max_iter = min(max_iter, n)

    # lower triangular matrix
    L = jnp.zeros((n, max_iter))
    if isinstance(mat, LinearOp):
        d = mat._diagonal()
    else:
        d = jnp.diagonal(mat, axis1=-2, axis2=-1)

    L = L.at[0, 0].set(jnp.sqrt(d[0]))
    L = L.at[1:, 0].set(mat[1:, 0] / L[0, 0])
    for i in range(1, max_iter):
        L = L.at[i, i].set(jnp.sqrt(d[i] - jnp.dot(L[i, :i], L[i, :i])))
        L = L.at[i + 1 :, i].set(
            (mat[i + 1 :, i] - jnp.dot(L[i + 1 :, :i], L[i, :i])) / L[i, i]
        )

    return L
