from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from pivoted_cholesky import pivoted_cholesky_numpy


class Preconditioner:
    def __init__(self, matrix: jnp.array, rank: int = 15, noise: float = 1e-06):
        if rank is None:
            piv_chol_self = pivoted_cholesky_numpy(matrix)
        else:
            piv_chol_self = pivoted_cholesky_numpy(matrix, max_iter=rank)
        n, k = piv_chol_self.shape
        # print(f'n: {n} k: {k}')
        eye = jnp.eye(k)
        noise_matrix = eye * jnp.sqrt(noise)
        # [D^{-1/2}; L]
        D_L = jnp.concatenate([piv_chol_self, noise_matrix], axis=-2)
        q_cache, r_cache = jnp.linalg.qr(D_L)
        self.q_cache = q_cache[:n, :]
        self.noise = noise

    def precondition(self, residual: jnp.array):
        qqt = jnp.matmul(self.q_cache, jnp.matmul(self.q_cache.T, residual))
        preconditioned_residual = 1.0 / self.noise * (residual - qqt)
        return preconditioned_residual
