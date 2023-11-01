from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from pivoted_cholesky import pivoted_cholesky_numpy


class Preconditioner:
    def __init__(
        self,
        matrix: jnp.array,
        rank: int = 15,
        noise: float = 1e-06,
        min_preconditioning_size: int = 2000,
    ):
        if len(matrix) < min_preconditioning_size:
            self.precondition = lambda residual: residual
        else:
            self.precondition = self._precondition
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
        self.q_cache, self.r_cache = jnp.linalg.qr(D_L)
        self.q_cache = self.q_cache[:n, :]
        self.noise = noise

        ## for logdet
        logdet = (
            jnp.sum(jnp.log(jnp.abs(jnp.diagonal(self.r_cache, axis1=0, axis2=1)))) * 2
        )
        logdet = logdet + (n - k) * jnp.log(self.noise)
        self._precond_logdet_cache = logdet

    def _precondition(self, residual: jnp.array):
        qqt = jnp.matmul(self.q_cache, jnp.matmul(self.q_cache.T, residual))
        preconditioned_residual = 1.0 / self.noise * (residual - qqt)
        return preconditioned_residual
