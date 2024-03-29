from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from bbmm.functions.pivoted_cholesky_jax import pivoted_cholesky_jax
from bbmm.functions.pivoted_cholesky_numpy import pivoted_cholesky_numpy
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.psd_sum_linear_operator import PsdSumLinearOp
from bbmm.operators.root_linear_operator import RootLinearOp


def setup_preconditioner(
    matrix: jnp.array,
    rank: int = 15,
    noise: float = 1e-06,
    min_preconditioning_size: int = 2000,
    func_pivoted_cholesky: callable = pivoted_cholesky_numpy,
):
    """
    function to setup preconditioner
    most is dupricated from
    - added_diag_linear_operator.py
    https://github.com/cornellius-gp/linear_operator/blob/54962429ab89e2a9e519de6da8853513236b283b/linear_operator/operators/added_diag_linear_operator.py#L4
    """
    if matrix.shape[0] < min_preconditioning_size:
        return None, None, None
    if rank is None:
        piv_chol_self = func_pivoted_cholesky(matrix)
    else:
        piv_chol_self = func_pivoted_cholesky(matrix, max_iter=rank)
    n, k = piv_chol_self.shape
    # print(f'n: {n} k: {k}')
    eye = jnp.eye(k)
    noise_matrix = eye * jnp.sqrt(noise)
    # [D^{-1/2}; L]
    D_L = jnp.concatenate([piv_chol_self, noise_matrix], axis=-2)
    q_cache, r_cache = jnp.linalg.qr(D_L)
    q_cache = q_cache[:n, :]
    noise = noise

    ## for logdet
    logdet = jnp.sum(jnp.log(jnp.abs(jnp.diagonal(r_cache, axis1=0, axis2=1)))) * 2
    logdet = logdet + (n - k) * jnp.log(noise)
    _precond_logdet_cache = logdet
    # _precond_lt = jnp.matmul(piv_chol_self, piv_chol_self.T) + jnp.eye(n) * noise
    _precond_lt = PsdSumLinearOp(
        RootLinearOp(piv_chol_self), DiagLinearOp(jnp.full(n, noise))
    )

    def precondition(residual: jnp.array):
        qqt = jnp.matmul(q_cache, jnp.matmul(q_cache.T, residual))
        preconditioned_residual = 1.0 / noise * (residual - qqt)
        return preconditioned_residual

    return precondition, _precond_lt, _precond_logdet_cache
