import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

import torch
from linear_operator.operators import (
    AddedDiagLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    PsdSumLinearOperator,
    RootLinearOperator,
)

import bbmm.utils.test_modules as test_modules
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.psd_sum_linear_operator import PsdSumLinearOp
from bbmm.operators.root_linear_operator import RootLinearOp


def test_psd_sum_linear_operator():
    N = 100
    noise = 1e-06
    K = test_modules.generate_K(N, seed=0, noise=noise)
    y = jax.random.normal(jax.random.PRNGKey(0), (N,))

    psd_sum = PsdSumLinearOp(RootLinearOp(K), DiagLinearOp(jnp.full(N, noise)))

    psd_sum_y = psd_sum.matmul(y)

    psd_sum_torch = PsdSumLinearOperator(
        RootLinearOperator(torch.from_numpy(np.array(K))),
        DiagLinearOperator(torch.from_numpy(np.full(N, noise))),
    )

    psd_sum_y_torch = psd_sum_torch._matmul(torch.from_numpy(np.array(y)))

    assert jnp.allclose(psd_sum_y, psd_sum_y_torch.numpy())
