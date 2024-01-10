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
)

import bbmm.utils.test_modules as test_modules
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.added_diag_linear_operator import AddedDiagLinearOp


def test_added_diag_linear_operator():
    N = 100
    noise = 1e-06
    K = test_modules.generate_K(N, seed=0, noise=noise)
    y = jax.random.normal(jax.random.PRNGKey(0), (N,))

    added_diag = AddedDiagLinearOp(DenseLinearOp(K), DiagLinearOp(jnp.full(N, noise)))

    added_diag_y = added_diag.matmul(y)

    added_diag_torch = AddedDiagLinearOperator(
        DenseLinearOperator(torch.from_numpy(np.array(K))),
        DiagLinearOperator(torch.from_numpy(np.full(N, noise))),
    )

    added_diag_y_torch = added_diag_torch._matmul(torch.from_numpy(np.array(y)))

    assert jnp.allclose(added_diag_y, added_diag_y_torch.numpy())
    assert jnp.allclose(added_diag._diagonal(), added_diag_torch._diagonal().numpy())
