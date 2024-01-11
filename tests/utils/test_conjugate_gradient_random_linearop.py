import os

import jax
import jax.numpy as jnp
from jax.config import config

from bbmm.utils import conjugate_gradient as cg
from bbmm.utils import test_modules
from bbmm.operators.dense_linear_operator import DenseLinearOp

print("\n################################")
print(os.path.basename(__file__))
print("################################")

config.update("jax_enable_x64", True)
rtol = 1e-02


def test_conjugate_gradient_no_tridiag_linearop():
    ## mpcg_bbmmを用いて連立一次方程式を解くことができるかテスト
    K = test_modules.generate_K(100, seed=0, noise=1e-06)
    K_linear_op = DenseLinearOp(K)
    rhs = jax.random.normal(jax.random.PRNGKey(0), (100, 10))

    precondition = None
    tolerance = 0.1
    max_iter_cg = 1000
    n_tridiag = 0

    Kinvy_mpcg, j = cg.mpcg_bbmm(
        K_linear_op,
        rhs,
        precondition=precondition,
        # print_process=True,
        tolerance=tolerance,
        max_iter_cg=max_iter_cg,
        n_tridiag=n_tridiag,
    )
    print(f"iteration: {j}")
    L = jnp.linalg.cholesky(K)
    v = jnp.linalg.solve(L, rhs)
    Kinvy_linalg = jnp.linalg.solve(L.T, v)

    rerr = test_modules.rel_error(rhs.T @ Kinvy_linalg, rhs.T @ Kinvy_mpcg)
    print(f"rerr: {rerr}")
    assert rerr < rtol
