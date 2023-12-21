import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.config import config

config.update("jax_enable_x64", True)


import bbmm.utils.calc_trace as calc_trace
import bbmm.utils.test_modules as test_modules

rtol = 0.1


def test_trace_random_50():
    N = 50
    dK_scale = 10
    n_tridiag = 100

    zs = jax.random.normal(jax.random.PRNGKey(0), (N, n_tridiag))
    zs_norms = jnp.linalg.norm(zs, axis=0, keepdims=True)
    zs = zs / zs_norms

    K = test_modules.generate_K(N, seed=0)
    test_modules.check_cholesky_inverse_accuracy(K)
    L = jnp.linalg.cholesky(K)
    v = jnp.linalg.solve(L, zs)
    Kinvz_linalg = jnp.linalg.solve(L.T, v)
    I = jnp.eye(N)
    dKdtheta = test_modules.generate_K(N, seed=1) * dK_scale
    Kinv_dKdtheta = jnp.linalg.solve(L.T, jnp.linalg.solve(L, dKdtheta))

    trace_mpcg = calc_trace.calc_trace(Kinvz_linalg, dKdtheta, zs, n_tridiag=n_tridiag)
    trace_linalg = jnp.sum(jnp.diag(Kinv_dKdtheta))

    print(f"trace_mpcg: {trace_mpcg:.3e}")
    print(f"trace_linalg: {trace_linalg:.3e}\n")

    assert test_modules.rel_error_scaler(trace_linalg, trace_mpcg) < rtol
