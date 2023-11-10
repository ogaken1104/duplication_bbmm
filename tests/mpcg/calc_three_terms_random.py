import importlib
import time

import cmocean as cmo
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import stopro.GP.gp_sinusoidal_independent as gp_sinusoidal_independent
from jax import grad, jit, lax, vmap
from jax.config import config
from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_handler.data_handle_module import *
from stopro.data_preparer.data_preparer import DataPreparer
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

import bbmm.functions.pivoted_cholesky_numpy as pc
import bbmm.utils.calc_logdet as calc_logdet
import bbmm.utils.calc_trace as calc_trace
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond

config.update("jax_enable_x64", True)


def is_positive_definite(matrix):
    # 行列の固有値を計算
    eigenvalues = np.linalg.eigvals(matrix)

    # 全ての固有値が正であるかをチェック
    if np.all(eigenvalues > 0):
        return True
    else:
        return False


def rel_error(true, pred):
    true_max = np.max(true)
    zero_threshold = (
        true_max * 1e-7
    )  # ignore the data that test value is smaller than 1e-7
    index = np.where(abs(true) > zero_threshold)
    if np.all(abs(true) <= zero_threshold):
        rel_error = 0.0
        return rel_error
    true2 = true[index]
    pred2 = pred[index]
    rel_error = np.abs((true2 - pred2) / true2)
    # print(rel_error)
    return rel_error


def generate_K(N, seed=0, noise=1e-06):
    """
    generate positive definite symmetric matrix
    """
    K = jax.random.normal(jax.random.PRNGKey(seed), (N, N))
    # K = K @ K.T + 30* jnp.eye(N) + noise*jnp.eye(N)
    # K = jnp.dot(K, K.T) + noise*jnp.eye(N)
    # K = jnp.dot(K, K.T) / N
    K = jnp.dot(K, K.T) / N
    # K += (noise+30)*jnp.eye(N) ## ??
    K += (1) * jnp.eye(N)
    K += (noise) * jnp.eye(N)
    if not is_positive_definite(K):
        raise Exception("K is not positive definite !")
    return K


def calc_three_terms_random(
    rank: int = 5,
    min_preconditioning_size: int = 2000,
    n_tridiag: int = 10,
    max_iter_cg: int = 1000,
    tolerance: float = 0.01,
    n_tridiag_iter: int = 20,
):
    N = 100

    ## calc covariance matrix
    K = generate_K(N)

    is_pd = is_positive_definite(K)
    if not is_pd:
        raise ValueError("K is not positive definite")
    cond_num = jnp.linalg.cond(K)
    print(f"condition number of K: {cond_num:.3e}")

    ## calc deriative of covariance matrix
    dKdtheta = generate_K(N, seed=1)

    ## calc linear solve
    time_start_linear_solve = time.time()
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, min_preconditioning_size=min_preconditioning_size
    )
    time_end_precondition = time.time()
    y = jax.random.normal(jax.random.PRNGKey(0), (N, 1))
    if precondition:
        zs = jax.random.multivariate_normal(
            jax.random.PRNGKey(0),
            jnp.zeros(len(y)),
            precond_lt,
            shape=(n_tridiag,),
        ).T
    else:
        zs = jax.random.normal(jax.random.PRNGKey(0), (N, n_tridiag))
    # generate zs deterministically from precond_lt = $LL^T+\sigma^2I$
    # zs = jnp.matmul(jnp.sqrt(precond_lt), zs)
    rhs = jnp.concatenate([zs, y], axis=1)
    time_start_mpcg = time.time()
    Kinvy, j, t_mat = cg.mpcg_bbmm(
        K,
        rhs,
        precondition=precondition,
        print_process=False,
        tolerance=tolerance,
        max_iter_cg=max_iter_cg,
        n_tridiag=n_tridiag,
        n_tridiag_iter=n_tridiag_iter,
    )
    time_end_mpcg = time.time()
    print(
        f"mpcg time: {time_end_precondition - time_start_linear_solve + time_end_mpcg - time_start_mpcg:.3f}"
    )
    print(f"prec time: {time_end_precondition - time_start_linear_solve:.3f}")
    print(f"cg   time: {time_end_mpcg - time_start_mpcg:.3f}")
    print(f"cg   iter: {j}\n")
    L = jnp.linalg.cholesky(K)
    v = jnp.linalg.solve(L, rhs)
    Kinvy_linalg = jnp.linalg.solve(L.T, v)
    # linear_solve_rel_error = jnp.mean((Kinvy[:, -1] - Kinvy_linalg) / Kinvy_linalg)
    linear_solve_rel_error = jnp.mean(rel_error(Kinvy_linalg, Kinvy))

    ## calc by logdet
    logdet = calc_logdet.calc_logdet(K.shape, t_mat, precond_logdet_cache)

    def calc_logdet_linalg(K):
        L = jnp.linalg.cholesky(K)
        return jnp.sum(jnp.log(jnp.diag(L))) * 2

    logdet_linalg = calc_logdet_linalg(K)

    logdet_rel_error = abs((logdet - logdet_linalg) / logdet_linalg)

    ## calc trace terms
    trace_rel_error_list = []
    I = jnp.eye(len(y))
    Kinv = jnp.linalg.solve(L.T, jnp.linalg.solve(L, I))
    if precondition:
        trace = calc_trace.calc_trace(
            Kinvy, dKdtheta, precondition(zs), n_tridiag=n_tridiag
        )
    else:
        trace = calc_trace.calc_trace(Kinvy, dKdtheta, zs, n_tridiag=n_tridiag)

    trace_linalg = jnp.sum(jnp.diag(jnp.matmul(Kinv, dKdtheta)))
    print(f"trace: {trace:.3e}")
    print(f"trace_linalg: {trace_linalg:.3e}\n")

    trace_rel_error = abs((trace - trace_linalg) / trace_linalg)

    print(f"linear_solve_rel_error: {linear_solve_rel_error:.3e}")
    print(f"logdet_rel_error: {logdet_rel_error:.3e}")
    print(f"trace_rel_error: {trace_rel_error:.3e}")
    return linear_solve_rel_error, logdet_rel_error, trace_rel_error