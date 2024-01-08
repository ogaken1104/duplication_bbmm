import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.config import config

config.update("jax_enable_x64", True)

import gpytorch
import linear_operator
import torch
from linear_operator.operators import (
    AddedDiagLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
    LinearOperator,
)

import bbmm.functions.pivoted_cholesky_numpy as pc_numpy
import bbmm.functions.pivoted_cholesky_ref as pc_ref  # to use this script we need "torch", please comment out if not needed.
import bbmm.utils.calc_logdet as calc_logdet
import bbmm.utils.calc_trace as calc_trace
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond
import bbmm.utils.test_modules as test_modules

rtol = 1e-4


def calc_dummy_t_mat(size):
    def tridiagonal_matrix(size):
        # 三重対角行列を生成
        matrix = np.zeros((size, size))

        # 対角線の要素を設定
        for i in range(size):
            matrix[i, i] = 2.0  # 対角線上の要素

        # 上下の対角線の要素を設定
        for i in range(size - 1):
            matrix[i, i + 1] = 1.0  # 上の対角線の要素
            matrix[i + 1, i] = 1.0  # 下の対角線の要素

        return matrix

    result_matrix = tridiagonal_matrix(size)
    return result_matrix


def generate_random_tridiagonal_arrays(m, n):
    result_arrays = []

    for _ in range(m):
        # 三重対角行列を生成（ランダムな値）
        tridiagonal_matrix = np.random.rand(n, n)

        # 三重対角行列の対角線およびその隣の対角線の要素を残し、他はゼロにする
        tridiagonal_matrix *= np.tri(n, n, 0) + np.tri(n, n, 1) + np.tri(n, n, -1)

        # 行列を対称にする
        symmetric_matrix = (
            tridiagonal_matrix
            + tridiagonal_matrix.T
            - np.diag(tridiagonal_matrix.diagonal())
        )

        result_arrays.append(symmetric_matrix)

    # サイズが m x n x n の多次元配列を生成
    result_array = np.stack(result_arrays)

    return result_array


def assert_logdet(
    N,
    noise,
    rank,
    n_tridiag,
    max_cg_iterations,
    max_tridiag_iter,
    cg_tolerance,
    min_preconditioning_size=2000,
    _K=None,
    seed=0,
):
    # set parameters
    linear_operator.settings.max_cg_iterations._set_value(cg_tolerance)
    linear_operator.settings.min_preconditioning_size._set_value(
        min_preconditioning_size
    )
    linear_operator.settings.max_cg_iterations._set_value(max_cg_iterations)
    linear_operator.settings.num_trace_samples._set_value(n_tridiag)
    linear_operator.settings.max_lanczos_quadrature_iterations._set_value(
        max_tridiag_iter
    )

    # generate covariance matrices
    if _K is not None:
        N = len(_K)
    else:
        _K = test_modules.generate_K(N, noise=0.0)
    K = _K + noise * np.eye(N)
    y = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N, 1))
    zs = jax.random.normal(jax.random.PRNGKey(1), (N, n_tridiag))
    rhs = jnp.concatenate([zs, y], axis=1)

    K_torch = torch.from_numpy(np.array(_K))
    K_linear_op = linear_operator.to_linear_operator(K_torch)
    diag_tensor = torch.ones(N, dtype=torch.float64) * noise
    diag_linear_op = DiagLinearOperator(diag_tensor)
    added_diag = AddedDiagLinearOperator(K_linear_op, diag_linear_op)
    rhs_torch = torch.from_numpy(np.array(rhs))

    # check precond_logdet
    # our implementation
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )
    if precondition:
        zs = precond_lt.zero_mean_mvn_samples(n_tridiag, seed=seed)
    else:
        zs = jax.random.normal(jax.random.PRNGKey(seed), (len(K), n_tridiag))
    zs_norm = jnp.linalg.norm(zs, axis=0)
    zs = zs / zs_norm

    # gpytorch
    preconditioner_torch, _, precond_logdet_torch = added_diag._preconditioner()
    if precondition:
        rerr = test_modules.rel_error_scaler(precond_logdet_torch, precond_logdet_cache)
        print(f"precond_logdet: {rerr:.2e}\n")
        assert rerr < rtol

    # check logdet
    # our implementation
    Kinvy, j, t_mat = cg.mpcg_bbmm(
        K,
        rhs,
        precondition=precondition,
        print_process=False,
        tolerance=1,
        n_tridiag=n_tridiag,
        max_iter_cg=max_cg_iterations,
    )
    logdet = calc_logdet.calc_logdet(K.shape, t_mat, precond_logdet_cache)
    if precond_logdet_cache:
        logdet -= precond_logdet_cache
    # gpytorch
    Kinvy_torch, t_mat_torch = added_diag._solve(
        rhs_torch,
        preconditioner=preconditioner_torch,
        num_tridiag=n_tridiag,
    )
    eval_torch, evec_torch = linear_operator.utils.lanczos.lanczos_tridiag_to_diag(
        t_mat_torch
    )
    slq = linear_operator.utils.stochastic_lq.StochasticLQ()
    (logdet_term,) = slq.to_dense(
        added_diag.matrix_shape, eval_torch, evec_torch, [lambda x: x.log()]
    )

    rerr = test_modules.rel_error_scaler(np.array(logdet_term), logdet)
    print(f"logdet torch: {np.array(logdet_term)}")
    print(f"logdet our imp.: {logdet}")
    L = jnp.linalg.cholesky(K)
    logdet_linalg = jnp.sum(jnp.log(jnp.diag(L))) * 2
    print(f"logdet linalg: {logdet_linalg}\n")
    print(f"rerr w.r.t torch: {rerr:.2e}")
    rerr_linalg = test_modules.rel_error_scaler(logdet_linalg, logdet)
    print(f"rerr w.r.t linalg: {rerr_linalg:.2e}\n\n")
    assert rerr < rtol


def test_logdet_dummy():
    n_tridiag = 10
    n_tridiag_iter = 20
    N = 100
    # t_mat = calc_dummy_t_mat(n_tridiag)
    t_mat = generate_random_tridiagonal_arrays(n_tridiag, n_tridiag_iter)
    # t_mat_for_ours = t_mat.transpose(1, 2, 0)
    t_mat_torch = torch.from_numpy(np.array(np.expand_dims(t_mat, 1)))

    # our impelementation
    logdet = calc_logdet.calc_logdet((N, N), t_mat, None)
    eval_torch, evec_torch = linear_operator.utils.lanczos.lanczos_tridiag_to_diag(
        t_mat_torch
    )
    # gpytorch
    slq = linear_operator.utils.stochastic_lq.StochasticLQ()
    (logdet_term,) = slq.to_dense((N, N), eval_torch, evec_torch, [lambda x: x.log()])
    rerr = test_modules.rel_error_scaler(np.array(logdet_term), logdet)
    print(f"logdet: {rerr:.2e}\n")
    assert rerr < rtol


def test_logdet_random_50():
    N = 50
    noise = 1e-06
    rank = 15
    n_tridiag = 10
    max_cg_iterations = 1000
    max_tridiag_iter = 20
    cg_tolerance = 1.0
    assert_logdet(
        N, noise, rank, n_tridiag, max_cg_iterations, max_tridiag_iter, cg_tolerance
    )


def test_logdet_random_1000():
    N = 1000
    noise = 1e-06
    rank = 15
    n_tridiag = 10
    max_cg_iterations = 1000
    max_tridiag_iter = 20
    cg_tolerance = 0.01
    assert_logdet(
        N, noise, rank, n_tridiag, max_cg_iterations, max_tridiag_iter, cg_tolerance
    )


## bbmm seems difficult to converge
# def test_logdet_sp():
#     N = None
#     noise = 1e-06
#     rank = 15
#     n_tridiag = 20
#     max_cg_iterations = 1000
#     max_tridiag_iter = 40
#     cg_tolerance = 0.01
#     _K = np.load("tests/data/cov_sinusoidal_direct.npy")
#     assert_logdet(
#         N,
#         noise,
#         rank,
#         n_tridiag,
#         max_cg_iterations,
#         max_tridiag_iter,
#         cg_tolerance,
#         _K=_K,
#     )
