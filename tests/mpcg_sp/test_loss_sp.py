import os

import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
#####


import linear_operator

from bbmm.utils import test_modules
from tests.mpcg_for_test import calc_loss

atol = 30


def deep_assert_loss(_K, delta_y):
    rank = 15
    n_tridiag = 10
    max_cg_iterations = 1000
    max_tridiag_iter = 20
    cg_tolerance = 0.01
    min_preconditioning_size = 1
    noise = 1e-06

    linear_operator.settings.cg_tolerance._set_value(cg_tolerance)
    linear_operator.settings.min_preconditioning_size._set_value(
        min_preconditioning_size
    )
    linear_operator.settings.max_cg_iterations._set_value(max_cg_iterations)
    linear_operator.settings.num_trace_samples._set_value(n_tridiag)
    linear_operator.settings.max_lanczos_quadrature_iterations._set_value(
        max_tridiag_iter
    )
    linear_operator.settings.max_preconditioner_size._set_value(rank)
    N = len(_K)
    K = _K + jnp.eye(N) * noise
    test_modules.check_cond(K)

    loss_linalg = calc_loss.calc_loss_linalg(K, delta_y)[0]
    loss_mpcg = calc_loss.calc_loss_mpcg(
        K,
        delta_y,
        n_tridiag,
        seed=0,
        rank=rank,
        cg_tolerance=cg_tolerance,
        max_tridiag_iter=max_tridiag_iter,
        max_cg_iterations=max_cg_iterations,
        min_preconditioning_size=min_preconditioning_size,
        noise=noise,
    )[0]

    print(f"linalg: {loss_linalg}")
    print(f"mpcg: {loss_mpcg}")
    # check if consistent with choleskiy
    assert abs(loss_linalg - loss_mpcg) < atol
    # check if consistent with torch
    loss_torch_inv_quad = calc_loss.calc_loss_torch_inv_quad(_K, delta_y, n_tridiag)[0]
    print(f"torch_inv_quad: {loss_torch_inv_quad}")
    assert abs(loss_linalg - loss_torch_inv_quad) < atol


def test_loss_direct_init():
    _K = np.load("tests/data/cov_sinusoidal_direct.npy")
    delta_y = np.load("tests/data/delta_y_sinusoidal_direct.npy")
    deep_assert_loss(_K, delta_y)


def test_loss_direct_opt():
    _K = np.load("tests/data/cov_sinusoidal_direct_opt.npy")
    delta_y = np.load("tests/data/delta_y_sinusoidal_direct.npy")
    deep_assert_loss(_K, delta_y)
