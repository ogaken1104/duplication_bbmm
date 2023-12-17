import jax
import numpy as np
from jax.config import config

import bbmm.utils.preconditioner as precond
import bbmm.utils.test_modules as test_modules

config.update("jax_enable_x64", True)

### don't swap

import gpytorch
import linear_operator
import torch
from linear_operator.operators import (
    AddedDiagLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
    LinearOperator,
)

################
# We should check three terms here
# precond_residual, precond_lt, precond_logdet_cache
# however, the second one, "precond_lt" is difficult to check because its fuction is related to random number. for this object, we should check when chceking logdet.
################
atol = 1e-4
rtol = 1e-4


def calc_precondition_torch(K_torch, y_torch, noise):
    # gpytorch implementation
    K_linear_op = linear_operator.to_linear_operator(K_torch)
    diag_tensor = torch.ones(len(y_torch), dtype=torch.float64) * noise
    diag_linear_op = DiagLinearOperator(diag_tensor)
    added_diag = AddedDiagLinearOperator(K_linear_op, diag_linear_op)

    (
        preconditioner_torch,
        precond_lt_torch,
        precond_logdet_torch,
    ) = added_diag._preconditioner()
    residual_precond_torch = preconditioner_torch(y_torch).numpy()
    return residual_precond_torch[0], precond_lt_torch, precond_logdet_torch


def test_preconditioner_random_50_rank_15():
    ## prepare array for our implementation and linear_operator for gpytorch implementation
    N = 50
    rank = 15
    noise = 1e-06
    min_preconditioning_size = 1
    linear_operator.settings.max_preconditioner_size._set_value(rank)
    linear_operator.settings.min_preconditioning_size._set_value(
        min_preconditioning_size
    )
    K = test_modules.generate_K(N, noise=0.0)
    K_torch = torch.from_numpy(np.array(K))
    K += noise * np.eye(N)
    print(test_modules.is_positive_definite(K))
    y = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N,))
    y_torch = torch.from_numpy(np.array(y))

    ## our impementation
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )
    residual_precond = precondition(y)

    (
        residual_precond_torch,
        precond_lt_torch,
        precond_logdet_torch,
    ) = calc_precondition_torch(K_torch, y_torch, noise)

    ## somehow relative error becomes very large (1e-06), when running as script. In jupyter notebook, its relative error is around 1e-14
    print(test_modules.rel_error(residual_precond, residual_precond_torch))
    assert np.allclose(residual_precond, residual_precond_torch, atol=atol, rtol=rtol)
    assert np.allclose(precond_logdet_cache, precond_logdet_torch, atol=atol, rtol=rtol)


def test_preconditioner_random_2000_rank_400():
    ## prepare array for our implementation and linear_operator for gpytorch implementation
    N = 2000
    rank = 100
    noise = 1e-06
    min_preconditioning_size = 1
    linear_operator.settings.max_preconditioner_size._set_value(rank)
    linear_operator.settings.min_preconditioning_size._set_value(
        min_preconditioning_size
    )
    K = test_modules.generate_K(N, noise=0.0)
    K_torch = torch.from_numpy(np.array(K))
    K += noise * np.eye(N)
    print(test_modules.is_positive_definite(K))
    y = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N,))
    y_torch = torch.from_numpy(np.array(y))

    ## our impementation
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )
    residual_precond = precondition(y)

    (
        residual_precond_torch,
        precond_lt_torch,
        precond_logdet_torch,
    ) = calc_precondition_torch(K_torch, y_torch, noise)

    ## somehow relative error becomes very large, when running as script. In jupyter notebook, its relative error is around 1e-14
    # print(test_modules.rel_error(residual_precond, residual_precond_torch))
    # assert np.allclose(residual_precond, residual_precond_torch, atol=atol, rtol=rtol)
    # assert np.allclose(precond_logdet_cache, precond_logdet_torch, atol=atol, rtol=rtol)
    rerr = test_modules.rel_error(residual_precond, residual_precond_torch)
    print(rerr)
    assert rerr < rtol
    precond_logdet_torch = precond_logdet_torch.numpy()
    assert (
        abs((precond_logdet_cache - precond_logdet_torch) / precond_logdet_torch) < rtol
    )


def test_preconditioner_sp_init():
    ## prepare array for our implementation and linear_operator for gpytorch implementation
    rank = 40
    noise = 1e-06
    min_preconditioning_size = 1
    linear_operator.settings.max_preconditioner_size._set_value(rank)
    linear_operator.settings.min_preconditioning_size._set_value(
        min_preconditioning_size
    )
    K = np.load("tests/data/cov_sinusoidal_direct.npy")
    K_torch = torch.from_numpy(np.array(K))
    K += noise * np.eye(len(K))
    print(test_modules.is_positive_definite(K))
    y = jax.random.normal(key=jax.random.PRNGKey(0), shape=(len(K),))
    y_torch = torch.from_numpy(np.array(y))

    ## our impementation
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )
    residual_precond = precondition(y)

    (
        residual_precond_torch,
        precond_lt_torch,
        precond_logdet_torch,
    ) = calc_precondition_torch(K_torch, y_torch, noise)

    # print(test_modules.rel_error(residual_precond, residual_precond_torch))
    # assert np.allclose(residual_precond, residual_precond_torch, atol=atol, rtol=rtol)
    # assert np.allclose(precond_logdet_cache, precond_logdet_torch, atol=atol, rtol=rtol)
    rerr = test_modules.rel_error(residual_precond, residual_precond_torch)
    print(rerr)
    assert rerr < rtol
    precond_logdet_torch = precond_logdet_torch.numpy()
    assert (
        abs((precond_logdet_cache - precond_logdet_torch) / precond_logdet_torch) < rtol
    )
