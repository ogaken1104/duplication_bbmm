import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

##
import linear_operator
import torch
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator

import bbmm.utils.calc_logdet as calc_logdet
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond


def calc_loss_mpcg(
    K,
    delta_y,
    n_tridiag,
    seed=0,
    rank=15,
    return_linear_solve=False,
    return_logdet=False,
    cg_tolerance=1.0,
    max_cg_iterations=1000,
    max_tridiag_iter=20,
    min_preconditioning_size=2000,
    return_loss=True,
    noise=1e-06,
):
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )
    zs = jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag))
    if precondition:
        zs = precond_lt.zero_mean_mvn_samples(n_tridiag, seed=seed)
    # else:
    #     zs = jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag))
    zs_norm = jnp.linalg.norm(zs, axis=0)
    zs = zs / zs_norm
    rhs = jnp.concatenate([zs, delta_y.reshape(-1, 1)], axis=1)
    Kinvy, j, t_mat = cg.mpcg_bbmm(
        K,
        rhs,
        precondition=precondition,
        print_process=False,
        tolerance=cg_tolerance,
        n_tridiag=n_tridiag,
        max_tridiag_iter=max_tridiag_iter,
        max_iter_cg=max_cg_iterations,
    )

    logdet = calc_logdet.calc_logdet(K.shape, t_mat, precond_logdet_cache)
    yKy = jnp.dot(delta_y, Kinvy[:, -1])
    loss = ((yKy + logdet) / 2 + len(delta_y) / 2 * jnp.log(jnp.pi * 2)) / len(K)

    return_list = []
    if return_loss:
        return_list.append(loss)
    if return_linear_solve:
        return_list.append(Kinvy)
    if return_logdet:
        return_list.append(logdet)
    return return_list


def calc_loss_linalg(
    K, delta_y, return_linear_solve=False, return_logdet=False, return_loss=True
):
    n = len(delta_y)
    L = jnp.linalg.cholesky(K)
    Kinvy = jnp.linalg.solve(L.T, jnp.linalg.solve(L, delta_y))
    v = jnp.linalg.solve(L, delta_y)
    logdet = jnp.sum(jnp.log(jnp.diag(L))) * 2
    loss = 0.5 * (jnp.dot(v, v) + logdet + n * jnp.log(2.0 * jnp.pi)) / len(K)
    return_list = []
    if return_loss:
        return_list.append(loss)
    if return_linear_solve:
        return_list.append(Kinvy)
    if return_logdet:
        return_list.append(logdet)
    return return_list


def calc_loss_torch_inv_quad(
    _K,
    delta_y,
    return_linear_solve=False,
    return_logdet=False,
    return_loss=True,
    seed=0,
    noise=1e-06,
):
    n = len(delta_y)
    K_torch = torch.from_numpy(np.array(_K))
    K_linear_op = linear_operator.to_linear_operator(K_torch)
    diag_tensor = torch.ones(n, dtype=torch.float64) * noise
    diag_linear_op = DiagLinearOperator(diag_tensor)
    added_diag = AddedDiagLinearOperator(K_linear_op, diag_linear_op)

    rhs_torch = torch.from_numpy(np.array(delta_y.reshape(-1, 1)))
    torch.manual_seed(seed)
    inv_quad, logdet_torch = added_diag.inv_quad_logdet(
        inv_quad_rhs=rhs_torch, logdet=True
    )
    yKinvy = inv_quad.numpy()
    logdet = logdet_torch.numpy()
    loss = 0.5 * (yKinvy + logdet + n * jnp.log(2.0 * jnp.pi)) / len(_K)
    return_list = []
    if return_loss:
        return_list.append(loss)
    if return_linear_solve:
        return_list.append(yKinvy)
    if return_logdet:
        return_list.append(logdet)
    return return_list
