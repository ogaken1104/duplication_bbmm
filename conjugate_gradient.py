from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

# class IdentityPreconditioner:
#     def precondition(self, residual: jnp.array):
#         return residual


def precondition_identity(residual: jnp.array):
    return residual


def mpcg_bbmm(
    A,
    rhs,
    precondition=None,
    max_iter_cg=1000,
    tolerance=1,
    print_process=False,
    eps=1e-10,
    n_tridiag=10,
    n_tridiag_iter=20,
):
    """
    function to implement modified preconditiond conjugate gradient (mPCG) in Algorithm 2, Appendix A.

    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Solve

    Args:
        - A: matrix represents lhs
        - rhs: matrix represents rhs, which is comprised of [probe_vectors, y]
        - max_iter_cg: maximum number of iteration for conjugate gradient
        - tolerance: l2 norm tolerance for residual
        - print_process: if print optimization detail at each step
        - eps: norm less than this is considered to be 0
        - n_tridiag: number of the first columns of rhs to be tridiagonalized
        - n_tridiag_iter: maximum size of the tridiagonalization matrix

    Returns:
        - u: result of solving the eq. the leftmost columns is K^{-1}y
        - t_mat: corresponding tridiagonal matrices (if n_tridiag > 0)
    """
    if not precondition:
        precondition = precondition_identity

    num_rows = rhs.shape[-2]
    n_tridiag_iter = min(n_tridiag_iter, num_rows)
    ### initial setting
    u = jnp.zeros_like(rhs)  ## current solution
    r0 = rhs - jnp.matmul(A, u)  ## current residual

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = jnp.linalg.norm(r0)
    rhs_is_zero = rhs_norm < eps
    if rhs_is_zero:
        rhs_norm = 1.0

    ## for tridiag
    if n_tridiag:
        t_mat = jnp.zeros((n_tridiag_iter, n_tridiag_iter, n_tridiag))
        # alpha_tridiag_is_zero = jnp.empty(n_tridiag)
        alpha_reciprocal = jnp.empty(n_tridiag)
        prev_alpha_reciprocal = jnp.empty_like(alpha_reciprocal)
        prev_beta = jnp.empty_like(alpha_reciprocal)
    update_tridiag = True

    # Let's normalize. We'll un-normalize afterwards
    r0 = r0 / rhs_norm

    z0 = precondition(r0)  ## preconditioned residual
    d = z0  ## search direction for next solution

    @partial(jit, static_argnames=["n_tridiag"])
    def linear_cg_updates(A, d, r0, z0, u, n_tridiag):
        v = jnp.dot(A, d)
        alpha = jnp.matmul(r0.T, z0) / jnp.matmul(d.T, v)
        u = u + jnp.diag(alpha) * d
        r1 = r0 - jnp.diag(alpha) * v

        z1 = precondition(r1)
        beta = jnp.matmul(r1.T, z1) / jnp.matmul(r0.T, z0)
        d = z1 + jnp.diag(beta) * d
        # r0 = r1
        # z0 = z1

        alpha_tridiag = jnp.diag(alpha)[:n_tridiag]
        beta_tridiag = jnp.diag(beta)[:n_tridiag]
        return d, r1, z1, u, alpha_tridiag, beta_tridiag

    for j in range(max_iter_cg):
        d, r0, z0, u, alpha_tridiag, beta_tridiag = linear_cg_updates(
            A, d, r0, z0, u, n_tridiag
        )

        if n_tridiag and j < n_tridiag_iter and update_tridiag:
            ### TODO implement setting coverged alpha_tridiag 0.
            # alpha_tridiag_is_zero = alpha_tridiag == 0
            # alpha_tridiag = alpha_tridiag.at[alpha_tridiag_is_zero].set(1)
            alpha_reciprocal = 1.0 / alpha_tridiag
            # alpha_tridiag = alpha_tridiag.at[alpha_tridiag_is_zero].set(0)

            # print(alpha_reciprocal)
            if j == 0:
                t_mat = t_mat.at[j, j].set(alpha_reciprocal)
            else:
                t_mat = t_mat.at[j, j].set(
                    alpha_reciprocal + prev_beta * prev_alpha_reciprocal
                )
                t_mat = t_mat.at[j, j - 1].set(jnp.sqrt(prev_beta) * alpha_reciprocal)
                t_mat = t_mat.at[j - 1, j].set(t_mat[j, j - 1])
                if jnp.max(t_mat[j - 1, j]) < 1e-06:
                    update_tridiag = False
            last_tridiag_iter = j

            prev_alpha_reciprocal = alpha_reciprocal.copy()
            prev_beta = beta_tridiag.copy()

        r0_norm = jnp.linalg.norm(r0, axis=0)
        r0_norm_mean = jnp.mean(r0_norm)
        # residual_norm.masked_fill_(rhs_is_zero, 0)
        # torch.lt(residual_norm, stop_updating_after, out=has_converged)
        if print_process:
            print(f"j={j} r1norm: {r0_norm_mean}")
        ## judge convergence, in the source of gpytorch, minimum_iteration is set to 10
        if (
            j >= min(10, max_iter_cg - 1)
            and r0_norm_mean < tolerance
            and not (n_tridiag and j < min(n_tridiag_iter, max_iter_cg))
        ):
            if print_process:
                print("converged")
            break
    if n_tridiag:
        return u * rhs_norm, jnp.transpose(
            t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1], (2, 0, 1)
        )
    else:
        return u * rhs_norm


def cg_bbmm(
    A,
    b,
    precondition=None,
    max_iter_cg=1000,
    tolerance=1,
    print_process=False,
    eps=1e-10,
):
    """
    function to check if we can use simple preconditiond conjugate gradient (PCG) in Algorithm 1, Appendix A.
    """
    if not precondition:
        precondition = precondition_identity
    ### initial setting
    u = jnp.zeros_like(b)  ## current solution
    r0 = b - jnp.matmul(A, u)  ## current residual

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = jnp.linalg.norm(r0)
    rhs_is_zero = rhs_norm < eps
    if rhs_is_zero:
        rhs_norm = 1.0

    # Let's normalize. We'll un-normalize afterwards
    r0 = r0 / rhs_norm

    z0 = precondition(r0)  ## preconditioned residual
    d = z0  ## search direction for next solution

    @jit
    def linear_cg_updates(A, d, r0, z0, u):
        v = jnp.dot(A, d)
        alpha = jnp.dot(r0.T, z0) / jnp.dot(d.T, v)
        u = u + alpha * d
        r1 = r0 - alpha * v

        z1 = precondition(r1)
        beta = jnp.dot(r1.T, z1) / jnp.dot(r0.T, z0)
        d = z1 + beta * d
        r0 = r1
        z0 = z1
        return d, r0, z0, u

    for j in range(max_iter_cg):
        d, r0, z0, u = linear_cg_updates(A, d, r0, z0, u)

        r0_norm = jnp.linalg.norm(r0)
        # residual_norm.masked_fill_(rhs_is_zero, 0)
        # torch.lt(residual_norm, stop_updating_after, out=has_converged)
        if print_process:
            print(f"j={j} r1norm: {np.linalg.norm(r0_norm)}")
        ## judge convergence, in the source of gpytorch, minimum_iteration is set to 10
        if j >= min(10, max_iter_cg - 1) and np.mean(r0_norm) < tolerance:
            if print_process:
                print("converged")
            break
    return u * rhs_norm


def bcg_bbmm(
    A,
    rhs,
    precondition=None,
    max_iter_cg=1000,
    tolerance=1,
    print_process=False,
    eps=1e-10,
):
    """
    function to chekck if we can implement batched preconditioned conjuaget gradient Algorithm 2 except for calculating T.
    """
    if not precondition:
        preconditioner = precondition_identity
    ### initial setting
    u = jnp.zeros_like(rhs)  ## current solution
    r0 = rhs - jnp.matmul(A, u)  ## current residual

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = jnp.linalg.norm(r0)
    rhs_is_zero = rhs_norm < eps
    if rhs_is_zero:
        rhs_norm = 1.0

    # Let's normalize. We'll un-normalize afterwards
    r0 = r0 / rhs_norm

    z0 = precondition(r0)  ## preconditioned residual
    d = z0  ## search direction for next solution

    @jit
    def linear_cg_updates(A, d, r0, z0, u):
        v = jnp.dot(A, d)
        alpha = jnp.matmul(r0.T, z0) / jnp.matmul(d.T, v)
        u = u + jnp.diag(alpha) * d
        r1 = r0 - jnp.diag(alpha) * v

        z1 = precondition(r1)
        beta = jnp.matmul(r1.T, z1) / jnp.matmul(r0.T, z0)
        d = z1 + jnp.diag(beta) * d
        r0 = r1
        z0 = z1
        return d, r0, z0, u

    for j in range(max_iter_cg):
        d, r0, z0, u = linear_cg_updates(A, d, r0, z0, u)

        r0_norm = jnp.linalg.norm(r0, axis=0)
        # residual_norm.masked_fill_(rhs_is_zero, 0)
        # torch.lt(residual_norm, stop_updating_after, out=has_converged)
        if print_process:
            print(f"j={j} r1norm: {jnp.mean(r0_norm)}")
        ## judge convergence, in the source of gpytorch, minimum_iteration is set to 10
        if j >= min(10, max_iter_cg - 1) and jnp.mean(r0_norm) < tolerance:
            if print_process:
                print("converged")
            break
    return u * rhs_norm
