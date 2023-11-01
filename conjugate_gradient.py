from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


class IdentityPreconditioner:
    def precondition(self, residual: jnp.array):
        return residual


def cg_bbmm(
    A,
    b,
    preconditioner=None,
    max_iter_cg=1000,
    tolerance=1,
    print_process=False,
    eps=1e-10,
):
    if not preconditioner:
        preconditioner = IdentityPreconditioner()
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

    z0 = preconditioner.precondition(r0)  ## preconditioned residual
    d = z0  ## search direction for next solution

    @jit
    def linear_cg_updates(A, d, r0, z0, u):
        v = jnp.dot(A, d)
        alpha = jnp.dot(r0.T, z0) / jnp.dot(d.T, v)
        u = u + alpha * d
        r1 = r0 - alpha * v

        z1 = preconditioner.precondition(r1)
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
    preconditioner=None,
    max_iter_cg=1000,
    tolerance=1,
    print_process=False,
    eps=1e-10,
):
    if not preconditioner:
        preconditioner = IdentityPreconditioner()
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

    z0 = preconditioner.precondition(r0)  ## preconditioned residual
    d = z0  ## search direction for next solution

    @jit
    def linear_cg_updates(A, d, r0, z0, u):
        v = jnp.dot(A, d)
        alpha = jnp.matmul(r0.T, z0) / jnp.matmul(d.T, v)
        u = u + jnp.diag(alpha) * d
        r1 = r0 - jnp.diag(alpha) * v

        z1 = preconditioner.precondition(r1)
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
