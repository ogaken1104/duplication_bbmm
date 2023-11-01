import jax
import jax.numpy as jnp
from jax import jit, lax, vmap


def lanczos_tridiag_to_diag(t_mat):
    eigvals, eigvectors = jnp.linalg.eigh(t_mat)
    mask = eigvals >= 0.0
    eigvectors = eigvectors * jnp.expand_dims(mask, axis=-2)

    eigvals = eigvals.at[~mask].set(1.0)
    return eigvals, eigvectors


def to_dense(matrix_shape, eigenvalues, eigenvectors, funcs):
    """
    duplicated from linear_operator.utils.stochastic_lq.py
    https://github.com/cornellius-gp/linear_operator/blob/54962429ab89e2a9e519de6da8853513236b283b/linear_operator/utils/stochastic_lq.py#L4
    """
    results = [
        jnp.zeros(eigenvalues.shape[1:-1], dtype=eigenvalues.dtype) for _ in funcs
    ]
    num_random_probes = eigenvalues.shape[0]
    for j in range(num_random_probes):
        eigenvalues_for_probe = eigenvalues[j]
        eigenvectors_for_probe = eigenvectors[j]
        for i, func in enumerate(funcs):
            eigenvecs_first_component = eigenvectors_for_probe[..., 0, :]
            func_eigenvalues = func(eigenvalues_for_probe)

            dot_products = (eigenvecs_first_component**2 * func_eigenvalues).sum(-1)
            results[i] = (
                results[i] + matrix_shape[-1] / float(num_random_probes) * dot_products
            )

    return results


def calc_logdet(matrix_shape, t_mat, preconditioner):
    eigvals, eigvectors = lanczos_tridiag_to_diag(t_mat)
    (pinvk_logdet,) = to_dense(
        matrix_shape, eigvals, eigvectors, [lambda x: jnp.log(x)]
    )

    try:
        logdet_p = preconditioner._precond_logdet_cache
    except:
        logdet_p = 0.0

    logdet = pinvk_logdet + logdet_p

    return logdet
