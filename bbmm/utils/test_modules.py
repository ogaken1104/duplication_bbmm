import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    import linear_operator
except:
    pass


def generate_K(N, seed=0, noise=1e-06):
    """
    generate positive definite symmetric matrix
    """
    K = jax.random.normal(jax.random.PRNGKey(seed), (N, N))
    K = jnp.dot(K, K.T) / N
    K += (noise) * jnp.eye(N)
    if not is_positive_definite(K):
        raise Exception("K is not positive definite !")
    return K


def is_positive_definite(matrix):
    # 行列の固有値を計算
    eigenvalues = np.linalg.eigvals(matrix)

    # 全ての固有値が正であるかをチェック
    if np.all(eigenvalues > 0):
        return True
    else:
        return False


def check_cond(matrix):
    cond_num = jnp.linalg.cond(matrix)
    print(f"cond. #: {cond_num:.2e}")
    return cond_num


def rel_error(true, pred, zero_threshold=None):
    if zero_threshold:
        nonzero_index = np.where(abs(true) > zero_threshold)
    else:
        nonzero_index = jnp.where(true != 0.0)
    true = true[nonzero_index]
    pred = pred[nonzero_index]
    return jnp.mean(jnp.abs((true - pred) / true))


def rel_error_scaler(true, pred):
    if true == 0.0:
        return "truth value is zero"
    return jnp.mean(jnp.abs((true - pred) / true))


def check_cholesky_inverse_accuracy(K):
    L = jnp.linalg.cholesky(K)
    I_reconstructed = jnp.linalg.solve(L.T, jnp.linalg.solve(L, K))
    I_reconstructed /= len(K)
    res = jnp.sum(jnp.linalg.norm(I_reconstructed, axis=0))
    print(res)


def set_linear_operator_settings(kwargs_setup_loss, use_lazy_matrix=True):
    linear_operator.settings.cg_tolerance._set_value(kwargs_setup_loss["max_iter_cg"])
    linear_operator.settings.min_preconditioning_size._set_value(
        kwargs_setup_loss["min_preconditioning_size"]
    )
    linear_operator.settings.cg_tolerance._set_value(kwargs_setup_loss["cg_tolerance"])
    linear_operator.settings.num_trace_samples._set_value(
        kwargs_setup_loss["n_tridiag"]
    )
    linear_operator.settings.max_lanczos_quadrature_iterations._set_value(
        kwargs_setup_loss["max_tridiag_iter"]
    )
    linear_operator.settings.max_preconditioner_size._set_value(
        kwargs_setup_loss["rank"]
    )
    if use_lazy_matrix:
        linear_operator.settings.max_cholesky_size._set_value(1)
