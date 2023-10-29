from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


def pivoted_cholesky_numpy(mat, error_tol=1e-3, return_pivots=None, max_iter=15):
    """
    mat: JAX NumPy array of N x N


    refered to this discussion
    https://colab.research.google.com/drive/1sLNdLi3sI0JKO9ooOsuS6aKFBDCdN3n8?usp=sharing#scrollTo=KE0CTvRmN-nP
    """
    n = mat.shape[-1]
    max_iter = min(max_iter, n)

    d = np.diag(mat).copy()
    orig_error = np.max(d)
    error = np.linalg.norm(d, 1) / orig_error
    pi = np.arange(n)

    L = np.zeros((max_iter, n))

    m = 0
    while m < max_iter and error > error_tol:
        permuted_d = d[pi]
        max_diag_idx = np.argmax(permuted_d[m:])
        max_diag_idx = max_diag_idx + m
        max_diag_val = permuted_d[max_diag_idx]
        i = max_diag_idx

        # swap pi_m and pi_i
        pi[m], pi[i] = pi[i], pi[m]

        pim = pi[m]

        L[m, pim] = np.sqrt(max_diag_val)
        L_mpim = L[m, pim]

        if m + 1 < n:
            row = apply_permutation_numpy(mat, pim, None)
            row = row.flatten()
            pi_i = pi[m + 1 :]

            L_m_new = row[pi_i]

            if m > 0:
                L_prev = L[:m, pi_i]
                update = L[:m, pim]
                prod = np.dot(update, L_prev)
                L_m_new = L_m_new - prod

            L_m = L[m, :]
            L_m_new = L_m_new / L_m[pim]
            L_m[pi_i] = L_m_new

            matrix_diag_current = d[pi_i]
            d[pi_i] = matrix_diag_current - L_m_new**2

            # L[m, :]=L_m
            error = np.linalg.norm(d[pi_i], 1) / orig_error
        m = m + 1

    return L.T


def apply_permutation_numpy(
    matrix,
    left_permutation,
    right_permutation,
):
    # If we don't have a left_permutation vector, we'll just use a slice
    if left_permutation is None:
        left_permutation = np.arange(matrix.shape[-2])
    if right_permutation is None:
        right_permutation = np.arange(matrix.shape[-1])

    def permute_submatrix(matrix, left_permutation, right_permutation):
        # return matrix[
        #     # (*batch_idx, np.expand_dims(left_permutation, -1), np.expand_dims(right_permutation, -2))
        # ]
        return matrix[left_permutation][right_permutation].reshape(
            1, -1
        )  ## maybe cuase errors when batch is not zero

    return np.asarray(permute_submatrix(matrix, left_permutation, right_permutation))
