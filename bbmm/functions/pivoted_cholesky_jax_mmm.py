from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np

from bbmm.operators._linear_operator import LinearOp


def pivoted_cholesky_jax_mmm(
    mat: LinearOp, error_tol=1e-3, return_pivots=None, max_iter=15
):
    """
    mat: LinearOp


    refered to this discussion
    https://colab.research.google.com/drive/1sLNdLi3sI0JKO9ooOsuS6aKFBDCdN3n8?usp=sharing#scrollTo=KE0CTvRmN-nP
    """
    n = mat.shape[-1]
    max_iter = min(max_iter, n)

    d = mat._diagonal()
    orig_error = jnp.max(d)
    error = jnp.linalg.norm(d, 1) / orig_error
    pi = jnp.arange(n)

    L = jnp.zeros((max_iter, n))

    m = 0
    while m < max_iter and error > error_tol:
        permuted_d = d[pi]
        max_diag_idx = jnp.argmax(permuted_d[m:])
        max_diag_idx = max_diag_idx + m
        max_diag_val = permuted_d[max_diag_idx]
        i = max_diag_idx

        # swap pi_m and pi_i
        pim = pi[m]
        pi = pi.at[m].set(pi[i])
        pi = pi.at[i].set(pim)

        pim = pi[m]

        L = L.at[m, pim].set(jnp.sqrt(max_diag_val))

        if m + 1 < n:
            row = apply_permutation_jax_mmm(mat, pim, None)
            row = row.flatten()
            pi_i = pi[m + 1 :]

            L_m_new = row[pi_i]

            if m > 0:
                L_prev = L[:m, pi_i]
                update = L[:m, pim]
                prod = jnp.dot(update, L_prev)
                L_m_new = L_m_new - prod

            L_m = L[m, :]
            L_m_new = L_m_new / L_m[pim]
            L_m = L_m.at[pi_i].set(L_m_new)

            matrix_diag_current = d[pi_i]
            d = d.at[pi_i].set(matrix_diag_current - L_m_new**2)

            L = L.at[m, :].set(L_m)  # maybe its possible to make faster here
            error = jnp.linalg.norm(d[pi_i], 1) / orig_error
        m = m + 1

    return L.T


def apply_permutation_jax_mmm(
    matrix,
    left_permutation,
    right_permutation,
):
    # If we don't have a left_permutation vector, we'll just use a slice
    if left_permutation is None:
        left_permutation = jnp.arange(matrix.shape[-2])
    if right_permutation is None:
        right_permutation = jnp.arange(matrix.shape[-1])

    def permute_submatrix(matrix, left_permutation, right_permutation):
        return matrix.__getitem__(
            (
                jnp.expand_dims(left_permutation, -1),
                # np.expand_dims(right_permutation, -2), ## right permutation is not used at this point, for easier implementation of linear operator
            )
        )
        ## maybe cuase errors when batch is not zero

    return permute_submatrix(matrix, left_permutation, right_permutation)


#####
#####Implementation using numpy is keeped here
#####
# def pivoted_cholesky(mat: LinearOp, error_tol=1e-3, return_pivots=None, max_iter=15):
#     """
#     mat: JAX NumPy array of N x N


#     refered to this discussion
#     https://colab.research.google.com/drive/1sLNdLi3sI0JKO9ooOsuS6aKFBDCdN3n8?usp=sharing#scrollTo=KE0CTvRmN-nP
#     """
#     n = mat.shape[-1]
#     max_iter = min(max_iter, n)

#     d = np.array(mat._diagonal())
#     orig_error = np.max(d)
#     error = np.linalg.norm(d, 1) / orig_error
#     pi = np.arange(n)

#     L = np.zeros((max_iter, n))

#     m = 0
#     while m < max_iter and error > error_tol:
#         permuted_d = d[pi]
#         max_diag_idx = np.argmax(permuted_d[m:])
#         max_diag_idx = max_diag_idx + m
#         max_diag_val = permuted_d[max_diag_idx]
#         i = max_diag_idx

#         # swap pi_m and pi_i
#         pi[m], pi[i] = pi[i], pi[m]

#         pim = pi[m]

#         L[m, pim] = np.sqrt(max_diag_val)
#         L_mpim = L[m, pim]

#         if m + 1 < n:
#             row = apply_permutation(mat, pim, None)
#             row = row.flatten()
#             pi_i = pi[m + 1 :]

#             L_m_new = row[pi_i]

#             if m > 0:
#                 L_prev = L[:m, pi_i]
#                 update = L[:m, pim]
#                 prod = np.dot(update, L_prev)
#                 L_m_new = L_m_new - prod

#             L_m = L[m, :]
#             L_m_new = L_m_new / L_m[pim]
#             L_m[pi_i] = L_m_new

#             matrix_diag_current = d[pi_i]
#             d[pi_i] = matrix_diag_current - L_m_new**2

#             # L[m, :]=L_m
#             error = np.linalg.norm(d[pi_i], 1) / orig_error
#         m = m + 1

#     return L.T


# def apply_permutation(
#     matrix,
#     left_permutation,
#     right_permutation,
# ):
#     # If we don't have a left_permutation vector, we'll just use a slice
#     if left_permutation is None:
#         left_permutation = np.arange(matrix.shape[-2])
#     if right_permutation is None:
#         right_permutation = np.arange(matrix.shape[-1])

#     def permute_submatrix(matrix, left_permutation, right_permutation):
#         # return matrix[
#         #     # (*batch_idx, np.expand_dims(left_permutation, -1), np.expand_dims(right_permutation, -2))
#         # ]
#         return matrix.__getitem__(
#             (
#                 np.expand_dims(left_permutation, -1),
#                 # np.expand_dims(right_permutation, -2), ## right permutation is not used at this point, for easier implementation of linear operator
#             )
#         )
#         ## maybe cuase errors when batch is not zero

#     return np.asarray(permute_submatrix(matrix, left_permutation, right_permutation))
