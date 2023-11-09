# simple implementation in numpy
import numpy as np
import torch
from gpytorch.utils.permutation import apply_permutation
from linear_operator.utils.permutation import apply_permutation, inverse_permutation

pp = lambda x: np.array2string(x, precision=4, floatmode="fixed")


def pivoted_cholesky_np_gpt(
    mat: np.matrix, error_tol=1e-3, return_pivots=None, max_iter=15, print_process=False
):
    """
    mat: numpy matrix of N x N

    This is to replicate what is done in GPyTorch verbatim.

    in the discussion, error_tol=1e-06, maxiter=50 but I used values in settings.py in linear_operator repository
    """
    n = mat.shape[-1]
    max_iter = min(max_iter, n)

    d = np.array(np.diag(mat))
    orig_error = np.max(d)
    error = np.linalg.norm(d, 1) / orig_error
    pi = np.arange(n)

    L = np.zeros((max_iter, n))

    m = 0
    while m < max_iter and error > error_tol:
        permuted_d = d[pi]
        if print_process:
            print(f" Permuted Matrix diag: {pp(permuted_d[m:])}")
        max_diag_idx = np.argmax(permuted_d[m:])
        max_diag_idx = max_diag_idx + m
        max_diag_val = permuted_d[max_diag_idx]
        i = max_diag_idx
        if print_process:
            print(f"M {m} Max diag idx {i} Max diag val {pp(max_diag_val)}")

        # swap pi_m and pi_i
        pi[m], pi[i] = pi[i], pi[m]
        pim, pii = pi[m], pi[i]  # easier to type later

        # print(d[pim])
        if print_process:
            print(f"Before L_m {pp(L[m,:])}")
        L[m, pim] = np.sqrt(max_diag_val)
        if print_process:
            print(f"After L_m {pp(L[m,:])}")
        L_mpim = L[m, pim]

        if m + 1 < n:
            # print(pi)
            row = apply_permutation(
                torch.from_numpy(mat), torch.tensor(pim), right_permutation=None
            )  # left permutation just swaps row
            row = row.numpy().flatten()
            # print(f"row shape : {row.shape} val : {row}") # len = 10 for 10 x 10
            pi_i = pi[m + 1 :]

            if print_process:
                print(f"pi_i {pi_i} pi_m {pim}")
            # print(f"pi_i {pi_i}") # length = 9 for 10 x 10 matrix iteration 0
            L_m_new = row[pi_i]  # length = 9
            # print(f"L_m_new.shape {L_m_new.shape}")

            if m > 0:
                # pdb.set_trace()
                L_prev = L[:m, pi_i]
                update = L[:m, pim]
                # print(f"Shapes update {update.shape} L_prev {L_prev.shape} L_m_new {L_m_new.shape}")
                if print_process:
                    print(f"pi_i {pi_i} pi_m {pim}")
                    print(f"{pp(update)} \n {pp(L_prev)}")
                prod = update @ L_prev
                # print(f"Shapes prod {prod.shape}")
                # pdb.set_trace()
                if print_process:
                    print(f"L_m_new(1) {pp(L_m_new)}")
                L_m_new = L_m_new - prod  # np.sum(prod, axis=-1)
                if print_process:
                    print(f"update*Lpred {pp(prod)}")
                    print(f"L_m_new(1.5) {pp(L_m_new)}")

            L_m = L[m, :]
            L_m_new = L_m_new / L_m[pim]
            # print(L_m_new.shape) # 10,9
            # print(L_m.shape)
            # print(L_m[pi_i].shape)
            L_m[pi_i] = L_m_new

            matrix_diag_current = d[pi_i]
            d[pi_i] = matrix_diag_current - L_m_new**2
            if print_process:
                print(f"d {pp(d)}\nL_m_new(2) {pp(L_m_new)}")

            L[m, :] = L_m
            error = np.linalg.norm(d[pi_i], 1) / orig_error
        m = m + 1
        if print_process:
            print("\n\n\n")
    return L.T
