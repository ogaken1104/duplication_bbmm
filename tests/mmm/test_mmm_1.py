from tests.mmm.test_mmm import calc_K_x_right_matrix

tol_rel_error = 1e-08
tol_abs_error = 1e-08


mean_rel_error, mean_abs_error_dKdtheta = calc_K_x_right_matrix()


def test_K_x_right_matrix():
    assert mean_rel_error < tol_rel_error


def test_dKdtheta_x_right_matrix():
    assert mean_abs_error_dKdtheta < tol_abs_error


if __name__ == "__main__":
    pass
