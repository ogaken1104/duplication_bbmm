from jax.config import config

from tests.mpcg.calc_three_terms import calc_three_terms

config.update("jax_enable_x64", True)

linear_solve_rel_error, logdet_rel_error, trace_rel_error = calc_three_terms(
    simulation_path="tests/data",
    rank=15,
    min_preconditioning_size=1,
    n_tridiag=20,
    max_tridiag_iter=40,
    max_iter_cg=5000,
    tolerance=0.0001,
    scale=10.0,
)
tol_solve_rel_error = 1e-2
tol_rel_error = 5e-02


def test_linear_solve():
    assert linear_solve_rel_error < tol_solve_rel_error


def test_logdet():
    assert logdet_rel_error < tol_rel_error


def test_trace_rel():
    assert trace_rel_error < tol_rel_error


if __name__ == "__main__":
    pass
