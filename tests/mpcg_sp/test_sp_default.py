import os

from jax.config import config

from tests.mpcg_for_test.calc_three_terms import calc_three_terms

print("\n################################")
print(os.path.basename(__file__))
print("################################")

config.update("jax_enable_x64", True)

linear_solve_rel_error, logdet_rel_error, trace_rel_error = calc_three_terms(
    simulation_path="tests/data/sinusoidal_direct",
    rank=15,
    min_preconditioning_size=1,
    n_tridiag=10,
    max_tridiag_iter=20,
    max_iter_cg=1000,
    tolerance=1,
    scale=1.0,
    length=0.0,
    cov_scale=0.0,
    epsilon=2e-6,
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
