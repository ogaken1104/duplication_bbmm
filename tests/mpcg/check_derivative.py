from jax.config import config

from tests.mpcg.calc_derivative import calc_derivative

config.update("jax_enable_x64", True)


def check_derivative():
    grad_solve, grad_t_mat = calc_derivative(
        simulation_path="tests/data",
        rank=50,
        min_preconditioning_size=1,
        n_tridiag=10,
        max_iter_cg=5000,
        tolerance=0.01,
        scale=10.0,
    )
