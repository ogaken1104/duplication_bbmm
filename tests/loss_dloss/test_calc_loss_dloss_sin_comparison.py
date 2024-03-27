import jax.numpy as jnp
from test_calc_loss_sin_linearop import calc_loss_sin


def test_loss_sin1d_1000():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive_y_1000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_5000():
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_5000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_10000():
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_10000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_20000():
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_20000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=False,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        test_cholesky=False,
        test_ours=True,
        matmul_blockwise=True,
    )


def test_loss_sin1d_30000():
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_30000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        test_cholesky=False,
    )


# def test_loss_sin1d_60000():
#     project_name = "tests/data"
#     simulation_name = "test_sin1d_naive_y_60000"
#     init = jnp.array([0.0, -1.0])
#     scale = 1.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 1,
#         "max_iter_cg": 1000,
#         "min_preconditioning_size": 2000,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         use_lazy_matrix=True,
#         kwargs_setup_loss=kwargs_setup_loss,
#         test_cholesky=False,
#         test_ours=True,
#         matmul_blockwise=True,
#     )
