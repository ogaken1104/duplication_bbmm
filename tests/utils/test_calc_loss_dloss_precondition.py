import jax.numpy as jnp
from test_calc_loss_sin_linearop import calc_loss_sin


def test_loss_sin1d_10_init_0():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 10,
        "n_tridiag": 10,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=False,
        use_lazy_matrix=False,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_10_init_0_lazy():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 10,
        "n_tridiag": 10,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=False,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_10_init_0_lazy_blockwise():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 10,
        "n_tridiag": 10,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        matmul_blockwise=True,
        kwargs_setup_loss=kwargs_setup_loss,
    )


def test_loss_sin1d_1000_x100_init_2_lazy():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive_y_1000"
    init = jnp.array([2.0, 2.0])
    scale = 100.0
    kwargs_setup_loss = {
        "rank": 50,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 0.01,
        "max_iter_cg": 2000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        kwargs_setup_loss=kwargs_setup_loss,
        test_gpytorch=False,
        use_lazy_matrix=True,
    )


def test_loss_sin1d_1000_x100_init_2_lazy_blockwise():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive_y_1000"
    init = jnp.array([2.0, 2.0])
    scale = 100.0
    kwargs_setup_loss = {
        "rank": 50,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 0.01,
        "max_iter_cg": 2000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        kwargs_setup_loss=kwargs_setup_loss,
        test_gpytorch=False,
        use_lazy_matrix=True,
        matmul_blockwise=True,
    )
