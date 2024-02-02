import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

import warnings

warnings.filterwarnings("always")

from test_calc_loss_sin_linearop import calc_loss_sin

import torch

torch.set_default_dtype(torch.float64)

from stopro.GP.gp_1D_laplacian import GPmodel1DLaplacian
from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent

#########


def test_loss_sin1d_pigp_init_0():
    project_name = "data"
    simulation_name = "test_sin1d"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 1,
        "max_iter_cg": 2000,
        "min_preconditioning_size": 1,
    }
    calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=False,
        kwargs_setup_loss=kwargs_setup_loss,
        gp_class=GPmodel1DLaplacian,
    )


def test_loss_sp_sinu_sparse():
    project_name = "tests/data"
    simulation_name = "sinusoidal_direct_sparse"
    init = None
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
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
        test_gpytorch=False,
        kwargs_setup_loss=kwargs_setup_loss,
        gp_class=GPSinusoidalWithoutPIndependent,
    )
