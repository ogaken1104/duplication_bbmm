import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

import warnings

warnings.filterwarnings("always")

from test_calc_loss_sin_linearop import calc_loss_sin


from stopro.GP.gp_1D_laplacian import GPmodel1DLaplacian
from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent
from stopro.GP.gp_stokes_3D import GPStokes3D
from stopro.GP.gp_stokes_3D_naive import GPStokes3DNaive

#########


# def test_loss_sin1d_pigp_init_0():
#     project_name = "data"
#     simulation_name = "test_sin1d"
#     init = jnp.array([0.0, 0.0])
#     scale = 1.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 1,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPmodel1DLaplacian,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#     )


# def test_loss_sp_sinu_sparse():
#     project_name = "tests/data"
#     simulation_name = "sinusoidal_direct_sparse"
#     init = None
#     scale = 1.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPSinusoidalWithoutPIndependent,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#     )


# def test_loss_sp_sinu_direct():
#     project_name = "tests/data"
#     simulation_name = "sinusoidal_direct"
#     init = None
#     scale = 1.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPSinusoidalWithoutPIndependent,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#     )


# def test_loss_sp_drag3D_scale5():
#     project_name = "tests/data"
#     simulation_name = "20240123_drag3D_scale5"
#     init = None
#     scale = 5.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3D,
#         use_lazy_matrix=True,
#         matmul_blockwise=False,
#         test_cholesky=True,
#         test_ours=True,
#     )


# def test_loss_sp_drag3D_only_v_11808_rowwise():
#     project_name = "tests/data"
#     simulation_name = "20240207_drag3D_only_v_11808"
#     init = None
#     scale = 10.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3DNaive,
#         use_lazy_matrix=True,
#         matmul_blockwise=False,
#         test_cholesky=False,
#         test_ours=True,
#     )


# def test_loss_sp_drag3D_only_v_11808_blockwise():
#     project_name = "tests/data"
#     simulation_name = "20240207_drag3D_only_v_11808"
#     init = None
#     scale = 5.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3DNaive,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#         test_cholesky=False,
#         test_ours=True,
#     )
# def test_loss_sp_drag3D_only_v_77736_rowwise():
#     project_name = "tests/data"
#     simulation_name = "20240207_drag3D_only_v_77736"
#     init = None
#     scale = 20.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3DNaive,
#         use_lazy_matrix=True,
#         matmul_blockwise=False,
#         test_cholesky=False,
#         test_ours=True,
#     )


def test_loss_sp_drag3D_only_v_23088_rowwise():
    project_name = "tests/data"
    simulation_name = "20240207_drag3D_only_v_23088"
    init = None
    scale = 20.0
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
        gp_class=GPStokes3DNaive,
        use_lazy_matrix=True,
        matmul_blockwise=False,
        test_cholesky=False,
        test_ours=True,
    )


# def test_loss_sp_drag3D_8794():
#     project_name = "tests/data"
#     simulation_name = "20240205_drag3D_8794_points"
#     init = None
#     scale = 5.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3D,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#         test_cholesky=False,
#     )


# def test_loss_sp_drag3D_20796():
#     project_name = "tests/data"
#     simulation_name = "20240205_drag3D_20976_points"
#     init = None
#     scale = 5.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 20,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3D,
#         use_lazy_matrix=True,
#         matmul_blockwise=False,
#         test_cholesky=False,
#     )


# def test_loss_sp_drag3D_less():
#     project_name = "tests/data"
#     simulation_name = "20240202_drag3D_less"
#     init = None
#     scale = 5.0
#     kwargs_setup_loss = {
#         "rank": 0,
#         "n_tridiag": 1,
#         "max_tridiag_iter": 40,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 1,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         test_gpytorch=False,
#         kwargs_setup_loss=kwargs_setup_loss,
#         gp_class=GPStokes3D,
#         use_lazy_matrix=True,
#         matmul_blockwise=True,
#         test_cholesky=True,
#         test_ours=True,
#     )
