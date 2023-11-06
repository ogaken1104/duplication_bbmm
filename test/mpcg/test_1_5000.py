import os
import sys

sys.path.append(
    os.getcwd(),
)  # pytestはカレントディレクトリをsys.pathに追加しないためカレントディレクトリ上にあるファイルを読み込みたい場合はimport os, import sysと一緒に明記
# # 親ディレクトリのファイルを読み込むための設定
# sys.path.append(os.pardir)
# # 2階層上の親ディレクトリのファイルを読み込むための設定
# sys.path.append(os.pardir + "/..")
print(sys.path)

import importlib

import calc_logdet
import calc_trace
import cmocean as cmo
import conjugate_gradient as cg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pivoted_cholesky as pc
import preconditioner as precond
import stopro.GP.gp_sinusoidal_independent as gp_sinusoidal_independent
from calc_three_terms import calc_three_terms
from jax import grad, jit, lax, vmap
from jax.config import config
from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_handler.data_handle_module import *
from stopro.data_preparer.data_preparer import DataPreparer
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

config.update("jax_enable_x64", True)

linear_solve_rel_error, logdet_rel_error, trace_rel_error = calc_three_terms(
    simulation_path="test/data",
    rank=5,
    min_preconditioning_size=2000,
    n_tridiag=10,
    max_iter_cg=5000,
    tolerance=0.01,
    scale=1.0,
)
tol_rel_error = 1e-02


def test_linear_solve():
    assert linear_solve_rel_error < tol_rel_error


def test_logdet():
    assert logdet_rel_error < tol_rel_error


def test_trace_rel():
    assert trace_rel_error < tol_rel_error


if __name__ == "__main__":
    pass
