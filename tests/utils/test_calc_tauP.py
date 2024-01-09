import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, grad

from jax.config import config

config.update("jax_enable_x64", True)

import warnings

warnings.filterwarnings("always")

from bbmm.utils import test_modules
import bbmm.utils.preconditioner as precond
import bbmm.functions.pivoted_cholesky_jax as pc_jax


from stopro.sub_modules.load_modules import load_params, load_data
from stopro.sub_modules.loss_modules import hessian, logposterior
from stopro.sub_modules.init_modules import get_init, reshape_init
import stopro.GP.gp_1D_naive as gp_1D_naive
from stopro.data_handler.data_handle_module import HdfOperator
from stopro.GP.kernels import define_kernel


def test_calc_tauP():
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive"
    params_main, params_prepare, lbls = load_params(
        f"{project_name}/{simulation_name}/data_input"
    )
    params_model = params_main["model"]
    params_optimization = params_main["optimization"]
    params_plot = params_prepare["plot"]
    vnames = params_prepare["vnames"]
    params_setting = params_prepare["setting"]
    params_generate_training = params_prepare["generate_training"]
    params_generate_test = params_prepare["generate_test"]
    params_kernel_arg = params_prepare["kernel_arg"]

    # prepare initial hyper-parameter
    init = get_init(
        params_model["init_kernel_hyperparameter"],
        params_model["kernel_type"],
        system_type=params_model["system_type"],
    )

    # prepare data
    hdf_operator = HdfOperator(f"{project_name}/{simulation_name}")
    r_test, μ_test, r_train, μ_train, f_train = load_data(lbls, vnames, hdf_operator)
    delta_y_train = jnp.empty(0)
    for i in range(len(r_train)):
        delta_y_train = jnp.append(delta_y_train, f_train[i] - μ_train[i])

    params_model["epsilon"] = 1e-06
    args_predict = r_test, μ_test, r_train, delta_y_train, params_model["epsilon"]
    noise = params_model["epsilon"]
    r_test, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])

    # setup model
    Kernel = define_kernel(params_model)
    gp_model = gp_1D_naive.GPmodel1DNaive(
        Kernel=Kernel,
    )
    gp_model.set_constants(*args_predict)
    loglikelihood, predictor = (
        gp_model.trainingFunction_all,
        gp_model.predictingFunction_all,
    )
    func = jit(logposterior(loglikelihood, params_optimization))
    dfunc = jit(grad(func, 0))
    hess = hessian(func)

    _K = gp_model.trainingK_all(init, r_train)
    K = gp_model.add_eps_to_sigma(_K, noise)
    test_modules.is_positive_definite(K), test_modules.check_cond(K)

    rank = 10
    n_tridiag = 10
    seed = 0
    cg_tolerance = 0.01
    max_tridiag_iter = 20
    min_preconditioning_size = 1
    max_iter_cg = 1000

    r = r_train
    delta_y = delta_y_train
    zs = jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag))

    ## explicit way
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        _K, rank=rank, noise=noise, min_preconditioning_size=min_preconditioning_size
    )

    def calc_dP(theta):
        _K = gp_model.trainingK_all(theta, r)
        # K = gp_model.add_eps_to_sigma(_K, noise, noise_parameter=None)
        _, precond_lt, _ = precond.setup_preconditioner(
            _K,
            rank=rank,
            noise=noise,
            min_preconditioning_size=min_preconditioning_size,
            func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        )
        return precond_lt.matmul(jnp.eye(len(_K)))

    # PinvdPz = precondition(jnp.transpose(jax.jacfwd(Pz)(init), (2, 0, 1)))
    dP = jnp.transpose(jax.jacfwd(calc_dP)(init), (2, 0, 1))

    trace_P_direct = jnp.sum(
        jnp.diagonal(precondition(dP), axis1=-2, axis2=-1), axis=-1
    )

    left_term = precond_lt.left_term_of_trace()

    def diagonal_dP(theta):
        _K = gp_model.trainingK_all(theta, r)
        # K = gp_model.add_eps_to_sigma(_K, noise, noise_parameter=None)
        _, precond_lt, _ = precond.setup_preconditioner(
            _K,
            rank=rank,
            noise=noise,
            min_preconditioning_size=min_preconditioning_size,
            func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        )
        return precond_lt._diagonal()

    diag_dP = jnp.transpose(jax.jacfwd(diagonal_dP)(init), (1, 0))

    def Poperand(theta, operand):
        _K = gp_model.trainingK_all(theta, r)
        # K = gp_model.add_eps_to_sigma(_K, noise, noise_parameter=None)
        _, precond_lt, _ = precond.setup_preconditioner(
            _K,
            rank=rank,
            noise=noise,
            min_preconditioning_size=min_preconditioning_size,
            func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        )
        return precond_lt.matmul(operand)

    dPL = jnp.transpose(
        jax.jacfwd(Poperand, 0)(init, precond_lt.linear_ops[0].root.array), (2, 0, 1)
    )

    trace_P_woodbury = (
        jnp.sum(diag_dP, axis=-1) - jnp.sum(jnp.multiply(left_term, dPL), axis=(-2, -1))
    ) / noise

    assert np.allclose(trace_P_direct, trace_P_woodbury)
