import importlib
import os
import time

import cmocean as cmo
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import stopro.GP.gp_sinusoidal_independent as gp_sinusoidal_independent
from jax import grad, jit, lax, vmap
from jax.config import config
from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_handler.data_handle_module import *
from stopro.data_preparer.data_preparer import DataPreparer
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

import bbmm.functions.pivoted_cholesky_numpy as pc
import bbmm.utils.calc_logdet as calc_logdet
import bbmm.utils.calc_trace as calc_trace
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond

config.update("jax_enable_x64", True)


def is_positive_definite(matrix):
    # 行列の固有値を計算
    eigenvalues = np.linalg.eigvals(matrix)

    # 全ての固有値が正であるかをチェック
    if np.all(eigenvalues > 0):
        return True
    else:
        return False


def rel_error(true, pred):
    true_max = np.max(true)
    zero_threshold = (
        true_max * 1e-7
    )  # ignore the data that test value is smaller than 1e-7
    index = np.where(abs(true) > zero_threshold)
    if np.all(abs(true) <= zero_threshold):
        rel_error = 0.0
        return rel_error
    true2 = true[index]
    pred2 = pred[index]
    rel_error = np.abs((true2 - pred2) / true2)
    # print(rel_error)
    return rel_error


def calc_three_terms(
    simulation_path: str = "tests/data",
    rank: int = 5,
    min_preconditioning_size: int = 2000,
    n_tridiag: int = 10,
    max_iter_cg: int = 1000,
    tolerance: float = 0.01,
    scale: float = 1.0,
    max_tridiag_iter: int = 20,
    length: float = 2.3,
    cov_scale: float = 0.0,
):
    params_main, params_prepare, lbls = load_params(f"{simulation_path}/data_input")
    params_model = params_main["model"]
    params_optimization = params_main["optimization"]
    params_plot = params_prepare["plot"]
    vnames = params_prepare["vnames"]
    params_setting = params_prepare["setting"]
    params_generate_training = params_prepare["generate_training"]
    params_generate_test = params_prepare["generate_test"]
    params_kernel_arg = params_prepare["kernel_arg"]

    # # prepare initial hyper-parameter
    # init = get_init(
    #     params_model["init_kernel_hyperparameter"],
    #     params_model["kernel_type"],
    #     system_type=params_model["system_type"],
    # )

    # prepare data
    hdf_operator = HdfOperator(simulation_path)
    _r_test, μ_test, _r_train, μ_train, f_train = load_data(lbls, vnames, hdf_operator)

    ### scale data ###
    # scale = 10
    r_train = [_r * scale for _r in _r_train]
    r_test = [_r * scale for _r in _r_test]
    delta_y_train = jnp.empty(0)
    for i in range(len(r_train)):
        delta_y_train = jnp.append(
            delta_y_train, f_train[i] / scale - μ_train[i]
        )  # scaleの1乗でよい，もう1乗分はf_inferを直接scaleする
    args_predict = r_test, μ_test, r_train, delta_y_train, params_model["epsilon"]

    # setup model
    Kernel = define_kernel(params_model)
    gp_model = gp_sinusoidal_independent.GPSinusoidalWithoutPIndependent(
        use_difp=params_setting["use_difp"],
        use_difu=params_setting["use_difu"],
        lbox=jnp.array([2.5 * scale, 0.0]),
        infer_governing_eqs=params_prepare["generate_test"]["infer_governing_eqs"],
        Kernel=Kernel,
        index_optimize_noise=params_model["index_optimize_noise"],
    )
    gp_model.set_constants(*args_predict)

    cov_scale = cov_scale
    length = length
    init = jnp.array(
        [
            cov_scale,
            length,
            length,
            cov_scale,
            length,
            length,
            cov_scale,
            length,
            length,
        ]
    )

    ## calc covariance matrix
    K = gp_model.trainingK_all(init, r_train)
    K = gp_model.add_eps_to_sigma(K, params_model["epsilon"], noise_parameter=None)

    is_pd = is_positive_definite(K)
    if not is_pd:
        raise ValueError("K is not positive definite")
    cond_num = jnp.linalg.cond(K)
    print(f"condition number of K: {cond_num:.3e}")

    ## calc deriative of covariance matrix
    def calc_trainingK(theta):
        Σ = gp_model.trainingK_all(theta, r_train)
        Σ = gp_model.add_eps_to_sigma(Σ, params_model["epsilon"], noise_parameter=None)
        return Σ

    dKdtheta = jnp.transpose(jax.jacfwd(calc_trainingK)(init), (2, 0, 1))

    ## calc linear solve
    time_start_linear_solve = time.time()
    precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        K, rank=rank, min_preconditioning_size=min_preconditioning_size
    )
    if precondition:
        cond_num = jnp.linalg.cond(precondition(K))
        print(f"condition number of P^{-1}K: {cond_num:.3e}\n")

    # zs = jax.random.normal(jax.random.PRNGKey(0), (len(delta_y_train), n_tridiag))
    # generate zs deterministically from precond_lt = $LL^T+\sigma^2I$
    # zs = jax.random.multivariate_normal(
    #     jax.random.PRNGKey(0),
    #     jnp.zeros(len(delta_y_train)),
    #     precond_lt,
    #     shape=(n_tridiag,),
    # ).T
    # zs = jnp.matmul(jnp.sqrt(precond_lt), zs)
    if precondition:
        zs = precond_lt.zero_mean_mvn_samples(n_tridiag, seed=0)
    else:
        zs = jax.random.normal(jax.random.PRNGKey(0), (len(delta_y_train), n_tridiag))
    # zs_norms = jnp.linalg.norm(zs, axis=0, keepdims=True)
    # zs = zs / zs_norms
    rhs = jnp.concatenate([zs, delta_y_train.reshape(-1, 1)], axis=1)
    time_end_precondition = time.time()
    Kinvy, j, t_mat = cg.mpcg_bbmm(
        K,
        rhs,
        precondition=precondition,
        print_process=False,
        tolerance=tolerance,
        max_iter_cg=max_iter_cg,
        n_tridiag=n_tridiag,
        max_tridiag_iter=max_tridiag_iter,
    )
    time_end_mpcg = time.time()
    print(f"mpcg time: {time_end_mpcg - time_start_linear_solve:.3f}")
    print(f"prec time: {time_end_precondition - time_start_linear_solve:.3f}")
    print(f"cg   time: {time_end_mpcg - time_end_precondition:.3f}")
    print(f"cg   iter: {j}\n")
    L = jnp.linalg.cholesky(K)
    v = jnp.linalg.solve(L, delta_y_train)
    Kinvy_linalg = jnp.linalg.solve(L.T, v)
    # linear_solve_rel_error = jnp.mean((Kinvy[:, -1] - Kinvy_linalg) / Kinvy_linalg)
    linear_solve_rel_error = jnp.mean(rel_error(Kinvy_linalg, Kinvy[:, -1]))

    ## calc by logdet
    logdet = calc_logdet.calc_logdet(K.shape, t_mat, precond_logdet_cache)

    def calc_logdet_linalg(K):
        L = jnp.linalg.cholesky(K)
        return jnp.sum(jnp.log(jnp.diag(L))) * 2

    logdet_linalg = calc_logdet_linalg(K)

    logdet_rel_error = abs((logdet - logdet_linalg) / logdet_linalg)

    ## calc trace terms
    trace_rel_error_list = []
    I = jnp.eye(len(delta_y_train))
    Kinv = jnp.linalg.solve(L.T, jnp.linalg.solve(L, I))
    for dK in dKdtheta:
        if precondition:
            trace = calc_trace.calc_trace(
                Kinvy, dK, precondition(zs), n_tridiag=n_tridiag
            )
        else:
            trace = calc_trace.calc_trace(Kinvy, dK, zs, n_tridiag=n_tridiag)

        trace_linalg = jnp.sum(jnp.diag(jnp.matmul(Kinv, dK)))

        trace_rel_error_list.append(abs((trace - trace_linalg) / trace_linalg))

        # print(f"mean of dK: {jnp.mean(dK):.3e}")
    # print(f"trace_rel_error_list: {np.array(trace_rel_error_list)}")
    # print(f"trace: {np.array(trace)}")
    # print(f"trace_linalg: {np.array(trace_linalg)}")
    trace_rel_error = np.mean(np.array(trace_rel_error_list))

    print(f"linear_solve_rel_error: {linear_solve_rel_error:.3e}")
    print(f"logdet_rel_error: {logdet_rel_error:.3e}")
    print(f"trace_rel_error: {trace_rel_error:.3e}")
    print("\n\n")

    return linear_solve_rel_error, logdet_rel_error, trace_rel_error
