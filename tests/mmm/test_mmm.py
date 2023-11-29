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

import jax
import jax.numpy as jnp
import numpy as np
import stopro.GP.gp_sinusoidal_independent as gp_sinusoidal_independent
from jax.config import config
from stopro.data_handler.data_handle_module import *
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.load_modules import load_data, load_params

import bbmm.utils.mmm as mmm

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


def calc_K_x_right_matrix(
    simulation_path: str = "tests/data/sinusoidal_direct",
    scale: float = 1.0,
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
        delta_y_train = jnp.append(delta_y_train, f_train[i] / scale**2 - μ_train[i])
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

    cov_scale = 0.0
    length = 2.3
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

    # ## calc covariance matrix
    # K = gp_model.trainingK_all(init, r_train)
    # K = gp_model.add_eps_to_sigma(K, params_model["epsilon"], noise_parameter=None)

    right_matrix = jax.random.normal(
        jax.random.PRNGKey(0), (params_prepare["num_points"]["training"]["sum"], 11)
    )

    # Ks = gp_model.trainingKs(init, r_train)
    # for i in range(len(Ks)):
    #     for j in list(range(len(Ks) - len(Ks[i])))[::-1]:
    #         Ks[i] = [Ks[j][i]] + Ks[i]

    mmm_K = mmm.setup_mmm_K(
        r_train=r_train,
        gp_model=gp_model,
        theta=init,
        jiggle=params_model["epsilon"],
    )
    K_x_right_matrix = mmm_K(right_matrix=right_matrix)

    K = gp_model.trainingK_all(init, r_train)
    K = gp_model.add_eps_to_sigma(K, params_model["epsilon"], noise_parameter=None)
    K_x_right_matrix_naive = jnp.matmul(K, right_matrix)

    mean_rel_error = jnp.mean(
        jnp.abs((K_x_right_matrix_naive - K_x_right_matrix) / K_x_right_matrix_naive)
    )

    mmm_dKdtheta = mmm.setup_mmm_dKdtheta(
        r_train, gp_model, init, params_model["epsilon"]
    )
    dKdtheta_x_right_matrix = mmm_dKdtheta(right_matrix=right_matrix)

    def calc_trainingK(theta):
        Σ = gp_model.trainingK_all(theta, r_train)
        Σ = gp_model.add_eps_to_sigma(Σ, params_model["epsilon"], noise_parameter=None)
        return Σ

    dKdtheta = jnp.transpose(jax.jacfwd(calc_trainingK)(init), (2, 0, 1))
    dKdtheta_x_right_matrix_ref = jnp.matmul(dKdtheta, right_matrix)

    mean_abs_error_dKdtheta = jnp.mean(
        jnp.abs(
            (
                dKdtheta_x_right_matrix_ref
                - jnp.transpose(dKdtheta_x_right_matrix, (2, 0, 1))
            )
        )
    )

    return mean_rel_error, mean_abs_error_dKdtheta
