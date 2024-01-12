import jax
import jax.numpy as jnp
import numpy as np
import stopro.GP.gp_sinusoidal_independent as gp_sinusoidal_independent
from jax.config import config
from stopro.data_handler.data_handle_module import HdfOperator
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.load_modules import load_data, load_params

from bbmm.operators.lazy_evaluated_kernel_matrix import LazyEvaluatedKernelMatrix
from bbmm.operators.added_diag_linear_operator import AddedDiagLinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp

config.update("jax_enable_x64", True)


def test_lazy_evaluated_kernel_matrix(
    simulation_path: str = "tests/data/sinusoidal_direct",
    scale: float = 1.0,
):
    params_main, params_prepare, lbls = load_params(f"{simulation_path}/data_input")
    params_model = params_main["model"]
    vnames = params_prepare["vnames"]
    params_setting = params_prepare["setting"]

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

    Kss = gp_model.trainingKs.copy()
    for i in range(len(Kss)):
        for j in list(range(len(Kss) - len(Kss[i])))[::-1]:
            Kss[i] = [Kss[j][i]] + Kss[i]

    right_matrix = jax.random.normal(
        jax.random.PRNGKey(0), (params_prepare["num_points"]["training"]["sum"], 11)
    )

    lazy_evaluated_kernel_matrix = LazyEvaluatedKernelMatrix(
        r1s=r_train,
        r2s=r_train,
        Kss=Kss,
        sec1=gp_model.sec_tr,
        sec2=gp_model.sec_tr,
        jiggle=params_model["epsilon"],
    )
    lazy_evaluated_kernel_matrix.set_theta(init)

    assert lazy_evaluated_kernel_matrix.shape == (
        gp_model.sec_tr[-1],
        gp_model.sec_tr[-1],
    )

    K_x_right_matrix = lazy_evaluated_kernel_matrix.matmul(rhs=right_matrix)

    K = gp_model.trainingK_all(init, r_train)
    K = gp_model.add_eps_to_sigma(K, params_model["epsilon"], noise_parameter=None)
    K_x_right_matrix_naive = jnp.matmul(K, right_matrix)

    assert jnp.allclose(K_x_right_matrix_naive, K_x_right_matrix)
    assert jnp.all(lazy_evaluated_kernel_matrix._diagonal() == jnp.diag(K))

    # assert jnp.all(lazy_evaluated_kernel_matrix[0] == jnp.array([1, 2]))
    # assert jnp.all(lazy_evaluated_kernel_matrix[0, 1] == jnp.array(2))
    assert jnp.all(
        lazy_evaluated_kernel_matrix[jnp.array([1, 0])] == K[jnp.array([1, 0])]
    )

    ## also check when using AddedDiagLinearOperator
    lazy_evaluated_kernel_matrix = LazyEvaluatedKernelMatrix(
        r1s=r_train,
        r2s=r_train,
        Kss=Kss,
        sec1=gp_model.sec_tr,
        sec2=gp_model.sec_tr,
        jiggle=None,
    )
    lazy_evaluated_kernel_matrix.set_theta(init)
    added_diag = AddedDiagLinearOp(
        lazy_evaluated_kernel_matrix,
        DiagLinearOp(jnp.full(gp_model.sec_tr[-1], params_model["epsilon"])),
    )
    K_x_right_matrix_added_diag = added_diag.matmul(rhs=right_matrix)
    assert jnp.allclose(K_x_right_matrix_naive, K_x_right_matrix_added_diag)
    # assert jnp.all(added_diag._diagonal() == jnp.diag(K))
    # assert jnp.all(
    #     added_diag[jnp.array([1, 0])] == K[jnp.array([1, 0])]
    # )
