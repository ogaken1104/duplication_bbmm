import inspect
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.config import config

config.update("jax_enable_x64", True)

import warnings

warnings.filterwarnings("always")

from bbmm.utils import calc_loss_dloss, test_modules

import gpytorch
import torch

torch.set_default_dtype(torch.float64)

# import stopro.solver.optimizers as optimizers
import stopro.GP.gp_1D_naive as gp_1D_naive
from stopro.data_handler.data_handle_module import HdfOperator
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

#########


atol = 1
## we want to check with using CG of gpytorch, but unknwon BUG arrised.
# linear_operator.settings.max_cholesky_size._set_value(1)


def calc_loss_sin(
    project_name,
    simulation_name,
    init,
    scale,
    test_gpytorch=False,
    kwargs_setup_loss=None,
):
    print("\n\n")
    caller_name = inspect.currentframe().f_back.f_code.co_name
    print(f"#############")
    print(f"{caller_name}")
    print(f"#############")
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

    # # prepare initial hyper-parameter
    # init = get_init(
    #     params_model["init_kernel_hyperparameter"],
    #     params_model["kernel_type"],
    #     system_type=params_model["system_type"],
    # )

    # prepare data
    hdf_operator = HdfOperator(f"{project_name}/{simulation_name}")
    r_test, μ_test, r_train, μ_train, f_train = load_data(lbls, vnames, hdf_operator)
    _, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    r_train = [_r * scale for _r in r_train]
    r_test = [_r * scale for _r in r_test]
    delta_y_train = jnp.empty(0)
    for i in range(len(r_train)):
        delta_y_train = jnp.append(delta_y_train, f_train[i] - μ_train[i])

    params_model["epsilon"] = 1e-06
    args_predict = r_test, μ_test, r_train, delta_y_train, params_model["epsilon"]
    noise = params_model["epsilon"]

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

    if kwargs_setup_loss is None:
        kwargs_setup_loss = {
            "rank": 5,
            "n_tridiag": 10,
            "max_tridiag_iter": 20,
            "cg_tolerance": 1,
            "max_iter_cg": 1000,
            "min_preconditioning_size": 2000,
        }
    func_value_grad_mpcg = calc_loss_dloss.setup_loss_dloss_mpcg(
        gp_model=gp_model,
        return_yKinvy=True,
        **kwargs_setup_loss,
    )

    loss_ours, dloss_ours, yKinvy_ours, yKdKKy_ours = func_value_grad_mpcg(
        init, *args_predict[2:]
    )

    loss_cholesky = func(init, *args_predict[2:]) / len(K)
    dloss_cholesky = dfunc(init, *args_predict[2:]) / len(K)
    yKinvy_cholesky = gp_model.calc_yKinvy(init, *args_predict[2:])
    yKdKKy_cholesky = gp_model.calc_yKdKKy(init, *args_predict[2:])

    ## check gpytorch
    if test_gpytorch:
        # set parameters same for ours
        test_modules.set_linear_operator_settings(kwargs_setup_loss)

        train_x = torch.from_numpy(np.array(r_train[0], dtype=np.float64))
        train_y = torch.from_numpy(np.array(f_train[0], dtype=np.float64))
        # test_x = torch.from_numpy(np.array(r_test[0]))

        # set up the model
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                # RBFKernelにはscaleのパラメータが含まれていないため、ScaleKernelで囲む
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # initialize liklihood and model
        noise_constraint = gpytorch.constraints.Interval(1e-7, 1e-5)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )
        model = ExactGPModel(train_x, train_y, likelihood)
        model.likelihood.noise = torch.tensor(1e-06, dtype=torch.float64)
        model.covar_module.outputscale = torch.tensor(
            np.exp(init[0]), dtype=torch.float64
        )
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            np.exp(init[1]), dtype=torch.float64
        )

        model.train()
        likelihood.train()

        output = model(train_x)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        loss_torch = (
            -mll(output, train_y)
            + torch.log(model.covar_module.outputscale)
            + torch.log(model.covar_module.base_kernel.lengthscale.squeeze())
        )
        dloss_torch = torch.autograd.grad(loss_torch, model.covar_module.parameters())

        loss_torch = loss_torch.detach().numpy().squeeze()
        ## dL/dtheta = dL/dexp(theta) * exp(theta)
        # dloss_torch = np.multiply(
        #     np.exp(init),
        #     np.array(
        #         [dloss_torch[0].detach().numpy(), dloss_torch[1].detach().numpy()[0][0]]
        #     ),
        # )
        dloss_torch = np.multiply(
            np.exp(init),
            np.array([np.squeeze(d.detach().numpy()) for d in dloss_torch]),
        )
    print(f"yKinvy ours: {yKinvy_ours:.1e}")
    print(f"yKinvy cholesky: {yKinvy_cholesky:.1e}")
    print(f"aerr yKinvy: {jnp.abs(yKinvy_cholesky-yKinvy_ours):.1e}")
    print("\n")
    print(f"loss ours: {loss_ours:.1e}")
    print(f"loss cholesky: {loss_cholesky:.1e}")
    if test_gpytorch:
        print(f"loss torch: {loss_torch:.1e}")
    print("\n")

    print(
        f"yKdKKy ours: {np.array2string(yKdKKy_ours, formatter={'float_kind': '{:.1e}'.format}, separator=', ')}"
    )
    print(
        f"yKdKKy cholesky: {np.array2string(yKdKKy_cholesky, formatter={'float_kind': '{:.1e}'.format}, separator=', ')}"
    )
    print(f"aerr yKdKKy: {jnp.mean(jnp.abs(yKdKKy_cholesky-yKdKKy_ours)):.1e}")
    print("\n")

    print(f"dloss ours: {dloss_ours}")
    print(f"dloss cholesky: {dloss_cholesky}")
    if test_gpytorch:
        print(f"dloss torch: {dloss_torch}")
    print("\n")
    aerr_loss_ours = jnp.abs(loss_cholesky - loss_ours)
    aerr_dloss_ours = jnp.mean(jnp.abs(dloss_cholesky - dloss_ours))
    print(f"aerr loss ours: {aerr_loss_ours:.1e}")
    print(f"aerr dloss ours: {aerr_dloss_ours:.1e}")
    print("\n")
    if test_gpytorch:
        aerr_loss_torch = jnp.abs(loss_cholesky - loss_torch)
        aerr_dloss_torch = jnp.mean(jnp.abs(dloss_cholesky - dloss_torch))
        print(f"aerr loss torch: {aerr_loss_torch:.1e}")
        print(f"aerr dloss torch: {aerr_dloss_torch:.1e}")

    assert aerr_loss_ours < atol
    assert aerr_dloss_ours < atol


# def test_loss_sin1d_10_init_0():
#     project_name = "data"
#     simulation_name = "test_loss_sin1d_naive"
#     init = jnp.array([0.0, 0.0])
#     scale = 1.0
#     calc_loss_sin(project_name, simulation_name, init, scale, test_gpytorch=True)


# def test_loss_sin1d_10_init_2():
#     project_name = "data"
#     simulation_name = "test_loss_sin1d_naive"
#     init = jnp.array([2.0, 2.0])
#     scale = 1.0
#     calc_loss_sin(project_name, simulation_name, init, scale, test_gpytorch=True)


# def test_loss_sin1d_1000_x100_init_0():
#     project_name = "data"
#     simulation_name = "test_loss_sin1d_naive_y_1000"
#     init = jnp.array([0.0, 0.0])
#     scale = 100.0
#     kwargs_setup_loss = {
#         "rank": 5,
#         "n_tridiag": 10,
#         "max_tridiag_iter": 20,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 2000,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         kwargs_setup_loss=kwargs_setup_loss,
#         test_gpytorch=True,
#     )


# def test_loss_sin1d_1000_x100_init_1():
#     project_name = "data"
#     simulation_name = "test_loss_sin1d_naive_y_1000"
#     init = jnp.array([1.0, 1.0])
#     scale = 100.0
#     kwargs_setup_loss = {
#         "rank": 5,
#         "n_tridiag": 40,
#         "max_tridiag_iter": 80,
#         "cg_tolerance": 0.01,
#         "max_iter_cg": 2000,
#         "min_preconditioning_size": 2000,
#     }
#     calc_loss_sin(
#         project_name,
#         simulation_name,
#         init,
#         scale,
#         kwargs_setup_loss=kwargs_setup_loss,
#         test_gpytorch=True,
#     )


def test_loss_sin1d_1000_x100_init_2():
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
        test_gpytorch=True,
    )
