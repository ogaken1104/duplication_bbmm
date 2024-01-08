import importlib

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.config import config

config.update("jax_enable_x64", True)

import warnings

warnings.filterwarnings("always")

import bbmm.operators.diag_linear_operator as diag_linear_operator
import bbmm.operators.psd_sum_linear_operator as psd_sum_linear_operator
import bbmm.operators.root_linear_operator as root_linear_operator
import bbmm.utils.preconditioner as precond
from bbmm.utils import calc_loss_dloss, calc_prediction, test_modules

importlib.reload(test_modules)
importlib.reload(psd_sum_linear_operator)
importlib.reload(diag_linear_operator)
importlib.reload(root_linear_operator)
importlib.reload(precond)
importlib.reload(calc_prediction)

import gpytorch
import linear_operator
import torch

# import stopro.solver.optimizers as optimizers
import stopro.GP.gp_1D_naive as gp_1D_naive
from stopro.data_handler.data_handle_module import HdfOperator
from stopro.GP.kernels import define_kernel
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

#########


atol = 1
## we want to check with using CG of gpytorch, but unknwon BUG arrised.
# linear_operator.settings.max_cholesky_size._set_value(1)


def test_loss_sin1d():
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

    func_value_grad_mpcg = calc_loss_dloss.setup_loss_dloss_mpcg(
        rank=5,
        n_tridiag=10,
        max_tridiag_iter=20,
        cg_tolerance=1,
        gp_model=gp_model,
    )

    loss_ours, dloss_ours = func_value_grad_mpcg(init, *args_predict[2:])

    loss_linalg = func(init, *args_predict[2:]) / len(K)
    dloss_linalg = dfunc(init, *args_predict[2:]) / len(K)

    ## check gpytorch
    train_x = torch.from_numpy(np.array(r_train[0]))
    train_y = torch.from_numpy(np.array(f_train[0]))
    test_x = torch.from_numpy(np.array(r_test[0]))

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
    model.likelihood.noise = torch.tensor([1e-06])
    model.covar_module.outputscale = torch.tensor(np.exp(np.array(init[0])))
    model.covar_module.base_kernel.lengthscale = torch.tensor(np.exp(np.array(init[1])))

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
        np.exp(init), np.array([np.squeeze(d.detach().numpy()) for d in dloss_torch])
    )

    print(f"loss ours: {loss_ours:.3e}")
    print(f"loss linalg: {loss_linalg:.3e}")
    print(f"loss torch: {loss_torch:.3e}")
    print(f"dloss ours: {dloss_ours}")
    print(f"dloss linalg: {dloss_linalg}")
    print(f"dloss torch: {dloss_torch}")
    aerr_loss_linalg = jnp.abs(loss_linalg - loss_ours)
    aerr_dloss_linalg = jnp.mean(jnp.abs(dloss_linalg - dloss_ours))
    aerr_loss_torch = jnp.abs(loss_torch - loss_ours)
    aerr_dloss_torch = jnp.mean(jnp.abs(dloss_torch - dloss_ours))
    print(f"aerr loss linalg: {aerr_loss_linalg:.2e}")
    print(f"aerr dloss linalg: {aerr_dloss_linalg:.2e}")
    print(f"aerr loss torch: {aerr_loss_torch:.2e}")
    print(f"aerr dloss torch: {aerr_dloss_torch:.2e}")

    assert aerr_loss_linalg < atol
    assert aerr_dloss_linalg < atol
