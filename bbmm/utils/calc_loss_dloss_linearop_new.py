import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
import copy

config.update("jax_enable_x64", True)

##

import bbmm.functions.pivoted_cholesky_jax as pc_jax
import bbmm.utils.calc_logdet as calc_logdet
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.added_diag_linear_operator import AddedDiagLinearOp
from bbmm.operators.lazy_evaluated_kernel_matrix import LazyEvaluatedKernelMatrix


def setup_loss_dloss_mpcg(
    rank=15,
    n_tridiag=10,
    seed=0,
    cg_tolerance=0.01,
    max_tridiag_iter=20,
    min_preconditioning_size=2000,
    max_iter_cg=1000,
    gp_model=None,
    return_yKinvy=False,
    use_lazy_matrix=False,
    matmul_blockwise=False,
    args=None,
    num_component_init=2,
):
    r, delta_y, noise = args

    Kss = gp_model.trainingKs
    gp_model.setup_Ks_dKdtheta()
    dKss = gp_model.Ks_dKdtheta

    _K_linear_op = LazyEvaluatedKernelMatrix(
        r1s=r,
        r2s=r,
        Kss=Kss,
        sec1=gp_model.sec_tr,
        sec2=gp_model.sec_tr,
        matmul_blockwise=matmul_blockwise,
    )
    K_linear_op = AddedDiagLinearOp(
        _K_linear_op, DiagLinearOp(jnp.full(_K_linear_op.shape[0], noise))
    )
    lazy_kernel_derivative = LazyEvaluatedKernelMatrix(
        r1s=r,
        r2s=r,
        Kss=dKss,
        sec1=gp_model.sec_tr,
        sec2=gp_model.sec_tr,
        num_component=num_component_init,
        matmul_blockwise=matmul_blockwise,
    )

    def loss_dloss_mpcg(init):
        #################################################
        ## this part can be changed when K is LinearOp ##

        ######################################
        # _K_linear_op.set_theta(init)
        # K_linear_op = AddedDiagLinearOp(
        #     _K_linear_op, DiagLinearOp(jnp.full(_K_linear_op.shape[0], noise))
        # )
        # lazy_kernel_derivative.set_theta(init)
        print("pass")
        ## 1. setup preconditioner
        # precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
        #     _K_linear_op,
        #     rank=rank,
        #     noise=noise,
        #     min_preconditioning_size=min_preconditioning_size,
        #     # func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        # )
        precondition = None
        precond_logdet_cache = None
        # print(
        #     f"precond_lt: {precond_lt.matmul(jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag)))[0]}"
        # )
        # print(f"precond_logdet_cache: {precond_logdet_cache}")

        ## 2. generate random probe vectors
        zs = jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag))
        zs_norm = jnp.linalg.norm(zs, axis=0)
        zs = zs / zs_norm
        rhs = jnp.concatenate([zs, delta_y.reshape(-1, 1)], axis=1)

        ## 3. solve linear system Kx = rhs
        Kinvy, j, t_mat = cg.mpcg_bbmm(
            K_linear_op,
            rhs,
            precondition=precondition,
            print_process=False,
            tolerance=cg_tolerance,
            n_tridiag=n_tridiag,
            max_tridiag_iter=max_tridiag_iter,
            max_iter_cg=max_iter_cg,
            theta=init,
        )

        ## 4. calc loss
        logdet = calc_logdet.calc_logdet(K_linear_op.shape, t_mat, precond_logdet_cache)
        yKy = jnp.dot(delta_y, Kinvy[:, -1])
        loss = (yKy + logdet) / 2 + len(delta_y) / 2 * jnp.log(jnp.pi * 2)

        # add jeffery's prior
        loss += jnp.sum(init)

        # del K_linear_op, t_mat

        ## calc dloss
        ## 5. prepare dKdtheta linear_op
        dKdtheta_linear_op = []
        # lazy_kernel_derivative.set_theta(init)
        dKzs_list = jnp.transpose(lazy_kernel_derivative.matmul(zs, init), (2, 0, 1))
        dKKinvy_list = jnp.transpose(lazy_kernel_derivative.matmul(Kinvy[:, -1], init))
        dloss = jnp.zeros(len(init))

        ## 6. calc tr(P^{-1}\frac{dP}{dtheta}) beforehand
        PinvdPz = jnp.zeros_like(zs)

        if return_yKinvy:
            yKdKKy_array = jnp.zeros(len(init))

        ## 7. calc dloss
        for i in range(len(init)):
            ## for large cond. # covariance, usually not reach convergence here.
            dKzs = dKzs_list[i]
            KinvdKz, j = cg.mpcg_bbmm(
                K_linear_op,
                dKzs,
                precondition=precondition,
                print_process=False,
                tolerance=cg_tolerance,
                n_tridiag=0,
                max_iter_cg=max_iter_cg,
                theta=init,
            )
            gamma = jnp.einsum("ij, ji->", zs.T, KinvdKz)
            gamma_sum = jnp.sum(gamma) / n_tridiag * K_linear_op.shape[0]
            tau = gamma_sum

            ## we have to modify here
            yKdKKy = Kinvy[:, -1].T @ dKKinvy_list[i]
            # print(Kinvy[:, -1].shape)
            if return_yKinvy:
                yKdKKy_array = yKdKKy_array.at[i].set(yKdKKy)
            dloss_i = (-yKdKKy + tau) / 2
            dloss = dloss.at[i].set(dloss_i)
        # add derivative of jeffery's prior
        dloss += 1.0
        # print(dloss)
        if return_yKinvy:
            return (
                loss / gp_model.sec_tr[-1],
                dloss / gp_model.sec_tr[-1],
                yKy,
                yKdKKy_array,
            )
        else:
            return loss / gp_model.sec_tr[-1], dloss / gp_model.sec_tr[-1]

    return loss_dloss_mpcg
