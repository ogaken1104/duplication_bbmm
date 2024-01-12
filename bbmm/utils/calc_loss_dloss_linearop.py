import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

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
):
    if use_lazy_matrix:
        Kss = gp_model.trainingKs.copy()
        for i in range(len(Kss)):
            for j in list(range(len(Kss) - len(Kss[i])))[::-1]:
                Kss[i] = [Kss[j][i]] + Kss[i]
        gp_model.setup_Ks_dKdtheta()
        dKss = gp_model.Ks_dKdtheta.copy()
        for i in range(len(dKss)):
            for j in list(range(len(dKss) - len(dKss[i])))[::-1]:
                dKss[i] = [dKss[j][i]] + dKss[i]

    def loss_dloss_mpcg(init, *args):
        ## this part can be changed when K is LinearOp ##
        r, delta_y, noise = args

        if use_lazy_matrix:
            _K_linear_op = LazyEvaluatedKernelMatrix(
                r1s=r,
                r2s=r,
                Kss=Kss,
                sec1=gp_model.sec_tr,
                sec2=gp_model.sec_tr,
                jiggle=False,
            )
            _K_linear_op.set_theta(init)
        else:
            _K = gp_model.trainingK_all(init, r)
            _K_linear_op = DenseLinearOp(_K)
        K_linear_op = AddedDiagLinearOp(
            _K_linear_op, DiagLinearOp(jnp.full(_K_linear_op.shape[0], noise))
        )
        ######################################

        precondition, precond_lt, precond_logdet_cache = precond.setup_preconditioner(
            _K_linear_op,
            rank=rank,
            noise=noise,
            min_preconditioning_size=min_preconditioning_size,
            # func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        )
        if precondition:
            zs = precond_lt.zero_mean_mvn_samples(n_tridiag)
        else:
            zs = jax.random.normal(jax.random.PRNGKey(seed), (len(delta_y), n_tridiag))
        zs_norm = jnp.linalg.norm(zs, axis=0)
        zs = zs / zs_norm
        rhs = jnp.concatenate([zs, delta_y.reshape(-1, 1)], axis=1)

        Kinvy, j, t_mat = cg.mpcg_bbmm(
            K_linear_op,
            rhs,
            precondition=precondition,
            print_process=False,
            tolerance=cg_tolerance,
            n_tridiag=n_tridiag,
            max_tridiag_iter=max_tridiag_iter,
            max_iter_cg=max_iter_cg,
        )

        ## calc loss
        logdet = calc_logdet.calc_logdet(K_linear_op.shape, t_mat, precond_logdet_cache)
        yKy = jnp.dot(delta_y, Kinvy[:, -1])
        loss = (yKy + logdet) / 2 + len(delta_y) / 2 * jnp.log(jnp.pi * 2)

        ## jeffery's prior
        loss += jnp.sum(init)

        ## calc dloss
        dKdtheta_linear_op = []
        if use_lazy_matrix:
            lazy_kernel_derivative = LazyEvaluatedKernelMatrix(
                r1s=r,
                r2s=r,
                Kss=dKss,
                sec1=gp_model.sec_tr,
                sec2=gp_model.sec_tr,
                jiggle=False,
                num_component=len(init),
            )
            lazy_kernel_derivative.set_theta(init)
            dKzs_list = jnp.transpose(lazy_kernel_derivative.matmul(zs), (2, 0, 1))
            dKKinvy_list = jnp.transpose(lazy_kernel_derivative.matmul(Kinvy[:, -1]))
        else:

            def calc_trainingK(theta):
                Σ = gp_model.trainingK_all(theta, r)
                Σ = gp_model.add_eps_to_sigma(Σ, noise, noise_parameter=None)
                return Σ

            dKdtheta = jnp.transpose(jax.jacfwd(calc_trainingK)(init), (2, 0, 1))
            dKdtheta_linear_op = []
            for dK in dKdtheta:
                dKdtheta_linear_op.append(DenseLinearOp(dK))
        dloss = jnp.zeros(len(init))

        ## calc tr(P^{-1}\frac{dP}{dtheta}) beforehand
        if precondition:

            def calc_Poperand(theta, operand):
                _K = gp_model.trainingK_all(theta, r)
                _K_linear_op = DenseLinearOp(_K)
                _, precond_lt, _ = precond.setup_preconditioner(
                    _K_linear_op,
                    rank=rank,
                    noise=noise,
                    min_preconditioning_size=min_preconditioning_size,
                    func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
                )
                return precond_lt.matmul(operand)

            PinvdPz = jnp.transpose(jax.jacfwd(calc_Poperand, 0)(init, zs), (2, 0, 1))
            for i, _PinvdPz in enumerate(PinvdPz):
                PinvdPz = PinvdPz.at[i].set(precondition(_PinvdPz))
            ##### directly calculate trace(P) but high computational cost #########
            # def calc_P(theta):
            #     K = gp_model.trainingK_all(theta, r)
            #     K = gp_model.add_eps_to_sigma(K, noise, noise_parameter=None)
            #     _, precond_lt, _ = precond.setup_preconditioner(
            #         K,
            #         rank=rank,
            #         noise=noise,
            #         min_preconditioning_size=min_preconditioning_size,
            #         func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
            #     )
            #     return precond_lt.matmul(jnp.eye(len(K)))

            # dP = jnp.transpose(jax.jacfwd(calc_P)(init), (2, 0, 1))
            # trace_P = jnp.sum(
            #     jnp.diagonal(precondition(dP), axis1=-2, axis2=-1), axis=-1
            # )
            #######################################################
            def diagonal_dP(theta):
                _K = gp_model.trainingK_all(theta, r)
                _K_linear_op = DenseLinearOp(_K)
                _, precond_lt, _ = precond.setup_preconditioner(
                    _K_linear_op,
                    rank=rank,
                    noise=noise,
                    min_preconditioning_size=min_preconditioning_size,
                    func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
                )
                return precond_lt._diagonal()

            diag_dP = jnp.transpose(jax.jacfwd(diagonal_dP)(init), (1, 0))
            dPL = jnp.transpose(
                jax.jacfwd(calc_Poperand, 0)(init, precond_lt.linear_ops[0].root.array),
                (2, 0, 1),
            )
            left_term = precond_lt.left_term_of_trace()
            trace_P = (
                jnp.sum(diag_dP, axis=-1)
                - jnp.sum(jnp.multiply(left_term, dPL), axis=(-2, -1))
            ) / noise
        else:
            PinvdPz = jnp.zeros_like(zs)

        if return_yKinvy:
            yKdKKy_array = jnp.zeros(len(init))
        for i in range(len(init)):
            ## for large cond. # covariance, usually not reach convergence here.
            if use_lazy_matrix:
                dKzs = dKzs_list[i]
            else:
                dK_linear_op = dKdtheta_linear_op[i]
                dKzs = dK_linear_op.matmul(zs)
            KinvdKz, j = cg.mpcg_bbmm(
                K_linear_op,
                dKzs,
                precondition=precondition,
                print_process=False,
                tolerance=cg_tolerance,
                n_tridiag=0,
                max_iter_cg=max_iter_cg,
            )
            if precondition:
                KinvdKz -= PinvdPz[i]
            gamma = jnp.einsum("ij, ji->", zs.T, KinvdKz)
            gamma_sum = jnp.sum(gamma) / n_tridiag * K_linear_op.shape[0]
            if precondition:
                tau = gamma_sum + trace_P[i]
            else:
                tau = gamma_sum

            ## we have to modify here
            if use_lazy_matrix:
                yKdKKy = Kinvy[:, -1].T @ dKKinvy_list[i]
            else:
                yKdKKy = Kinvy[:, -1].T @ dK_linear_op.matmul(Kinvy[:, -1])
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
