import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

##

import bbmm.functions.pivoted_cholesky_jax as pc_jax
import bbmm.utils.conjugate_gradient as cg
import bbmm.utils.preconditioner as precond
from bbmm.operators.dense_linear_operator import DenseLinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.added_diag_linear_operator import AddedDiagLinearOp
from bbmm.operators.lazy_evaluated_kernel_matrix import LazyEvaluatedKernelMatrix


def setup_predictor_mpcg(
    rank=15,
    n_tridiag=10,
    seed=0,
    tolerance=0.001,
    max_tridiag_iter=20,
    max_iter_cg=1000,
    min_preconditioning_size=1000,
    gp_model=None,
    use_lazy_matrix=False,
    matmul_blockwise=False,
):
    if use_lazy_matrix:
        Kss = gp_model.trainingKs.copy()
        for i in range(len(Kss)):
            for j in list(range(len(Kss) - len(Kss[i])))[::-1]:
                Kss[i] = [Kss[j][i]] + Kss[i]

    def predictor_mpcg(opt, *args):
        r_test, μ_test, r, delta_y, noise = args

        if use_lazy_matrix:
            _K_linear_op = LazyEvaluatedKernelMatrix(
                r1s=r,
                r2s=r,
                Kss=Kss,
                sec1=gp_model.sec_tr,
                sec2=gp_model.sec_tr,
                jiggle=False,
                matmul_blockwise=matmul_blockwise,
            )
            _K_linear_op.set_theta(opt)
        else:
            _K = gp_model.trainingK_all(opt, r)
            _K_linear_op = DenseLinearOp(_K)
        K_linear_op = AddedDiagLinearOp(
            _K_linear_op, DiagLinearOp(jnp.full(_K_linear_op.shape[0], noise))
        )

        Kab = gp_model.mixedK_all(opt, r_test, r)
        Kaa = gp_model.testK_all(opt, r_test)

        precondition, _, _ = precond.setup_preconditioner(
            _K_linear_op,
            rank=rank,
            noise=noise,
            min_preconditioning_size=min_preconditioning_size,
            func_pivoted_cholesky=pc_jax.pivoted_cholesky_jax,
        )
        rhs = jnp.concatenate([delta_y.reshape(-1, 1)], axis=1)

        Kinvy, j = cg.mpcg_bbmm(
            K_linear_op,
            rhs,
            precondition=precondition,
            print_process=False,
            tolerance=tolerance,
            n_tridiag=0,
            max_tridiag_iter=max_tridiag_iter,
            max_iter_cg=max_iter_cg,
        )

        rhs = Kab.T

        Kinvk, j = cg.mpcg_bbmm(
            K_linear_op,
            rhs,
            precondition=precondition,
            print_process=False,
            tolerance=tolerance,
            n_tridiag=0,
            max_tridiag_iter=max_tridiag_iter,
            max_iter_cg=max_iter_cg,
        )

        kKy = jnp.squeeze(jnp.matmul(Kab, Kinvy), axis=1)
        kKk = jnp.matmul(Kab, Kinvk)
        sec0 = 0
        sec1 = 0
        fs_mpcg = []
        Sigmas_mpcg = []
        for i in range(len(r_test)):
            sec1 += len(r_test[i])
            fs_mpcg.append(μ_test[i] + kKy[sec0:sec1])
            Sigmas_mpcg.append(Kaa[sec0:sec1, sec0:sec1] - kKk[sec0:sec1, sec0:sec1])
            sec0 += len(r_test[i])
        return fs_mpcg, Sigmas_mpcg

    return predictor_mpcg
