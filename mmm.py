from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from stopro.GP.gp import GPmodel


def setup_mmm_K(
    r_train: list[jnp.array],
    gp_model: GPmodel,
    theta: jnp.array,
    jiggle: float,
):
    """
    setup function to calculate covariance matrix x right_matrix.
    """
    Kss = gp_model.trainingKs(theta)
    for i in range(len(Kss)):
        for j in list(range(len(Kss) - len(Kss[i])))[::-1]:
            Kss[i] = [Kss[j][i]] + Kss[i]
    mmm_K = partial(
        mmm,
        r1s=r_train,
        r2s=r_train,
        Kss=Kss,
        sec1=gp_model.sec_tr,
        sec2=gp_model.sec_tr,
        jiggle=jiggle,
    )
    return mmm_K


def setup_mmm_dKdtheta(
    r_train: list[jnp.array],
    gp_model: GPmodel,
    theta: jnp.array,
    jiggle: float,
):
    """
    setup function to calculate derivative of covariance matrix x right_matrix

    TODO This implementaion is not efficient because it takes derivatives of K x right_matrix. It is better to first take derivatives of K and next multiply with right_matrix.
    """

    def mmm_K_givne_theta(right_matrix, theta):
        Kss = gp_model.trainingKs(theta)
        for i in range(len(Kss)):
            for j in list(range(len(Kss) - len(Kss[i])))[::-1]:
                Kss[i] = [Kss[j][i]] + Kss[i]
        return mmm(
            r1s=r_train,
            r2s=r_train,
            sec1=gp_model.sec_tr,
            sec2=gp_model.sec_tr,
            jiggle=jiggle,
            right_matrix=right_matrix,
            Kss=Kss,
        )

    def mmm_dKdtheta(right_matrix):
        return jax.jacfwd(mmm_K_givne_theta, argnums=1)(right_matrix, theta)

    return mmm_dKdtheta
    # return mmm_K_givne_theta


def mmm(
    r1s: list[jnp.array],
    r2s: list[jnp.array],
    right_matrix: jnp.array,
    Kss: list[list[Callable]],
    sec1: list[int],
    sec2: list[int],
    jiggle: float,
):
    """
    function to calculate matrix-matrix multiplication K(r1s, r2s) @ right_matrix
    """
    ## 解のarrayを確保
    res = jnp.zeros_like(right_matrix)

    ## Kの各行を計算する関数を返す関数
    def setup_calc_K_row(sec1_index, sec2, Ks, r2s):
        def calc_K_row(r1):
            """
            function to calculate each row of K

            Returns:
                K_row: jnp.array
            """
            K_row = jnp.zeros(sec2[-1])
            for j in range(len(sec2) - 1):
                if j >= sec1_index:
                    K_row = K_row.at[sec2[j] : sec2[j + 1]].set(
                        jnp.squeeze(Ks[j](r1, r2s[j]))
                    )
                else:
                    K_row = K_row.at[sec2[j] : sec2[j + 1]].set(
                        jnp.squeeze(Ks[j](r2s[j], r1))
                    )
            return K_row

        return calc_K_row

    for k in range(len(sec1[:-1])):
        r1s_k = jnp.expand_dims(r1s[k], axis=1)
        index_scan = jnp.arange(sec1[k], sec1[k + 1])
        calc_K_row = setup_calc_K_row(k, sec2, Kss[k], r2s)

        def calc_vmm(res, xs):
            """
            function to calculate vector-matrix multiplication K(r1, ) x right_matrix
            """
            i, r1 = xs
            K_row = calc_K_row(r1)
            if jiggle:
                K_row = K_row.at[i].add(jiggle)
            res = res.at[i, :].set(jnp.matmul(K_row, right_matrix))
            return res, None

        ## calculate vmm for each row
        res, _ = lax.scan(calc_vmm, res, xs=(index_scan, r1s_k))

    return res
