from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax


def mmm(
    r1s: list[jnp.array],
    r2s: list[jnp.array],
    right_matrix: jnp.array,
    Kss: list[list[Callable]],
    sec1: list[int],
    sec2: list[int],
):
    """
    function to calculate matrix-matrix multiplication K(r1s, r2s) @ right_matrix
    """
    ## 解のarrayを確保
    res = jnp.zeros_like(right_matrix)

    ## Kの各行を計算する関数を返す関数
    def setup_calc_K_row(sec1_index):
        Ks = Kss[sec1_index]

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

    ## calculate vmm for each row
    for k in range(len(sec1[:-1])):
        r1s_k = jnp.expand_dims(r1s[k], axis=1)
        index_scan = jnp.arange(sec1[k], sec1[k + 1])
        calc_K_row = setup_calc_K_row(k)

        def calc_vmm(res, xs):
            """
            function to calculate vector-matrix multiplication K(r1, ) x right_matrix
            """
            i, r1 = xs
            res = res.at[i, :].set(jnp.matmul(calc_K_row(r1), right_matrix))
            return res, None

        res, _ = lax.scan(calc_vmm, res, xs=(index_scan, r1s_k))

    return res
