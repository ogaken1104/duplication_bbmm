import jax.numpy as jnp
from jax import lax

from bbmm.operators._linear_operator import LinearOp


class LazyEvaluatedKernelMatrix(LinearOp):
    """
    refer to gpytroch.lazy.lazy_evaluated_kernel_tensor
    """

    def __init__(self, r1s, r2s, Kss, sec1, sec2, jiggle) -> None:
        self.r1s = r1s
        self.r2s = r2s
        self.Kss = Kss
        self.sec1 = sec1
        self.sec2 = sec2
        self.jiggle = jiggle

    @property
    def shape(self) -> tuple[int]:
        return self.sec1[-1], self.sec2[-1]

    def _diagonal(self) -> jnp.array:
        """
        not efficient because we use lax.scan for each component. we cannot use broadcasting of numpy because each kerenl functions in Kss is "vmapped".
        """
        res = jnp.zeros(self.sec1[-1])
        for k in range(len(self.sec1[:-1])):
            r1s_k = jnp.expand_dims(self.r1s[k], axis=1)
            index_scan = jnp.arange(self.sec1[k], self.sec1[k + 1])
            # calc_K_component = self.setup_calc_K_component(self.Kss[k, k])
            K = self.Kss[k][k]

            def calc_K_component(res, xs):
                i, r1 = xs
                res = res.at[i].set(jnp.squeeze(K(r1, r1)))
                return res, None

            ## calculate vmm for each row
            res, _ = lax.scan(calc_K_component, res, xs=(index_scan, r1s_k))
        res = jnp.add(res, self.jiggle)

        return res

    ## Kの各行を計算する関数を返す関数
    def setup_calc_K_row(self, sec1_index, Ks):
        def calc_K_row(r1):
            """
            function to calculate each row of K

            Returns:
                K_row: jnp.array
            """
            K_row = jnp.zeros(self.sec2[-1])
            for j in range(len(self.sec2) - 1):
                if j >= sec1_index:
                    K_row = K_row.at[self.sec2[j] : self.sec2[j + 1]].set(
                        jnp.squeeze(Ks[j](r1, self.r2s[j]))
                    )
                else:
                    K_row = K_row.at[self.sec2[j] : self.sec2[j + 1]].set(
                        jnp.squeeze(Ks[j](self.r2s[j], r1))
                    )
            return K_row

        return calc_K_row

    def _matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        ## 解のarrayを確保
        res = jnp.zeros_like(rhs)
        for k in range(len(self.sec1[:-1])):
            r1s_k = jnp.expand_dims(self.r1s[k], axis=1)
            index_scan = jnp.arange(self.sec1[k], self.sec1[k + 1])
            calc_K_row = self.setup_calc_K_row(k, self.Kss[k])

            def calc_vmm(res, xs):
                """
                function to calculate vector-matrix multiplication K(r1, ) x right_matrix
                """
                i, r1 = xs
                K_row = calc_K_row(r1)
                if self.jiggle:
                    K_row = K_row.at[i].add(self.jiggle)
                res = res.at[i, :].set(jnp.matmul(K_row, rhs))
                return res, None

            ## calculate vmm for each row
            res, _ = lax.scan(calc_vmm, res, xs=(index_scan, r1s_k))
        return res

    def __getitem__(self, index) -> jnp.ndarray:
        """
        at this point, just returns designated row permutation.
        """

        def get_sec1_index(i):
            for k, i_max in enumerate(self.sec1[1:]):
                if i < i_max:
                    return k

        res = jnp.zeros((len(index), self.shape[1]))
        for index_res, i in enumerate(index):
            k = get_sec1_index(i)
            r1s_k = jnp.expand_dims(self.r1s[k], axis=1)
            calc_K_row = self.setup_calc_K_row(k, self.Kss[k])

            K_row = calc_K_row(r1s_k[i - self.sec1[k]])
            if self.jiggle:
                K_row = K_row.at[i].add(self.jiggle)
            res = res.at[index_res, :].set(K_row)

        return res
