import jax.numpy as jnp

from bbmm.operators._linear_operator import LinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp
from bbmm.operators.root_linear_operator import RootLinearOp


class SumLinearOp(LinearOp):
    def __init__(self, *linear_ops):
        self.linear_ops = linear_ops

    def matmul(
        self,
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        return sum(linear_op.matmul(rhs) for linear_op in self.linear_ops)

    def _diagonal(self):
        return sum(linear_op._diagonal() for linear_op in self.linear_ops)


class PsdSumLinearOp(SumLinearOp):
    def zero_mean_mvn_samples(self, num_samples: int, seed: int = 0) -> jnp.ndarray:
        return jnp.sum(
            jnp.array(
                [
                    linear_op.zero_mean_mvn_samples(num_samples, seed=seed)
                    for linear_op in self.linear_ops
                ]
            ),
            axis=0,
        )

    def left_term_of_trace(self):
        """
        fuction to calculatethe left term of element-wise multiplicatoin in trace
        https://arxiv.org/abs/2107.00243

        calculate L_l(\sigma^2I + L_l^TL_l) directly using cholesky decomposition
        """
        # if not isinstance(self.linear_ops[0], RootLinearOp):
        #     raise TypeError(
        #         f"first linear op must be RootLinearOp, but now {self.linear_ops[0].__class__.__name__}"
        #     )
        # if not isinstance(self.linear_ops[1], DiagLinearOp):
        #     raise TypeError(
        #         f"second linear op must be DiagLinearOp, but now {self.linear_ops[1].__class__.__name__}"
        #     )

        L = self.linear_ops[0].root
        sigma = self.linear_ops[1]._diag[0]
        eye = jnp.eye(L.shape[1])
        sigma_LTL = sigma * eye + L.t_matmul(L.matmul(eye))

        cho_L = jnp.linalg.cholesky(sigma_LTL)
        Inv = jnp.linalg.solve(
            cho_L.T, jnp.linalg.solve(cho_L, jnp.eye(cho_L.shape[0]))
        )

        return L.matmul(Inv)
