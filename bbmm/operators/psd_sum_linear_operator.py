import jax.numpy as jnp

from bbmm.operators._linear_operator import LinearOp


class SumLinearOp(LinearOp):
    def __init__(self, *linear_ops):
        self.linear_ops = linear_ops

    def matmul(
        self,
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        return sum(linear_op.matmul(rhs) for linear_op in self.linear_ops)


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
