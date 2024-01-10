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

    def _diagonal(self):
        return sum(linear_op._diagonal() for linear_op in self.linear_ops)

    @property
    def shape(self) -> tuple[int]:
        return self.linear_ops[0].shape
