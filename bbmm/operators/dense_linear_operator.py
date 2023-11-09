## _linear_operator.pyからLinearOperatorクラスをimportしてDenseLInearOPeratorクラスをうtくる
# Path: duplication_of_bbmm/operators/dese_linear_operator.py
# Compare this snippet from duplication_of_bbmm/operators/dese_linear_operator.py:
import jax.numpy as jnp

from bbmm.operators._linear_operator import LinearOp


class DenseLinearOp(LinearOp):
    def __init__(self, array: jnp.ndarray) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int]:
        return self.array.shape

    def _diagonal(self) -> jnp.array:
        return jnp.diagonal(self.array, axis1=-2, axis2=-1)

    def _matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(self.array, rhs)

    def __getitem__(self, index) -> jnp.ndarray:
        return self.array[index]
