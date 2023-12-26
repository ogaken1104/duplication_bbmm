## _linear_operator.pyからLinearOperatorクラスをimportしてDenseLInearOPeratorクラスをうtくる
# Path: duplication_of_bbmm/operators/dese_linear_operator.py
# Compare this snippet from duplication_of_bbmm/operators/dese_linear_operator.py:
import jax.numpy as jnp

from bbmm.operators._linear_operator import LinearOp
from bbmm.operators.dense_linear_operator import DenseLinearOp, to_linear_operator


class DiagLinearOp(LinearOp):
    def __init__(self, diag: jnp.ndarray) -> None:
        self._diag = diag

    @property
    def shape(self) -> tuple[int]:
        return (self._diag.shape[0], self._diag.shape[0])

    def matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        diag = self._diag if rhs.ndim == 1 else jnp.expand_dims(self._diag, axis=-1)
        return diag * rhs

    def _root_decomposition(self):
        return self.sqrt()

    def sqrt(self):
        return self.__class__(jnp.sqrt(self._diag))

    def _diagonal(self) -> jnp.array:
        return self._diag
