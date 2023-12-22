## _linear_operator.pyからLinearOperatorクラスをimportしてDenseLInearOPeratorクラスをうtくる
# Path: duplication_of_bbmm/operators/dese_linear_operator.py
# Compare this snippet from duplication_of_bbmm/operators/dese_linear_operator.py:
import jax.numpy as jnp

from bbmm.operators._linear_operator import LinearOp
from bbmm.operators.dense_linear_operator import DenseLinearOp, to_linear_operator


class RootLinearOp(LinearOp):
    def __init__(self, root: jnp.ndarray) -> None:
        root = to_linear_operator(root)
        # super().__init__(root)
        self.root = root

    @property
    def shape(self) -> tuple[int]:
        return (self.root.shape[-2], self.root.shape[-2])

    def matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        return self.root.matmul(
            self.root.t_matmul(rhs)
        )  # different from reference: do not permutate

    def root_decomposition(self):
        return self
