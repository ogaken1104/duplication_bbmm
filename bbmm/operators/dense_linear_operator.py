from typing import Union

import jax.numpy as jnp
import numpy as np

from bbmm.operators._linear_operator import LinearOp

# from bbmm.operators.diag_linear_operator import DiagLinearOp


class DenseLinearOp(LinearOp):
    def __init__(self, array: jnp.ndarray) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int]:
        return self.array.shape

    def _diagonal(self) -> jnp.array:
        return jnp.diagonal(self.array, axis1=-2, axis2=-1)

    def matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(self.array, rhs)

    def t_matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(self.array.T, rhs)

    def __getitem__(self, index) -> jnp.ndarray:
        return self.array[index]


def to_linear_operator(obj: Union[jnp.array, LinearOp]) -> LinearOp:
    """
    A function which ensures that `obj` is a LinearOperator.
    - If `obj` is a LinearOperator, this function does nothing.
    - If `obj` is a (normal) jnp.array, this function wraps it with a `DenseLinearOperator`.
    """
    if isinstance(obj, jnp.ndarray) or isinstance(obj, np.ndarray):
        return DenseLinearOp(obj)
    elif isinstance(obj, LinearOp):
        return obj
    else:
        raise TypeError(
            "object of class {} cannot be made into a LinearOp".format(
                obj.__class__.__name__
            )
        )
