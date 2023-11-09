from typing import Iterable, Union

import jax.numpy as jnp

IndexType = Union[type(Ellipsis), slice, Iterable[int], int]


class LinearOperator:
    """
    for simplicity, this class only supports 2D array
    """

    def __init__(self) -> None:
        pass

    def _matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "The class {} requires a _matmul function!".format(self.__class__.__name__)
        )

    def _diagonal(self) -> jnp.array:
        r"""
        As :func:`jnp.diagonal()`, returns the diagonal of the matrix

        :return: The diagonal (or batch of diagonals) of :math:`\mathbf A`.
        """
        raise NotImplementedError(
            "The class {} requires a _diagonal function!".format(
                self.__class__.__name__
            )
        )

    @property
    def shape(self) -> tuple[int]:
        raise NotImplementedError(
            "The class {} requires a shape function!".format(self.__class__.__name__)
        )

    def __getitem__(self, index: IndexType) -> jnp.ndarray:
        raise NotImplementedError(
            "The class {} requires a __getitem__ function!".format(
                self.__class__.__name__
            )
        )
