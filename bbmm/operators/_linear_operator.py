from typing import Iterable, Union

import jax
import jax.numpy as jnp

IndexType = Union[type(Ellipsis), slice, Iterable[int], int]


class LinearOp:
    """
    for simplicity, this class only supports 2D array
    """

    def __init__(self) -> None:
        pass

    def matmul(self, rhs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "The class {} requires a matmul function!".format(self.__class__.__name__)
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

    def root_decomposition(self):
        from bbmm.operators.root_linear_operator import RootLinearOp

        root = self._root_decomposition()
        return RootLinearOp(root)

    def zero_mean_mvn_samples(self, num_samples: int, seed: int = 0) -> jnp.ndarray:
        covar_root = self.root_decomposition().root

        base_samples = jax.random.normal(
            key=jax.random.PRNGKey(seed), shape=(covar_root.shape[-1], num_samples)
        )

        samples = covar_root.matmul(base_samples)

        return samples
