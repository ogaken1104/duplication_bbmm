import jax.numpy as jnp

from bbmm.operators.sum_linear_operator import SumLinearOp
from bbmm.operators._linear_operator import LinearOp
from bbmm.operators.diag_linear_operator import DiagLinearOp


class AddedDiagLinearOp(SumLinearOp):
    """
    :param linear_ops: The Linear Op and the DiagLinearOp to add to it.
    below is copy paste of AddedDiagLinearOperator from the linear_operator library in github.
    ##################
    A :class:`~linear_operator.operators.SumLinearOperator`, but of only two
    linear operators, the second of which must be a
    :class:`~linear_operator.operators.DiagLinearOperator`.

    :param linear_ops: The LinearOperator, and the DiagLinearOperator to add to it.
    :param preconditioner_override: A preconditioning method to be used with conjugate gradients.
        If not provided, the default preconditioner (based on the partial pivoted Cholesky factorization) will be used
        (see `Gardner et al., NeurIPS 2018`_ for details).

    .. _Gardner et al., NeurIPS 2018:
        https://arxiv.org/abs/1809.11165
    """

    def __init__(self, *linear_ops: [LinearOp, DiagLinearOp]):
        linear_ops = list(linear_ops)
        super().__init__(*linear_ops)
        self._linear_op = linear_ops[0]
        self._diag_tensor = linear_ops[1]

    def matmul(
        self,
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._linear_op.matmul(rhs) + self._diag_tensor.matmul(rhs)
