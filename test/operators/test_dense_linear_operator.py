import jax.numpy as jnp

from bbmm.operators.dese_linear_operator import DenseLinearOperator


def test_dense_linear_operator():
    array = jnp.array([[1, 2], [3, 4]])
    dense_linear_operator = DenseLinearOperator(array)
    assert dense_linear_operator.shape == (2, 2)
    assert jnp.all(dense_linear_operator._diagonal() == jnp.array([1, 4]))
    assert jnp.all(
        dense_linear_operator._matmu(jnp.array([1, 2])) == jnp.array([5, 11])
    )
    assert jnp.all(dense_linear_operator[0] == jnp.array([1, 2]))
    assert jnp.all(dense_linear_operator[0, 1] == jnp.array(2))
    assert jnp.all(
        dense_linear_operator[jnp.array([1, 0])] == jnp.array([[3, 4], [1, 2]])
    )
