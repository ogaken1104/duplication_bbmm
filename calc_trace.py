import jax
import jax.numpy as jnp


def calc_trace(
    Kinvy: jnp.array, dKdtheta: jnp.array, probe_vectors: jnp.array, n_tridiag: int
):
    """
    function to calculate trace term
    """
    return (
        jnp.einsum(
            "ij, ij ->",
            Kinvy[:, :n_tridiag],
            jnp.einsum("ij, jk->ik", dKdtheta, probe_vectors),
        )
        / n_tridiag
    )
