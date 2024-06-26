import jax
import jax.numpy as jnp


def calc_trace(
    Kinvy: jnp.array,
    dKdtheta: jnp.array,
    probe_vectors: jnp.array,
    n_tridiag: int,
    precondition: callable = None,
):
    """
    function to calculate trace term
    """
    ### TODO this implementation is not right. need to fix
    ## see to_dense in stochastic_lq.py
    if precondition:
        probe_vectors = precondition(probe_vectors)
    return (
        jnp.einsum(
            "ij, ij ->",
            Kinvy[:, :n_tridiag],
            jnp.einsum("ij, jk->ik", dKdtheta, probe_vectors),
        )
        / n_tridiag
        * len(Kinvy)
    )
