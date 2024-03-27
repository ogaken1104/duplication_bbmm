import jax.numpy as jnp
import numpy as np

from tests.loss_dloss.test_calc_loss_sin_linearop import calc_loss_sin


def test_loss_sin1d_1000(seed):
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive_y_1000"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    test_ours, test_cholesky, test_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        kwargs_setup_loss=kwargs_setup_loss,
        test_gpytorch=True,
        use_lazy_matrix=True,
        return_time=True,
        seed=seed,
    )
    return test_ours, test_cholesky, test_torch


def test_loss_sin1d_5000(seed):
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_5000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    test_ours, test_cholesky, test_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        return_time=True,
        seed=seed,
    )
    return test_ours, test_cholesky, test_torch


def test_loss_sin1d_10000(seed):
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_10000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 20,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    test_ours, test_cholesky, test_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        return_time=True,
        seed=seed,
    )
    return test_ours, test_cholesky, test_torch


def test_loss_sin1d_20000(seed):
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_20000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    test_ours, test_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        test_cholesky=False,
        test_ours=True,
        return_time=True,
        seed=seed,
    )
    return test_ours, test_torch


def test_loss_sin1d_30000(seed):
    project_name = "tests/data"
    simulation_name = "test_sin1d_naive_y_30000"
    init = jnp.array([0.0, -1.0])
    scale = 1.0
    kwargs_setup_loss = {
        "rank": 0,
        "n_tridiag": 20,
        "max_tridiag_iter": 40,
        "cg_tolerance": 1,
        "max_iter_cg": 1000,
        "min_preconditioning_size": 2000,
    }
    test_ours, test_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        use_lazy_matrix=True,
        kwargs_setup_loss=kwargs_setup_loss,
        test_cholesky=False,
        return_time=True,
        seed=seed,
    )
    return test_ours, test_torch


def test_loss_sin1d_10_init_0(seed=0):
    project_name = "data"
    simulation_name = "test_loss_sin1d_naive"
    init = jnp.array([0.0, 0.0])
    scale = 1.0
    time_ours, time_cholesky, time_torch = calc_loss_sin(
        project_name,
        simulation_name,
        init,
        scale,
        test_gpytorch=True,
        return_time=True,
        seed=seed,
    )
    return time_ours, time_cholesky, time_torch


if __name__ == "__main__":
    seed_list = list(np.arange(10, dtype=int))
    num_list = [1000, 5000, 10000]
    func_list = [
        test_loss_sin1d_1000,
        test_loss_sin1d_5000,
        test_loss_sin1d_10000,
    ]
    for i, num in enumerate(num_list):
        time_ours_list = []
        time_cholesky_list = []
        time_torch_list = []
        for j, seed in enumerate(seed_list):
            tim_ours, time_cholesky, time_torch = func_list[i](seed=seed)
            time_ours_list.append(tim_ours)
            time_cholesky_list.append(time_cholesky)
            time_torch_list.append(time_torch)
        save_array = np.array([time_ours_list, time_cholesky_list, time_torch_list])
        print(save_array)
        np.save(f"tests/for_paper/data/sin_{num}.npy", save_array)

    num_list = [20000, 30000]
    func_list = [
        test_loss_sin1d_20000,
        test_loss_sin1d_30000,
    ]
    for i, num in enumerate(num_list):
        time_ours_list = []
        time_torch_list = []
        for j, seed in enumerate(seed_list):
            tim_ours, time_torch = func_list[i](seed=seed)
            time_ours_list.append(tim_ours)
            # time_cholesky_list.append(time_cholesky)
            time_torch_list.append(time_torch)
        save_array = np.array([time_ours_list, time_torch_list])
        print(save_array)
        np.save(f"tests/for_paper/data/sin_{num}.npy", save_array)
