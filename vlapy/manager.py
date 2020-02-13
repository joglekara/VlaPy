import shutil
import mlflow
import numpy as np

from vlapy.core import step, field
from vlapy import storage


def start_run(temp_path, nx, nv, nt, tmax, w0, k0, a0, diagnostics, name="test"):
    """
    End to end mlflow and xarray storage!!


    :param temp_path:
    :param nx:
    :param nv:
    :param nt:
    :param tmax:
    :param w0:
    :param k0:
    :param a0:
    :param name:
    :return:
    """

    mlflow_client = __start_client__()

    with mlflow.start_run(experiment_id=__get_exp_id(name, mlflow_client)):
        # Log initial conditions
        params_dict = {
            "nx": nx,
            "nv": nv,
            "w0": w0,
            "k0": k0,
            "nt": nt,
            "tmax": tmax,
            "a0": a0,
        }

        mlflow.log_params(params_dict)

        # Initialize machinery
        f = step.initialize(nx, nv)

        xmax = 2 * np.pi / k0
        xmin = 0.0
        dx = (xmax - xmin) / nx
        x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

        vmax = 6.0
        dv = 2 * vmax / nv
        v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
        kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

        t = np.linspace(0, tmax, nt)
        dt = t[1] - t[0]

        def driver_function(x, t):
            envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)
            return envelope * a0 * np.cos(k0 * x + w0 * t)

        e = field.get_total_electric_field(driver_function(x, t[0]), f=f, dv=dv, kx=kx)

        # Storage
        if nt // 4 < 100:
            t_store = 100
        else:
            t_store = nt // 4
        temp_field_store = np.zeros([t_store, nx])
        temp_dist_store = np.zeros([t_store, nx, nv])
        temp_t_store = np.zeros(t_store)
        it_store = 0
        storage_manager = storage.StorageManager(x, v, t, temp_path)

        # Time Loop
        for it in range(nt):
            e, f = step.full_leapfrog_ps_step(
                f, x, kx, v, kv, dv, t[it], dt, e, driver_function
            )

            # All storage stuff here
            temp_t_store[it_store] = t[it]
            temp_dist_store[it_store] = f
            temp_field_store[it_store] = e
            it_store += 1

            if it_store == t_store:
                storage_manager.batched_write_to_file(
                    temp_t_store, temp_field_store, temp_dist_store
                )
                it_store = 0

        # Diagnostics here
        diagnostics(storage_manager)

        # Log
        mlflow.log_artifacts(temp_path)

        # Cleanup
        shutil.rmtree(temp_path)


def __start_client__():
    return mlflow.tracking.MlflowClient()


def __get_exp_id(name, mlflow_client):
    experiment = mlflow_client.get_experiment_by_name(name)

    if experiment is None:
        exp_id = mlflow_client.create_experiment(name,)
    else:
        exp_id = experiment.experiment_id

    return exp_id
