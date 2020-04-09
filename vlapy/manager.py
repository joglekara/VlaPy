import shutil
import mlflow
import numpy as np
import os
import uuid

from vlapy.core import step, field, lenard_bernstein
from vlapy import storage


def _get_rise_fall_coeff_(normalized_time):
    """
    This is a smooth function that goes from 0 to 1. It is a 5th order polynomial because it satisfies the following
    5 conditions

    value = 0 at t=0 and t=1 (2 conditions)
    first derivative = 0 at t=0 and t=1 (2 conditions)
    value = 0.5 in the middle

    :param normalized_time:
    :return:
    """
    coeff = (
        6.0 * pow(normalized_time, 5.0)
        - 15.0 * pow(normalized_time, 4.0)
        + 10.0 * pow(normalized_time, 3.0)
    )
    return coeff


def get_pulse_coefficient(pulse_profile_dictionary, tt):
    """
    This function generates an envelope that smoothly goes from 0 to 1, and back down to 0.
    It follows the nomenclature introduced to me by working with oscilloscopes.

    The pulse profile dictionary will contain a rise time, flat time, and fall time.
    The rise time is the time over which the pulse goes from 0 to 1
    The flat time is the duration of the flat-top of the pulse
    The fall time is the time over which the pulse goes from 1 to 0.

    :param pulse_profile_dictionary:
    :param tt:
    :return:
    """
    start_time = pulse_profile_dictionary["start_time"]
    end_time = (
        start_time
        + pulse_profile_dictionary["rise_time"]
        + pulse_profile_dictionary["flat_time"]
        + pulse_profile_dictionary["fall_time"]
    )

    this_pulse = 0

    if (tt > start_time) and (tt < end_time):
        rise_time = pulse_profile_dictionary["rise_time"]
        flat_time = pulse_profile_dictionary["flat_time"]
        fall_time = pulse_profile_dictionary["fall_time"]
        end_rise = start_time + rise_time
        end_flat = start_time + rise_time + flat_time

        if tt <= end_rise:
            normalized_time = (tt - start_time) / rise_time
            this_pulse = _get_rise_fall_coeff_(normalized_time)

        elif (tt > end_rise) and (tt < end_flat):
            this_pulse = 1.0

        elif tt >= end_flat:
            normalized_time = (tt - end_flat) / fall_time
            this_pulse = 1.0 - _get_rise_fall_coeff_(normalized_time)

        this_pulse *= pulse_profile_dictionary["a0"]

    return this_pulse


def start_run(all_params, pulse_dictionary, diagnostics, name="test", mlflow_path=None):
    """
    End to end mlflow and xarray storage!!


    :param temp_path:
    :param nx:
    :param nv:
    :param nt:
    :param tmax:
    :param nu:
    :param pulse_dictionary:
    :param diagnostics:
    :param name:
    :param mlflow_path:
    :return:
    """
    if mlflow_path is None:
        mlflow_client = mlflow.tracking.MlflowClient()
    else:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_path)

    mlflow.set_experiment(name)

    with mlflow.start_run():
        # Log desired parameters
        params_to_log_dict = {}
        for param in diagnostics.params_to_log:
            if param in ["a0", "k0", "w0"]:
                params_to_log_dict[param] = pulse_dictionary["first pulse"][param]
            else:
                params_to_log_dict[param] = all_params[param]

        mlflow.log_params(params_to_log_dict)

        # Initialize machinery
        nx = all_params["nx"]
        nv = all_params["nv"]
        nu = all_params["nu"]
        tmax = all_params["tmax"]
        nt = all_params["nt"]

        # Distribution function
        f = step.initialize(nx, nv)

        # Spatial Grid
        # Fixed to single wavenumber domains
        xmax = all_params["xmax"]
        xmin = all_params["xmin"]
        dx = (xmax - xmin) / nx
        x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

        # Velocity grid
        vmax = all_params["vmax"]
        dv = 2 * vmax / nv
        v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
        kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

        t = np.linspace(0, tmax, nt)
        dt = t[1] - t[0]

        def driver_function(x, tt):
            total_field = np.zeros_like(x)

            for this_pulse in list(pulse_dictionary.keys()):
                kk = pulse_dictionary[this_pulse]["k0"]
                ww = pulse_dictionary[this_pulse]["w0"]

                envelope = get_pulse_coefficient(
                    pulse_profile_dictionary=pulse_dictionary[this_pulse], tt=tt
                )

                if np.abs(envelope) > 0.0:
                    total_field += envelope * np.cos(kk * x + ww * tt)

            return total_field

        e = field.get_total_electric_field(
            driver_function(x=x, tt=t[0]), f=f, dv=dv, kx=kx
        )

        # Storage
        temp_path = os.path.join(os.getcwd(), "temp-" + str(uuid.uuid4())[-6:])
        os.makedirs(temp_path, exist_ok=True)

        if nt // 4 < 100:
            t_store = 100
        else:
            t_store = nt // 4

        temp_field_store = np.zeros([t_store, nx])
        temp_driver_store = np.zeros([t_store, nx])
        temp_dist_store = np.zeros([t_store, nx, nv])
        temp_t_store = np.zeros(t_store)
        it_store = 0
        storage_manager = storage.StorageManager(x, v, t, temp_path)
        storage_manager.write_parameters_to_file(all_params, "all_parameters")
        storage_manager.write_parameters_to_file(pulse_dictionary, "pulses")

        # Matrix representing collision operator
        A = lenard_bernstein.make_philharmonic_matrix(
            vax=v, nv=nv, nu=nu, dt=dt, dv=dv, v0=1.0
        )

        # Time Loop
        for it in range(nt):
            e, f = step.full_leapfrog_ps_step(
                f, x, kx, v, kv, dv, t[it], dt, e, driver_function
            )

            if nu > 0.0:
                for ix in range(nx):
                    f[ix,] = lenard_bernstein.take_collision_step(leftside=A, f=f[ix])

            # All storage stuff here
            temp_t_store[it_store] = t[it]
            temp_dist_store[it_store] = f
            temp_field_store[it_store] = e
            temp_driver_store[it_store] = driver_function(x=x, tt=t[it])

            it_store += 1

            if it_store == t_store:
                storage_manager.batched_write_to_file(
                    temp_t_store, temp_field_store, temp_driver_store, temp_dist_store
                )
                it_store = 0

        # Diagnostics
        diagnostics(storage_manager)

        # Log
        storage_manager.close()
        mlflow.log_artifacts(temp_path)

        # Cleanup
        shutil.rmtree(temp_path)
