# MIT License
#
# Copyright (c) 2020 Archis Joglekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import mlflow
import numpy as np

from diagnostics import z_function
from vlapy.core import step, field, field_driver


def initialize_velocity_quantities(vmax, nv):
    """
    This function initializes the velocity grid and related quantities

    :param vmax:
    :param nv:
    :return:
    """
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    return dv, v, kv


def initialize_spatial_quantities(xmin, xmax, nx):
    """
    This function initializes the spatial grid and related quantities

    :param xmin:
    :param xmax:
    :param nx:
    :return:
    """
    dx = (xmax - xmin) / nx
    x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi
    one_over_kx = np.zeros_like(kx)
    one_over_kx[1:] = 1.0 / kx[1:]

    return dx, x, kx, one_over_kx


def log_initial_conditions(all_params, pulse_dictionary, diagnostics):
    """
    This function logs initial conditions to the mlflow server.

    :param all_params:
    :param pulse_dictionary:
    :param diagnostics:
    :return:
    """
    params_to_log_dict = {}
    for param in diagnostics.params_to_log:
        if param in ["a0", "k0", "w0"]:
            params_to_log_dict[param] = pulse_dictionary["first pulse"][param]
        else:
            params_to_log_dict[param] = all_params[param]

    mlflow.log_params(params_to_log_dict)


def get_everything_ready_for_time_loop(
    diagnostics, all_params, pulse_dictionary, num_steps
):
    """
    In order to keep the main time loop clean, this function handles all the
    necessary initialization and array creation for the main simulation time loop.

    Initialized here:
    spatial grid
    velocity grid
    distribution function
    time grid
    driver array

    :param diagnostics:
    :param all_params:
    :param pulse_dictionary:
    :param num_steps:
    :return:
    """
    # Log desired parameters
    log_initial_conditions(
        all_params=all_params,
        pulse_dictionary=pulse_dictionary,
        diagnostics=diagnostics,
    )

    # Initialize machinery
    # Distribution function
    f = step.initialize(all_params["nx"], all_params["nv"])

    # Spatial Grid
    dx, x, kx, one_over_kx = initialize_spatial_quantities(
        xmin=all_params["xmin"], xmax=all_params["xmax"], nx=all_params["nx"]
    )

    # Velocity grid
    dv, v, kv = initialize_velocity_quantities(
        vmax=all_params["vmax"], nv=all_params["nv"]
    )

    t_dummy = np.linspace(0, all_params["tmax"], all_params["nt"])
    dt = t_dummy[1] - t_dummy[0]
    t = dt * np.arange(num_steps)

    # Field Driver
    driver_array = field_driver.get_driver_array_after_defining_function(
        x, t, num_steps, pulse_dictionary=pulse_dictionary
    )
    e = field.get_total_electric_field(
        driver_array[0], f=f, dv=dv, one_over_kx=one_over_kx
    )

    everything_for_time_loop = {
        "e": e,
        "f": f,
        "electric_field": e,
        "nx": all_params["nx"],
        "kx": kx,
        "x": x,
        "one_over_kx": one_over_kx,
        "v": v,
        "kv": kv,
        "nv": all_params["nv"],
        "dv": dv,
        "driver": driver_array,
        "dt": dt,
        "nu": all_params["nu"],
        "t": t,
        "vlasov-poisson": all_params["vlasov-poisson"],
    }

    return everything_for_time_loop


def get_jax_arrays_for_time_loop(stuff_for_time_loop, nt_in_loop, store_f_rules):
    """
    This function converts the previously created NumPy arrays to JAX arrays.

    It also creates the temporary storage for e and f that is used for the
    low-level JAX time-loop

    :param stuff_for_time_loop:
    :param nt_in_loop:
    :param store_f_rules:
    :return:
    """

    # This is where we initialize the right distribution function storage array
    # If we're saving all the "x" values then that array is created
    if store_f_rules == "all-x":
        store_f = np.zeros(
            (nt_in_loop,) + stuff_for_time_loop["f"].shape, dtype=jnp.complex64
        )
        store_f[0,] = stuff_for_time_loop["f"]

    # If we're saving the first M spatial-modes then that array is created
    elif isinstance(store_f_rules["space"], list) and store_f_rules["space"][0] == "k0":
        store_f = np.zeros(
            (nt_in_loop, len(store_f_rules), stuff_for_time_loop["f"].shape[1]),
            dtype=np.complex64,
        )

        store_f[0,] = np.fft.fft(stuff_for_time_loop["f"], axis=0)[: len(store_f_rules)]

    else:
        raise NotImplementedError

    # This is where we return the whole simulation configuration dictionary
    return {
        "e": jnp.array(stuff_for_time_loop["e"]),
        "f": jnp.array(stuff_for_time_loop["f"]),
        "stored_e": jnp.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
        "stored_f": jnp.array(store_f),
        "x": jnp.array(stuff_for_time_loop["x"]),
        "kx": jnp.array(stuff_for_time_loop["kx"]),
        "one_over_kx": jnp.array(stuff_for_time_loop["one_over_kx"]),
        "v": jnp.array(stuff_for_time_loop["v"]),
        "kv": jnp.array(stuff_for_time_loop["kv"]),
        "dv": jnp.array(stuff_for_time_loop["dv"]),
        "t": jnp.array(stuff_for_time_loop["t"]),
        "dt": stuff_for_time_loop["dt"],
        "vlasov-poisson": stuff_for_time_loop["vlasov-poisson"],
    }


def make_default_params_dictionary():
    """
    Return a dictionary of default parameters

    :return:
    """
    all_params_dict = {
        "nx": 64,
        "xmin": 0.0,
        "nv": 1024,
        "vmax": 6.0,
        "nt": 1000,
        "tmax": 100,
        "collision operator": "lb",
        "vlasov-poisson": "leapfrog",
        "a0": 1e-7,
    }

    return all_params_dict


def specify_collisions_to_dict(log_nu_over_nu_ld, all_params_dict):
    """
    This function adds the collision frequency to the parameters dictionary

    :param log_nu_over_nu_ld:
    :param all_params_dict:
    :return:
    """
    if log_nu_over_nu_ld is None:
        all_params_dict["nu"] = 0.0
    else:
        all_params_dict["nu"] = (
            np.abs(all_params_dict["nu_ld"]) * 10 ** log_nu_over_nu_ld
        )

    return all_params_dict


def specify_epw_params_to_dict(k0, all_params_dict):
    """
    This function calculates the roots to the dispersion relation for EPWs for a
    particular k0. This stores the `w_epw` and `nu_ld` for that EPW in the
    `all_params_dict` as well as dictates the size of the box through `xmax`.

    :param k0:
    :param all_params_dict:
    :return:
    """
    solution_to_dispersion_relation = z_function.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=k0
    )

    all_params_dict["w_epw"] = np.real(solution_to_dispersion_relation)
    all_params_dict["nu_ld"] = np.imag(solution_to_dispersion_relation)
    all_params_dict["xmax"] = 2.0 * np.pi / k0
    all_params_dict["v_ph"] = all_params_dict["w_epw"] / k0

    return all_params_dict
