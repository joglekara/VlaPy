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

from vlapy.diagnostics import z_function
from vlapy.core import field_driver


def initialize_distribution(nx, nv, vmax=6.0):
    """
    Initializes a Maxwell-Boltzmann distribution

    TODO: temperature and density pertubations

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return:
    """

    f = np.zeros([nx, nv], dtype=np.float64)
    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    for ix in range(nx):
        f[ix,] = np.exp(-(vax ** 2.0) / 2.0)

    # normalize
    f = f / np.trapz(f, dx=dv, axis=1)[:, None]

    return f


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


def log_initial_conditions(all_params, pulse_dictionary):
    """
    This function logs initial conditions to the mlflow server.

    :param all_params:
    :param pulse_dictionary:
    :return:
    """
    params_to_log_dict = {}
    # for param in diagnostics.params_to_log:
    #     if param in ["a0", "k0", "w0"]:
    #         params_to_log_dict[param] = pulse_dictionary["first pulse"][param]
    #     else:
    #         params_to_log_dict[param] = all_params[param]

    for key, val in all_params.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                params_to_log_dict[key + "-" + sub_key] = sub_val
        else:
            params_to_log_dict[key] = val

    for key, val in pulse_dictionary.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                params_to_log_dict[key + "-" + sub_key] = sub_val
        else:
            params_to_log_dict[key] = val

    mlflow.log_params(params_to_log_dict)


def get_everything_ready_for_time_loop(
    diagnostics, all_params, pulse_dictionary, overall_num_steps
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
        # diagnostics=diagnostics,
    )

    # Initialize machinery
    # Distribution function
    f = initialize_distribution(
        nx=all_params["nx"], nv=all_params["nv"], vmax=all_params["vmax"]
    )

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
    t = dt * np.arange(overall_num_steps)

    # Field Driver
    driver_function = field_driver.get_driver_function(
        x=x, pulse_dictionary=pulse_dictionary
    )
    driver_array = field_driver.get_driver_array_using_function(
        x, t, pulse_dictionary=pulse_dictionary
    )

    everything_for_time_loop = {
        "e": np.zeros(x.size),
        "f": f,
        "nx": all_params["nx"],
        "kx": kx,
        "x": x,
        "one_over_kx": one_over_kx,
        "v": v,
        "kv": kv,
        "nv": all_params["nv"],
        "dv": dv,
        "driver": driver_array,
        "driver_function": driver_function,
        "dt": dt,
        "nu": all_params["nu"],
        "t": t,
        "rules_to_store_f": diagnostics.rules_to_store_f,
        "vlasov-poisson": all_params["vlasov-poisson"],
        "fokker-planck": all_params["fokker-planck"],
    }

    return everything_for_time_loop


def make_default_params_dictionary():
    """
    Return a dictionary of default parameters

    :return:
    """
    all_params_dict = {
        "nx": 64,
        "xmin": 0.0,
        "nv": 1024,
        "vmax": 6.4,
        "nt": 1000,
        "tmax": 100,
        "fokker-planck": {"type": "lb", "solver": "batched_tridiagonal",},
        "vlasov-poisson": {
            "time": "leapfrog",
            "vdfdx": "exponential",
            "edfdv": "exponential",
            "poisson": "spectral",
        },
        "backend": {"core": "numpy", "max_doubles_per_file": int(1e8),},
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
