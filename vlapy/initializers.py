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


def initialize_distribution(nx, nv, vmax=6.0):
    """
    Initializes a Maxwell-Boltzmann distribution

    TODO: temperature and density pertubations

    :param nx: (int) size of grid in x
    :param nv: (int) size of grid in v
    :param vmax: (float) maximum absolute value of v
    :return: (float array (nx, nv)) distribution function initialized as a Maxwell-Boltzmann
    """

    f = np.zeros([nx, nv], dtype=np.float64)
    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    for ix in range(nx):
        f[
            ix,
        ] = np.exp(-(vax ** 2.0) / 2.0)

    # normalize
    f = f / np.trapz(f, dx=dv, axis=1)[:, None]

    return f


def initialize_velocity_quantities(vmax, nv):
    """
    This function initializes the velocity grid and related quantities

    :param vmax: (float) the absolute value of the maximum velocity
    :param nv: (int) the size of the grid in v
    :return: (float, float array (nv, ), float array (nv, )) the grid spacing, the velocity grid,
    and the corresponding velocity wave-number grid
    """
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    return dv, v, kv


def initialize_spatial_quantities(xmin, xmax, nx):
    """
    This function initializes the spatial grid and related quantities

    :param xmin: (float) the minimum value of x represented by the simulation
    :param xmax: (float) the maximum value of x represented by the simulation
    :param nx: (int) the size of the x-grid
    :return: (float, float array (nx, ), float array (nx, )) the grid spacing, the spatial grid,
    and the corresponding spatial wave-number grid

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

    :param all_params: (dictionary) contains the provided input parameters to the simulation
    :param pulse_dictionary: (dictionary) contains the derived input parameters to the simulation
    """
    params_to_log_dict = {}

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


def make_default_params_dictionary():
    """
    Return a dictionary of default parameters

    :return: (dictionary) Contains the default parameters for VlaPy
    """

    all_params_dict = {
        "nx": 64,
        "xmin": 0.0,
        "nv": 512,
        "vmax": 6.4,
        "nt": 500,
        "tmax": 80,
        "fokker-planck": {
            "bool": False,
            "type": "lb",
            "solver": "batched_tridiagonal",
        },
        "vlasov-poisson": {
            "time": "leapfrog",
            "vdfdx": "exponential",
            "edfdv": "exponential",
            "poisson": "spectral",
        },
        "backend": {
            "core": "numpy",
            "max_GB_for_device": int(1),
        },
        "a0": 1e-7,
        "resume": {
            "bool": False,
            "experiment_name": "None",
            "run_id": "None",
            "t0": 0.0,
        },
    }

    return all_params_dict


def specify_collisions_to_dict(log_nu_over_nu_ld, all_params_dict):
    """
    This function adds the collision frequency to the parameters dictionary

    :param log_nu_over_nu_ld: (float) log( nu_ee / nu_ld )
    :param all_params_dict: (dictionary) contains the input parameters for the simulation
    :return: (dictionary) parameters updated with the desired collision frequency
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

    :param k0: (float) the desired EPW wavenumber
    :param all_params_dict: (dictionary) contains the input parameters for the simulation
    :return: (dictionary) parameters updated with the solution to the dispersion relation
    for the desired wavenumber. This is typically used by the ponderomotive force driver.
    """
    solution_to_dispersion_relation = z_function.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=k0
    )

    all_params_dict["w_epw"] = np.real(solution_to_dispersion_relation)
    all_params_dict["nu_ld"] = np.imag(solution_to_dispersion_relation)
    all_params_dict["xmax"] = 2.0 * np.pi / k0
    all_params_dict["v_ph"] = all_params_dict["w_epw"] / k0

    return all_params_dict
