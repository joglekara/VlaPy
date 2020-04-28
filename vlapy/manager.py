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

import os
import uuid

import shutil
from tqdm import tqdm
import mlflow
import numpy as np

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


def start_run(all_params, pulse_dictionary, diagnostics, name="test"):
    """
    This is the highest level function that calls the time integration loop

    MLFlow is initialized here.
    Domain configuration is also performed here.
    All file storage is initialized here.

    :param all_params:
    :param pulse_dictionary:
    :param diagnostics:
    :param name:
    :param mlflow_path:
    :return:
    """

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
        one_over_kx = np.zeros_like(kx)
        one_over_kx[1:] = 1.0 / kx[1:]

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
                    total_field += envelope * np.cos(kk * x - ww * tt)

            return total_field

        e = field.get_total_electric_field(
            driver_function(x=x, tt=t[0]), f=f, dv=dv, one_over_kx=one_over_kx
        )

        # Storage
        temp_path = os.path.join(os.getcwd(), "temp-" + str(uuid.uuid4())[-6:])
        os.makedirs(temp_path, exist_ok=True)
        storage_manager = storage.StorageManager(
            x, v, t, temp_path, store_f=diagnostics.f_rules
        )
        storage_manager.write_parameters_to_file(all_params, "all_parameters")
        storage_manager.write_parameters_to_file(pulse_dictionary, "pulses")

        # Matrix representing collision operator
        leftside = lenard_bernstein.make_philharmonic_matrix(
            vax=v, nv=nv, nu=nu, dt=dt, dv=dv, v0=1.0
        )

        # Time Loop
        for it in tqdm(range(nt)):
            e, f = step.full_PEFRL_ps_step(
                f, x, kx, one_over_kx, v, kv, dv, t[it], dt, e, driver_function
            )

            if nu > 0.0:
                f = lenard_bernstein.take_collision_step(leftside=leftside, f=f,)

            # All storage stuff here
            storage_manager.temp_update(
                tt=t[it], f=f, e=e, driver=driver_function(x=x, tt=t[it])
            )

        # Diagnostics
        diagnostics(storage_manager)

        # Log
        storage_manager.close()
        mlflow.log_artifacts(temp_path)

        # Cleanup
        shutil.rmtree(temp_path)
