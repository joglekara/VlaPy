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
from datetime import datetime


import numpy as np

from vlapy import manager
from diagnostics import landau_damping, z_function

if __name__ == "__main__":

    all_params_dict = {
        "nx": 48,
        "xmin": 0.0,
        "xmax": 2.0 * np.pi / 0.3,
        "nv": 512,
        "vmax": 6.0,
        "nt": 1000,
        "tmax": 100,
        "nu": 0.0,
    }

    pulse_dictionary = {
        "first pulse": {
            "start_time": 0,
            "rise_time": 5,
            "flat_time": 10,
            "fall_time": 5,
            "a0": 1e-6,
            "k0": 0.3,
        }
    }

    params_to_log = ["w0", "k0", "a0"]

    pulse_dictionary["first pulse"]["w0"] = np.real(
        z_function.get_roots_to_electrostatic_dispersion(
            wp_e=1.0, vth_e=1.0, k0=pulse_dictionary["first pulse"]["k0"]
        )
    )

    mlflow_exp_name = "Landau Damping-test"

    print("Starting VlaPy at " + datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    print("MLFlow experiment name: " + mlflow_exp_name)
    print("mlruns folder located at " + os.getcwd())
    print("Run parameters: ")
    print(all_params_dict)
    print()
    print(
        "run `mlflow ui` at the command line, and go "
        "to `http://localhost:5000` in your browser to "
        "view the results"
    )

    manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(params_to_log),
        name=mlflow_exp_name,
    )
