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
import numpy as np

from vlapy import manager
from diagnostics import landau_damping


def test_manager_folder():
    test_folder = os.path.join(os.getcwd(), str(uuid.uuid4())[-6:])
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
            "w0": 1.1598,
        }
    }

    params_to_log = ["k0"]

    manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(params_to_log),
        name="Landau Damping",
        mlflow_path=test_folder,
    )

    assert os.path.exists(test_folder)
    shutil.rmtree(test_folder)
