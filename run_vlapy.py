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

import numpy as np

from vlapy import manager, initializers
from vlapy.infrastructure import mlflow_helpers, print_to_screen
from vlapy.diagnostics import landau_damping

if __name__ == "__main__":
    k0 = np.random.uniform(0.3, 0.4, 1)[0]
    log_nu_over_nu_ld = None

    all_params_dict = initializers.make_default_params_dictionary()
    all_params_dict = initializers.specify_epw_params_to_dict(
        k0=k0, all_params_dict=all_params_dict
    )
    all_params_dict = initializers.specify_collisions_to_dict(
        log_nu_over_nu_ld=log_nu_over_nu_ld, all_params_dict=all_params_dict
    )

    all_params_dict["vlasov-poisson"]["time"] = "leapfrog"
    all_params_dict["vlasov-poisson"]["edfdv"] = "exponential"
    all_params_dict["vlasov-poisson"]["vdfdx"] = "exponential"

    all_params_dict["backend"]["core"] = "numpy"
    all_params_dict["backend"]["max_GB_for_device"] = 1

    all_params_dict["fokker-planck"]["type"] = "lb"
    all_params_dict["fokker-planck"]["type"] = "batched_tridiagonal"


    pulse_dictionary = {
        "first pulse": {
            "start_time": 0,
            "t_L": 6,
            "t_wL": 2.5,
            "t_R": 20,
            "t_wR": 2.5,
            "w0": all_params_dict["w_epw"],
            "a0": 1e-7,
            "k0": k0,
        }
    }

    mlflow_exp_name = "landau-damping"

    uris = {
        "tracking": "local",
    }

    print_to_screen.print_startup_message(
        mlflow_exp_name, all_params_dict, pulse_dictionary
    )

    that_run = manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(
            vph=all_params_dict["v_ph"],
            wepw=all_params_dict["w_epw"],
        ),
        uris=uris,
        name=mlflow_exp_name,
    )

    print(
        mlflow_helpers.get_this_metric_of_this_run("damping_rate", that_run),
        all_params_dict["nu_ld"],
    )
