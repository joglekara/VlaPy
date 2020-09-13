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
from vlapy.diagnostics import nlepw

if __name__ == "__main__":
    k0 = np.random.uniform(0.3, 0.4, 1)[0]
    log_nu_over_nu_ld = -4

    all_params_dict = initializers.make_default_params_dictionary()
    all_params_dict = initializers.specify_epw_params_to_dict(
        k0=k0, all_params_dict=all_params_dict
    )
    all_params_dict = initializers.specify_collisions_to_dict(
        log_nu_over_nu_ld=log_nu_over_nu_ld, all_params_dict=all_params_dict
    )

    all_params_dict["nx"] = 256
    all_params_dict["nv"] = 2048
    all_params_dict["tmax"] = 1000
    all_params_dict["nt"] = all_params_dict["tmax"] * 4

    all_params_dict["vlasov-poisson"]["time"] = "leapfrog"
    all_params_dict["vlasov-poisson"]["edfdv"] = "exponential"
    all_params_dict["vlasov-poisson"]["vdfdx"] = "exponential"

    pulse_dictionary = {
        "first pulse": {
            "start_time": 0,
            "t_L": 6,
            "t_wL": 2.5,
            "t_R": 25,
            "t_wR": 2.5,
            "w0": all_params_dict["w_epw"],
            "a0": 4e-2,
            "k0": k0,
        }
    }

    mlflow_exp_name = "non-linear electron plasma wave"

    uris = {
        "tracking": "local",
    }

    that_run = manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=nlepw.NLEPW(
            vph=all_params_dict["v_ph"],
            wepw=all_params_dict["w_epw"],
        ),
        uris=uris,
        name=mlflow_exp_name,
    )
