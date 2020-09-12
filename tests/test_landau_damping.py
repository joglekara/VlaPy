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
import pytest

from vlapy import manager, initializers
from vlapy.infrastructure import mlflow_helpers, print_to_screen
from vlapy.diagnostics import landau_damping

ALL_TIME_INTEGRATORS = ["leapfrog", "pefrl", "h-sixth"]
ALL_VDFDX_INTEGRATORS = ["exponential"]
ALL_EDFDV_INTEGRATORS = ["exponential", "cd2"]

ALL_VDFDX_INTEGRATORS_FOR_FAST_TESTING = ["exponential", "sl"]
ALL_EDFDV_INTEGRATORS_FOR_FAST_TESTING = ["exponential", "cd2", "sl"]
ALL_BACKENDS = ["numpy"]


def __run_integrated_landau_damping_test_and_return_damping_rate__(
    k0,
    fp_type,
    time_integrator,
    edfdv_integrator,
    vdfdx_integrator,
    backend,
):
    """
    This is the fully integrated flow for a Landau damping run

    :param k0:
    :param log_nu_over_nu_ld:
    :return:
    """
    all_params_dict = initializers.make_default_params_dictionary()
    all_params_dict = initializers.specify_epw_params_to_dict(
        k0=k0, all_params_dict=all_params_dict
    )
    all_params_dict = initializers.specify_collisions_to_dict(
        log_nu_over_nu_ld=None, all_params_dict=all_params_dict
    )
    all_params_dict["backend"]["core"] = backend

    try:
        all_params_dict["vlasov-poisson"]["time"] = time_integrator
        all_params_dict["vlasov-poisson"]["edfdv"] = edfdv_integrator
        all_params_dict["vlasov-poisson"]["vdfdx"] = vdfdx_integrator
        all_params_dict["fokker-planck"]["type"] = fp_type

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

        mlflow_exp_name = "vlapy-unit-test"

        uris = {
            "tracking": "local",
        }

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

        return (
            mlflow_helpers.get_this_metric_of_this_run("damping_rate", that_run),
            all_params_dict["nu_ld"],
        )
    except NotImplementedError:
        return 0.0, 0.0


# @pytest.mark.parametrize("vdfdx_integrator", ALL_VDFDX_INTEGRATORS)
# @pytest.mark.parametrize("edfdv_integrator", ALL_EDFDV_INTEGRATORS)
@pytest.mark.parametrize("vdfdx_integrator", ALL_VDFDX_INTEGRATORS_FOR_FAST_TESTING)
@pytest.mark.parametrize("edfdv_integrator", ALL_EDFDV_INTEGRATORS_FOR_FAST_TESTING)
@pytest.mark.parametrize("time_integrator", ALL_TIME_INTEGRATORS)
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_landau_damping(vdfdx_integrator, edfdv_integrator, time_integrator, backend):
    """
    Tests Landau Damping for a random wavenumber for a given combination of integrators

    :return:
    """
    rand_k0 = 0.3  # np.random.uniform(0.25, 0.4, 1)[0]

    (
        measured_rate,
        actual_rate,
    ) = __run_integrated_landau_damping_test_and_return_damping_rate__(
        k0=rand_k0,
        fp_type="None",
        time_integrator=time_integrator,
        vdfdx_integrator=vdfdx_integrator,
        edfdv_integrator=edfdv_integrator,
        backend=backend,
    )

    np.testing.assert_almost_equal(measured_rate, actual_rate, decimal=4)
